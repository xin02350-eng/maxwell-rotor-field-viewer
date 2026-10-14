"""Controlled POD/MLP experiments; this module never imports archive programs.

Cache arrays: features[C,T,F], field[C,T,D], coeff[C,T,R], basis[D,R],
mean[D], scale[R], case_ids[C], point_id[P], xy[P,2]. Field components,
when present, must be flattened point-major. POD/scale must be fitted upstream
using training cases only. train() touches labels from train/validation only.

train(...) returns {model, model_path, summary, history}. predict(model, x)
returns reconstructed fields; predict(..., return_coeff=True) returns raw POD
coefficients. Models may be dictionaries or saved NPZ paths. NumPy prediction
does not need PyTorch; training imports PyTorch lazily. No code runs on import.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

ARCHITECTURES = ((64, 64), (128, 128), (64, 64, 64), (128, 128, 128))
REFERENCE_ARCHITECTURE = (32, 32)
REQUIRED = ("features", "field", "basis", "mean", "coeff", "case_ids",
            "point_id", "xy", "scale")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   default=_jsonable, allow_nan=False) + "\n",
                          encoding="utf-8")


def load_cache(path):
    """Load only safe arrays; validation does not evaluate any test labels."""
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED) - set(archive.files)
        if missing:
            raise ValueError(f"Missing cache arrays: {sorted(missing)}")
        cache = {key: archive[key] for key in archive.files}
    _validate_cache(cache)
    cache["_source_path"] = str(Path(path).resolve())
    return cache


def _validate_cache(cache):
    for key in REQUIRED:
        if key not in cache:
            raise ValueError(f"Missing cache array {key}")
    x, y, a, u = (np.asarray(cache[k]) for k in
                  ("features", "field", "coeff", "basis"))
    if x.ndim != 3 or y.ndim != 3 or a.ndim != 3 or u.ndim != 2:
        raise ValueError("features, field, coeff must be 3D; basis must be 2D")
    c, t, f = x.shape
    d, r = u.shape
    if not all(v > 0 for v in (c, t, f, d, r)):
        raise ValueError("Empty cache dimension")
    if y.shape != (c, t, d) or a.shape != (c, t, r):
        raise ValueError("Case/time/field/POD dimensions do not agree")
    if np.asarray(cache["mean"]).shape != (d,) or np.asarray(cache["scale"]).shape != (r,):
        raise ValueError("mean/scale shape mismatch")
    if np.asarray(cache["case_ids"]).shape != (c,):
        raise ValueError("case_ids must have one entry per case")
    if len(np.unique(cache["case_ids"])) != c:
        raise ValueError("case_ids must be unique")
    ids, xy = np.asarray(cache["point_id"]), np.asarray(cache["xy"])
    if ids.ndim != 1 or len(ids) == 0 or d % len(ids) or xy.shape != (len(ids), 2):
        raise ValueError("point_id/xy dimensions do not match point-major fields")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("point_id must be unique")
    for key in ("basis", "mean", "scale", "xy"):
        if not np.isfinite(cache[key]).all():
            raise ValueError(f"Nonfinite {key}")
    if np.any(np.asarray(cache["scale"]) <= 0):
        raise ValueError("Every coefficient scale must be positive")


def _split_indices(split, c):
    if isinstance(split, (str, os.PathLike)):
        split = json.loads(Path(split).read_text(encoding="utf-8"))
    groups = {}
    for key in ("train", "validation", "test"):
        raw = np.asarray(split.get(key, []))
        if raw.size and (raw.ndim != 1 or raw.dtype.kind not in "iu"):
            raise ValueError(f"{key} must contain integer case indices")
        values = raw.astype(np.int64).reshape(-1)
        if len(np.unique(values)) != len(values) or np.any(values < 0) or np.any(values >= c):
            raise ValueError(f"Invalid/duplicate {key} case indices")
        groups[key] = values
    if not len(groups["train"]):
        raise ValueError("Training cases are required")
    all_groups = np.concatenate(list(groups.values()))
    if len(np.unique(all_groups)) != len(all_groups):
        raise ValueError("Train, validation, and test case sets must be disjoint")
    return groups


def _samples(cases, times, selected, total_cases, name):
    if selected is None:
        return (cases[:, None] * times + np.arange(times)).reshape(-1)
    raw = np.asarray(selected)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} sample indices must be integers")
    if raw.ndim == 2 and raw.shape[1] == 2:
        if np.any(raw[:, 1] < 0) or np.any(raw[:, 1] >= times):
            raise ValueError(f"{name} time index out of bounds")
        indices = raw[:, 0] * times + raw[:, 1]
    elif raw.ndim == 1:
        indices = raw
    else:
        raise ValueError(f"{name} sample indices must be flattened or [n,2]")
    indices = indices.astype(np.int64)
    if (np.any(indices < 0) or np.any(indices >= total_cases * times)
            or not np.isin(indices // times, cases).all()
            or len(np.unique(indices)) != len(indices)):
        raise ValueError(f"{name} samples must be unique members of their split")
    return indices


def _expand_point_weights(weights, cache):
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64)
    d, p = len(cache["mean"]), len(cache["point_id"])
    if w.shape == (p,):
        w = np.repeat(w, d // p)
    if w.shape != (d,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("point_weights must be nonnegative [points] or [field_dim], with positive sum")
    return w / w.mean()


def weighted_pod_terms(field, basis, mean, coeff, weights, batch_size=256):
    """Exact spatial weighted loss terms, including POD residual cross term.

    Let delta=prediction_coeff-coeff and residual=field-mean-coeff@basis.T.
    Weighted SSE = delta Q delta - 2 delta cross + floor. This also works
    when the supplied POD is only approximately orthonormal in floating point.
    """
    u = np.asarray(basis, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    a = np.asarray(coeff, dtype=np.float64)
    if w.shape != (u.shape[0],) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid expanded field weights")
    q = u.T @ (w[:, None] * u)
    cross = np.empty_like(a)
    floor = np.empty(len(a), dtype=np.float64)
    target_sse = 0.0
    for lo in range(0, len(a), batch_size):
        hi = min(lo + batch_size, len(a))
        y = np.asarray(field[lo:hi], dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite train/validation field labels")
        residual = y - mu - a[lo:hi] @ u.T
        cross[lo:hi] = (residual * w) @ u
        floor[lo:hi] = np.einsum("ij,j,ij->i", residual, w, residual)
        target_sse += float(np.einsum("ij,j,ij->", y, w, y))
    return dict(gram=q, cross=cross, floor=floor, weight_sum=float(w.sum()),
                target_sse=target_sse)


def weighted_field_loss(pred_coeff, true_coeff, gram, cross, floor, weight_sum):
    """Torch-compatible differentiable exact weighted field MSE, per sample.

    Inputs must be tensors with matching dtype/device. Returning each sample's
    loss lets callers use a documented importance-sampling correction.
    """
    delta = pred_coeff - true_coeff
    return (((delta @ gram) * delta).sum(dim=1)
            - 2 * (delta * cross).sum(dim=1) + floor) / weight_sum


def _model_dict(model):
    if isinstance(model, (str, os.PathLike)):
        with np.load(model, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    return model


def _predict_coeff(model, x):
    a = (np.asarray(x, dtype=np.float64) - model["feature_mean"]) / model["feature_scale"]
    n = int(np.asarray(model["layer_count"]).item())
    for i in range(n):
        a = a @ model[f"w{i}"] + model[f"b{i}"]
        if i + 1 < n:
            a = np.tanh(a)
    a = a * model["scale"]
    if int(np.asarray(model.get("has_base", 0)).item()):
        base = {key[6:]: value for key, value in model.items() if key.startswith("base__")}
        a = a + _predict_coeff(base, x)
    return a


def predict(model, features, return_coeff=False, batch_size=1024):
    """NumPy inference, preserving all feature leading dimensions."""
    model = _model_dict(model)
    x = np.asarray(features)
    if x.ndim < 1 or x.shape[-1] != len(model["feature_mean"]) or not np.isfinite(x).all():
        raise ValueError("Invalid feature dimensions/values")
    shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    dim = len(model["scale"]) if return_coeff else len(model["mean"])
    out = np.empty((len(flat), dim), dtype=np.float64)
    for lo in range(0, len(flat), batch_size):
        a = _predict_coeff(model, flat[lo:lo + batch_size])
        out[lo:lo + batch_size] = a if return_coeff else model["mean"] + a @ model["basis"].T
    return out.reshape(*shape, dim)


def _partition(cache, indices, weights, batch_size=256):
    """Copies only the explicitly selected train/validation labels."""
    d, r = cache["basis"].shape
    x = np.asarray(cache["features"]).reshape(-1, cache["features"].shape[-1])[indices]
    a = np.asarray(cache["coeff"]).reshape(-1, r)[indices]
    y = np.asarray(cache["field"]).reshape(-1, d)[indices]
    if not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError("Nonfinite selected train/validation features or coefficients")
    ordinary = weighted_pod_terms(y, cache["basis"], cache["mean"], a,
                                 np.ones(d), batch_size=batch_size)
    weighted = ordinary if weights is None else weighted_pod_terms(
        y, cache["basis"], cache["mean"], a, weights, batch_size=batch_size)
    return dict(features=x, coeff=a.astype(np.float64), ordinary=ordinary,
                weighted=weighted, count=len(indices))


def _metrics(coeff_pred, partition, scale):
    delta = np.asarray(coeff_pred, dtype=np.float64) - partition["coeff"]
    result = {"coefficient_loss": float(np.mean((delta / scale) ** 2)),
              "raw_coefficient_loss": float(np.mean(delta ** 2))}
    for key, prefix in (("ordinary", ""), ("weighted", "weighted_")):
        terms = partition[key]
        # This exact Gram calculation avoids an approximate orthogonality assumption.
        sse = float(np.sum((delta @ terms["gram"]) * delta)
                    - 2 * np.sum(delta * terms["cross"]) + np.sum(terms["floor"]))
        sse = max(sse, 0.0)
        result[prefix + "field_rmse_T"] = math.sqrt(sse / (len(delta) * terms["weight_sum"]))
        result[prefix + "relative_L2_pct"] = (100 * math.sqrt(sse / terms["target_sse"])
                                                if terms["target_sse"] > 0 else None)
    return result


def _train_impl(cache, split, hidden, seed, output_dir, variant="standard", max_epochs=600,
          patience=80, batch_size=256, learning_rate=1e-3, threads=4,
          min_delta=1e-9, sample_weights=None, point_weights=None, base_model=None,
          train_sample_indices=None, validation_sample_indices=None,
          importance_correction=False, normalize_features=True,
          monitor="field_rmse_T", weight_decay=0.0, overwrite=False):
    """Train one fixed architecture; there is no test evaluation in this function.

    sample_weights: [C,T], [C*T], or selected training-sample length; sampling
    uses replacement and N_train draws/epoch. Default changes the training
    distribution; importance_correction=True recovers the uniform objective.
    point_weights: [P] (repeated for each point component) or [D].
    base_model: frozen compatible NPZ/dict, embedded in the resulting NPZ.
    Sample indices may be flat case*T+time or [n,2], and must stay in the split.
    Empty validation is allowed only as a clearly logged training-only check.
    """
    import torch

    start = time.perf_counter()
    if isinstance(cache, (str, os.PathLike)):
        cache = load_cache(cache)
    _validate_cache(cache)
    if max_epochs < 1 or patience < 1 or batch_size < 1 or threads < 1 or learning_rate <= 0:
        raise ValueError("Training budget/patience/batch/threads/lr must be positive")
    if min_delta < 0 or weight_decay < 0:
        raise ValueError("min_delta/weight_decay must be nonnegative")
    hidden = tuple(int(h) for h in hidden)
    if not hidden or min(hidden) < 1:
        raise ValueError("hidden must contain positive layer widths")
    c, times, feature_dim = cache["features"].shape
    field_dim, rank = cache["basis"].shape
    groups = _split_indices(split, c)
    if "pod_train_case_indices" in cache:
        if not np.array_equal(np.sort(cache["pod_train_case_indices"]), np.sort(groups["train"])):
            raise ValueError("Cache POD provenance differs from the requested training cases")
    tr_idx = _samples(groups["train"], times, train_sample_indices, c, "train")
    va_idx = _samples(groups["validation"], times, validation_sample_indices, c, "validation")
    if not len(tr_idx):
        raise ValueError("No selected training samples")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((out / p).exists() for p in ("best_model.npz", "history.csv", "summary.json")):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {out}")
    torch.set_num_threads(int(threads))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    weights = _expand_point_weights(point_weights, cache)
    tr = _partition(cache, tr_idx, weights)
    va = _partition(cache, va_idx, weights) if len(va_idx) else None
    if monitor not in ("field_rmse_T", "weighted_field_rmse_T", "coefficient_loss"):
        raise ValueError("Unsupported checkpoint monitor")
    scale = np.asarray(cache["scale"], dtype=np.float64)
    feature_mean = tr["features"].astype(np.float64).mean(axis=0) if normalize_features else np.zeros(feature_dim)
    feature_scale = tr["features"].astype(np.float64).std(axis=0) if normalize_features else np.ones(feature_dim)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    base = _model_dict(base_model) if base_model is not None else None
    if base is not None:
        for key in ("basis", "mean", "scale", "point_id"):
            if (np.asarray(base[key]).shape != np.asarray(cache[key]).shape
                    or not np.allclose(base[key], cache[key], rtol=1e-6, atol=1e-8)):
                raise ValueError(f"Residual base and cache must share {key}")
    for part in (tr, va):
        if part is not None:
            part["base_coeff"] = (predict(base, part["features"], return_coeff=True)
                                  if base is not None else np.zeros_like(part["coeff"]))
    probabilities = None
    if sample_weights is not None:
        supplied = np.asarray(sample_weights, dtype=np.float64)
        if supplied.shape == (c, times) or supplied.shape == (c * times,):
            supplied = supplied.reshape(-1)[tr_idx]
        elif supplied.shape == (len(groups["train"]), times):
            # Rows follow split['train'] order, even when the case indices are unsorted.
            case_row = {int(case): row for row, case in enumerate(groups["train"])}
            supplied = np.asarray([supplied[case_row[int(i // times)], int(i % times)] for i in tr_idx])
        elif supplied.shape != (len(tr_idx),):
            raise ValueError("sample_weights shape must be [C,T], [C*T], [C_train,T], or selected train length")
        if not np.isfinite(supplied).all() or np.any(supplied < 0) or supplied.sum() <= 0:
            raise ValueError("sample_weights must be finite, nonnegative, and not all zero")
        if importance_correction and np.any(supplied == 0):
            raise ValueError("Uniform-risk importance correction requires positive probability everywhere")
        probabilities = supplied / supplied.sum()
    dims = (feature_dim,) + hidden + (rank,)
    layers = []
    linear_layers = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        layer = torch.nn.Linear(din, dout)
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        linear_layers.append(layer)
        layers.append(layer)
        if i + 1 < len(dims) - 1:
            layers.append(torch.nn.Tanh())
    network = torch.nn.Sequential(*layers)
    if base is not None:
        torch.nn.init.zeros_(linear_layers[-1].weight)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_train = torch.as_tensor(((tr["features"] - feature_mean) / feature_scale).astype(np.float32))
    y_train = torch.as_tensor(((tr["coeff"] - tr["base_coeff"]) / scale).astype(np.float32))
    coeff_train = torch.as_tensor(tr["coeff"].astype(np.float32))
    base_train = torch.as_tensor(tr["base_coeff"].astype(np.float32))
    scale_tensor = torch.as_tensor(scale.astype(np.float32))
    if weights is not None:
        terms = tr["weighted"]
        gram_t = torch.as_tensor(terms["gram"].astype(np.float32))
        cross_t = torch.as_tensor(terms["cross"].astype(np.float32))
        floor_t = torch.as_tensor(terms["floor"].astype(np.float32))
    parameter_count = sum(p.numel() for p in network.parameters())
    metadata = dict(format="pod_mlp_numpy_v1", variant=variant, hidden=list(hidden), seed=int(seed),
                    input_dim=feature_dim, output_rank=rank, field_dim=field_dim,
                    train_case_indices=groups["train"].tolist(), validation_case_indices=groups["validation"].tolist(),
                    train_case_ids=np.asarray(cache["case_ids"])[groups["train"]].tolist(),
                    validation_case_ids=np.asarray(cache["case_ids"])[groups["validation"]].tolist(),
                    excluded_test_case_indices=groups["test"].tolist(), test_labels_accessed=False,
                    selected_training_samples=len(tr_idx), selected_validation_samples=len(va_idx),
                    max_epochs=int(max_epochs), patience=int(patience), min_delta=min_delta,
                    batch_size=int(batch_size), learning_rate=learning_rate, threads=int(threads),
                    optimizer="Adam", activation="tanh", weight_decay=weight_decay,
                    normalize_features=normalize_features, checkpoint_monitor=monitor,
                    checkpoint_partition="validation" if va is not None else "training_only",
                    point_weighted_loss=weights is not None, resampled=probabilities is not None,
                    importance_correction=importance_correction, frozen_residual_base=base is not None,
                    source_cache=cache.get("_source_path"), parameter_count=parameter_count,
                    numpy_version=np.__version__, torch_version=torch.__version__,
                    loss="exact_weighted_field_MSE" if weights is not None else "standardized_coefficient_MSE")
    metadata["basis_sha256"] = hashlib.sha256(np.ascontiguousarray(cache["basis"]).tobytes()).hexdigest()
    metadata["training_sample_indices_sha256"] = hashlib.sha256(tr_idx.tobytes()).hexdigest()
    _write_json(out / "config.json", metadata)
    np.savez_compressed(out / "sample_indices.npz", train=tr_idx, validation=va_idx,
                        sampling_probability=(probabilities if probabilities is not None else np.full(len(tr_idx), 1 / len(tr_idx))))
    if weights is not None:
        np.savez_compressed(out / "point_weights.npz", weights=weights)

    def snapshot():
        model = dict(layer_count=np.asarray(len(linear_layers)), scale=np.asarray(cache["scale"]),
                     basis=np.asarray(cache["basis"]), mean=np.asarray(cache["mean"]),
                     feature_mean=feature_mean, feature_scale=feature_scale,
                     point_id=np.asarray(cache["point_id"]), xy=np.asarray(cache["xy"]),
                     has_base=np.asarray(int(base is not None)),
                     metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, default=_jsonable)))
        for i, layer in enumerate(linear_layers):
            model[f"w{i}"] = layer.weight.detach().numpy().T.copy()
            model[f"b{i}"] = layer.bias.detach().numpy().copy()
        if base is not None:
            model.update({"base__" + key: np.asarray(value) for key, value in base.items()})
        return model

    def evaluate(epoch, grad_norm, grad_max, objective):
        current = snapshot()
        row = dict(epoch=int(epoch), optimizer_step=optimizer_steps,
                   learning_rate=float(optimizer.param_groups[0]["lr"]), grad_norm=float(grad_norm),
                   grad_norm_max=float(grad_max), sampled_training_objective=objective,
                   elapsed_seconds=float(time.perf_counter() - start))
        for part, name in ((tr, "train"), (va, "validation")):
            if part is not None:
                metrics = _metrics(predict(current, part["features"], return_coeff=True), part, scale)
                row.update({name + "_" + k: v for k, v in metrics.items()})
        return current, row

    history, optimizer_steps = [], 0
    best_model, initial = evaluate(0, 0.0, 0.0, None)
    history.append(initial)
    checkpoint_key = ("validation_" if va is not None else "train_") + monitor
    best_score, best_epoch, final_epoch, stale = initial[checkpoint_key], 0, 0, 0
    stop_reason = "max_epochs"
    sample_counts = np.zeros(len(tr_idx), dtype=np.int64)
    csv_keys = list(initial)
    with (out / "history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerow(initial)
        stream.flush()
        try:
            for epoch in range(1, max_epochs + 1):
                final_epoch = epoch
                network.train()
                order = (rng.permutation(len(tr_idx)) if probabilities is None else
                         rng.choice(len(tr_idx), len(tr_idx), replace=True, p=probabilities))
                np.add.at(sample_counts, order, 1)
                grad_values, total_objective, seen = [], 0.0, 0
                failed = False
                for lo in range(0, len(order), batch_size):
                    idx = order[lo:lo + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    output = network(x_train[idx])
                    if weights is None:
                        per_sample = ((output - y_train[idx]) ** 2).mean(dim=1)
                    else:
                        pred_coeff = output * scale_tensor + base_train[idx]
                        per_sample = weighted_field_loss(pred_coeff, coeff_train[idx], gram_t,
                                                         cross_t[idx], floor_t[idx], terms["weight_sum"])
                    if probabilities is not None and importance_correction:
                        factors = torch.as_tensor((1 / (len(tr_idx) * probabilities[idx])).astype(np.float32))
                        per_sample = per_sample * factors
                    loss = per_sample.mean()
                    if not torch.isfinite(loss):
                        failed = True
                        break
                    loss.backward()
                    squared_norm = sum(float(torch.sum(p.grad.detach().double() ** 2))
                                       for p in network.parameters() if p.grad is not None)
                    norm = math.sqrt(squared_norm)
                    if not math.isfinite(norm):
                        failed = True
                        break
                    optimizer.step()
                    optimizer_steps += 1
                    grad_values.append(norm)
                    total_objective += float(loss.detach()) * len(idx)
                    seen += len(idx)
                final_epoch = epoch
                if failed:
                    stop_reason = "nonfinite"
                    break
                network.eval()
                current, row = evaluate(epoch, np.mean(grad_values), max(grad_values), total_objective / seen)
                if not math.isfinite(row[checkpoint_key]):
                    stop_reason = "nonfinite"
                    break
                history.append(row)
                writer.writerow(row)
                stream.flush()
                if row[checkpoint_key] < best_score - min_delta:
                    best_score, best_model, best_epoch, stale = row[checkpoint_key], current, epoch, 0
                else:
                    stale += 1
                if stale >= patience:
                    stop_reason = "early_stopping"
                    break
        except KeyboardInterrupt:
            stop_reason = "interrupted"
    summary = dict(**metadata, best_epoch=int(best_epoch), final_epoch=int(final_epoch),
                   stop_reason=stop_reason, best_monitor=float(best_score), optimizer_steps=optimizer_steps,
                   elapsed_seconds=float(time.perf_counter() - start), completed_logged_epochs=len(history) - 1,
                   best_metrics=history[best_epoch] if best_epoch < len(history) else None,
                   unique_train_snapshots_sampled=int(np.count_nonzero(sample_counts)),
                   minimum_snapshot_draws=int(sample_counts.min()), maximum_snapshot_draws=int(sample_counts.max()))
    best_model["metadata_json"] = np.asarray(json.dumps(summary, ensure_ascii=False, default=_jsonable, allow_nan=False))
    model_path = out / "best_model.npz"
    np.savez_compressed(model_path, **best_model)
    np.savez_compressed(out / "sampling_counts.npz", sample_indices=tr_idx, counts=sample_counts)
    _write_json(out / "summary.json", summary)
    _write_json(out / "history.json", history)
    return dict(model=best_model, model_path=str(model_path.resolve()), summary=summary, history=history)


def train(cache, split, hidden, seed, output_dir, variant="standard", max_epochs=600,
          patience=80, batch_size=256, learning_rate=1e-3, threads=4,
          min_delta=1e-9, sample_weights=None, point_weights=None, base_model=None,
          train_sample_indices=None, validation_sample_indices=None,
          importance_correction=False, normalize_features=True,
          monitor="field_rmse_T", weight_decay=0.0, overwrite=False):
    """See _train_impl for argument details; limit NumPy BLAS as well as Torch."""
    arguments = locals().copy()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        return _train_impl(**arguments)
