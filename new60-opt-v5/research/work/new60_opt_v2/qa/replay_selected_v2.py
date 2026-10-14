"""Independent, float64 replay of declared V2 models; no model selection.

Only NumPy inference entry points are called. Prediction mode defaults to the
42 training and 9 validation cases. Excluded raw NPY rows are skipped without
numeric decoding. Full60 fitting is a separate explicitly requested run.
See REPLAY_SELECTED_V2.md for the choices schema and evaluation limitations.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "e456656a2aefb718de541ae3169c79e3950a6d7ada0ac980e4a22fdc097b1e96"
QUANTITIES = {"Br,Bt": "vector", "vector": "vector", "planar_magnitude": "planar_magnitude",
              "sqrt(Br^2+Bt^2)": "planar_magnitude", "|B|": "planar_magnitude", "Br": "Br"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def pointer(document, path):
    """RFC6901 JSON pointer, including the empty root pointer."""
    for part in path.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def walk(document, path=""):
    if isinstance(document, dict):
        for key, value in document.items():
            p = path + "/" + str(key)
            yield p, key, value
            yield from walk(value, p)
    elif isinstance(document, list):
        for i, value in enumerate(document):
            yield from walk(value, path + "/" + str(i))


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, details=None, fatal=False):
        row = {"name": name, "pass": bool(passed)}
        if details is not None:
            row["details"] = details
        self.checks.append(row)
        if fatal and not passed:
            raise ValueError(name + ": " + str(details))
        return bool(passed)

    def same(self, name, a, b, atol=0, rtol=0, fatal=False):
        a, b = np.asarray(a), np.asarray(b)
        shape = a.shape == b.shape
        numeric = a.dtype.kind in "biufc" and b.dtype.kind in "biufc"
        passed = shape and (np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
                            if numeric else np.array_equal(a, b))
        delta = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if shape and numeric and a.size else None
        return self.check(name, passed, {"actual_shape": list(a.shape), "expected_shape": list(b.shape),
                                        "max_abs_delta": delta, "atol": atol, "rtol": rtol}, fatal)

    def artifact(self, name, path, expected, fatal=True):
        actual = sha256(path)
        self.check(name, isinstance(expected, str) and actual == expected,
                   {"path": str(path), "expected_sha256": expected, "actual_sha256": actual}, fatal)
        return actual


def read_rows(path, key, indices, count=60, return_masked=False):
    """Materialize ONLY authorized case rows of a C-order NPY ZIP member.

    Deflate may internally decompress skipped bytes; excluded labels are never
    converted to numerical arrays. Optional masked output fills them with NaN.
    """
    indices = [int(i) for i in indices]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
        raise ValueError("Invalid or duplicate case indices")
    selected = {i: j for j, i in enumerate(indices)}
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if fortran or len(shape) != 3 or shape[0] != count or dtype.kind != "f":
            raise ValueError("Expected case-major, floating, C-order 3D label member")
        out = np.full((count if return_masked else len(indices), *shape[1:]), np.nan, dtype=dtype)
        block_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        for i in range(count):
            if i in selected:
                raw = stream.read(block_bytes)
                if len(raw) != block_bytes:
                    raise ValueError("Truncated NPY label member")
                out[i if return_masked else selected[i]] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
            else:
                stream.seek(block_bytes, 1)
    viewed = out[indices] if return_masked else out
    if not np.isfinite(viewed).all():
        raise ValueError("Nonfinite authorized source labels")
    return out


def label_policy(choices, branch, allow_test, choices_path):
    if branch == "fitting_full60":
        if allow_test:
            raise ValueError("Full60 fitting has no test partition; do not pass test authorization")
        if not choices.get("full60_fitting_authorized", False):
            raise ValueError("choices must record full60_fitting_authorized=true")
        return ["training_fit"]
    if allow_test:
        sidecars = [choices_path.with_suffix(choices_path.suffix + ".sha256"), choices_path.with_suffix(".sha256")]
        valid = any(p.exists() and p.read_text(encoding="ascii").split()[0] == sha256(choices_path) for p in sidecars)
        declared_frozen = choices.get("frozen") is True or (bool(choices.get("frozen_utc")) and choices.get("test_metrics_computed") is False)
        if not (declared_frozen and choices.get("previous_test_already_seen") is True and valid):
            raise ValueError("Reused-test evaluation requires frozen=true, previous_test_already_seen=true, valid choices SHA sidecar")
        return ["train", "validation", "reused_test"]
    return ["train", "validation"]


def signed_truth(source, region, indices):
    br = read_rows(source, region + "_Br", indices)
    if region == "airgap":
        return br.astype(np.float64)
    bt = read_rows(source, region + "_Bt", indices)
    return np.stack((br, bt), axis=-1).reshape(*br.shape[:2], -1).astype(np.float64)


def target_truth(signed, quantity):
    if quantity == "planar_magnitude":
        v = signed.reshape(*signed.shape[:2], -1, 2)
        return np.sqrt(np.sum(v * v, axis=-1))
    return signed


def aggregate_metrics(prediction, truth, axes=None, near_zero_rms_T=1e-8):
    """Equal scalar sample weights. Undefined relative error is JSON null."""
    truth = np.asarray(truth, dtype=np.float64)
    error = np.asarray(prediction, dtype=np.float64) - truth
    squared = np.sum(error * error, axis=axes, dtype=np.float64)
    energy = np.sum(truth * truth, axis=axes, dtype=np.float64)
    rmse = np.sqrt(np.mean(error * error, axis=axes, dtype=np.float64))
    trms = np.sqrt(np.mean(truth * truth, axis=axes, dtype=np.float64))
    relative = np.full(np.shape(energy), np.nan, dtype=np.float64)
    np.divide(squared, energy, out=relative, where=energy > 0)
    relative = 100 * np.sqrt(relative)
    def clean(x):
        if np.ndim(x) == 0:
            value = scalar(x)
            return value if not isinstance(value, float) or np.isfinite(value) else None
        return [clean(v) for v in x]
    return {key: clean(value) for key, value in dict(
        rmse_T=rmse, mae_T=np.mean(np.abs(error), axis=axes, dtype=np.float64),
        max_abs_T=np.max(np.abs(error), axis=axes), relative_L2_pct=relative,
        truth_L2_T=np.sqrt(energy), truth_rms_T=trms, squared_error_T2=squared,
        truth_squared_norm_T2=energy, near_zero_truth_rms=trms <= near_zero_rms_T).items()}


def metric_report(prediction, truth, case_ids, point_ids, quantity, near_zero=1e-8):
    if prediction.shape != truth.shape or prediction.ndim != 3 or not np.isfinite(prediction).all():
        raise ValueError("Wrong prediction shape or nonfinite prediction")
    components = ["Br", "Bt"] if quantity == "vector" else ["Br" if quantity == "Br" else "planar_magnitude"]
    p4 = prediction.reshape(*prediction.shape[:2], len(point_ids), len(components))
    y4 = truth.reshape(*truth.shape[:2], len(point_ids), len(components))
    cases = aggregate_metrics(prediction, truth, (1, 2), near_zero)
    worst = int(np.argmax(cases["rmse_T"]))
    result = dict(global_metrics=aggregate_metrics(prediction, truth, None, near_zero),
                  per_case=dict(case_ids=np.asarray(case_ids).tolist(), **cases),
                  per_case_frame=dict(case_ids=np.asarray(case_ids).tolist(), frame_indices=list(range(truth.shape[1])),
                                      **aggregate_metrics(prediction, truth, 2, near_zero)),
                  per_frame=dict(frame_indices=list(range(truth.shape[1])), **aggregate_metrics(prediction, truth, (0, 2), near_zero)),
                  per_point=dict(point_ids=np.asarray(point_ids).tolist(), **aggregate_metrics(p4, y4, (0, 1, 3), near_zero)),
                  components={name: aggregate_metrics(p4[..., i], y4[..., i], None, near_zero) for i, name in enumerate(components)},
                  worst_case_id=scalar(case_ids[worst]), worst_case_rmse_T=cases["rmse_T"][worst],
                  near_zero_rms_threshold_T=near_zero,
                  denominator_note="relative L2 is 100*sqrt(sum(error^2)/sum(truth^2)); zero energy gives null. Near-zero flags are explanatory, no samples are removed or denominators clamped.")
    if quantity == "vector":
        mp = np.sqrt(np.sum(p4 * p4, axis=-1)); my = np.sqrt(np.sum(y4 * y4, axis=-1))
        result["derived_planar_magnitude_error_separate_scope"] = aggregate_metrics(mp, my, None, near_zero)
    if quantity == "planar_magnitude":
        result["negative_prediction_count"] = int(np.count_nonzero(prediction < 0))
        result["minimum_prediction_T"] = float(np.min(prediction))
    return result


def load_model(path, audit):
    with np.load(path, allow_pickle=False) as archive:
        if {"field", "truth", "targets", "test_truth"}.intersection(archive.files):
            raise ValueError("Model file unexpectedly contains raw labels")
        model = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(model.get("metadata_json", np.asarray("{}"))))
    if "representation_relative_path" in model:
        rp = (path.parent / str(model["representation_relative_path"])).resolve()
        audit.artifact("representation artifact SHA", rp, str(model["representation_sha256"]))
        with np.load(rp, allow_pickle=False) as z:
            # Representations contain bases/scales, not complete case label arrays.
            if {"field", "truth", "targets", "test_truth"}.intersection(z.files):
                raise ValueError("Representation unexpectedly contains raw labels")
            model.update({k: z[k] for k in z.files if k not in model})
    for key, value in model.items():
        if value.dtype.kind in "fc":
            audit.check("finite model array: " + key, np.isfinite(value).all(), fatal=True)
    return model, metadata


def import_inference(path, audit):
    name = "v2_replay_" + sha256(path)[:20]
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    new = set(sys.modules) - before
    forbidden = sorted(x for x in new if x.split(".")[0] in {"torch", "tensorflow", "sklearn", "jax"})
    audit.check("inference imports no training framework", not forbidden, forbidden, fatal=True)
    return module


def source_and_provenance(choice, model, metadata, root, cache_path, split_path, train_ids, val_ids, audit):
    documents = {"model_metadata": metadata, "frozen_choice_declaration": choice}
    for spec in choice.get("provenance_files", []):
        path = (root / spec["relative_path"]).resolve()
        audit.artifact("provenance file SHA", path, spec["sha256"])
        if path.suffix == ".json":
            documents[spec["relative_path"]] = read_json(path)
    cache_hash, split_hash = sha256(cache_path), sha256(split_path)
    evidence = {"cache": [], "split": [], "train_ids": []}
    train_keys = {"train_case_ids", "training_case_ids", "training_original_ids", "train_original_ids", "train_original_case_ids", "train_ids"}
    val_keys = {"validation_case_ids", "validation_original_ids", "validation_original_case_ids", "validation_ids"}
    for origin, document in documents.items():
        for p, key, value in walk(document):
            label = origin + p
            if key in {"cache_sha256", "source_cache_sha256"} and isinstance(value, str):
                audit.check("recorded cache SHA: " + label, value == cache_hash, fatal=True); evidence["cache"].append(label)
            elif key == "split_sha256" and isinstance(value, str):
                audit.check("recorded split SHA: " + label, value == split_hash, fatal=True); evidence["split"].append(label)
            elif key in train_keys and isinstance(value, list):
                audit.same("recorded train IDs: " + label, value, train_ids, fatal=True); evidence["train_ids"].append(label)
            elif key in val_keys and isinstance(value, list) and choice.get("branch", "prediction") == "prediction":
                audit.same("recorded validation IDs: " + label, value, val_ids, fatal=True)
    # A deterministic magnitude wrapper has a separately bound vector source.
    model_source = choice.get("model_source_module_relative_path", choice["inference_module_relative_path"])
    if model_source != choice["inference_module_relative_path"]:
        audit.artifact("base inference source SHA", root / model_source, choice["model_source_module_sha256"])
    # Direct metadata hashes refer to the actual inference/training source file.
    for key in ("source_sha256", "source_code_sha256"):
        if isinstance(metadata.get(key), str):
            audit.artifact("training-time source SHA: " + key, root / model_source, metadata[key])
    for binding in choice.get("source_bindings", []):
        expected = binding.get("expected_sha256")
        record = binding.get("record", "model_metadata")
        if "pointer" in binding:
            expected = pointer(documents[record], binding["pointer"])
        audit.artifact("bound source SHA", root / binding["source_relative_path"], expected)
    if "support_source_sha256" in metadata:
        found = any(b.get("record", "model_metadata") == "model_metadata" and b.get("pointer") == "/support_source_sha256"
                    for b in choice.get("source_bindings", []))
        audit.check("support source hash has explicit file binding", found, fatal=True)
    audit.check("cache provenance recorded", bool(evidence["cache"]), evidence["cache"])
    audit.check("split provenance recorded", bool(evidence["split"]), evidence["split"])
    audit.check("training case identity recorded", bool(evidence["train_ids"]), evidence["train_ids"])
    has_source = any(isinstance(metadata.get(k), str) for k in ("source_sha256", "source_code_sha256")) or any(
        (root / b["source_relative_path"]).resolve() == (root / model_source).resolve()
        for b in choice.get("source_bindings", []))
    audit.check("training-time source identity recorded", has_source)
    return {"documents": list(documents), "evidence": evidence,
            "note": "Hashes and training IDs check artifact identity; they cannot alone prove how an arbitrary past training program used labels. Fields evidenced only by frozen_choice_declaration are final bindings, not retroactively claimed training-time records."}


def representation_checks(model, metadata, truth_train, conditions_train, geometry, audit):
    """Numerically check recognizable direct means/bases; never refit SVD."""
    numerical = []
    for key in ("point_id", "xy", "theta"):
        if key in model:
            audit.same("model geometry: " + key, model[key], geometry[key], atol=1e-10)
    if "train_conditions" in model:
        audit.same("model train conditions", model["train_conditions"], conditions_train)
    expected = truth_train.mean(axis=(0, 1), dtype=np.float64)
    for key in ("field_mean", "spatial_mean"):
        if key in model and model[key].shape == expected.shape:
            # Some models embed the V1 float32 mean after casting to float64.
            tol = 2e-7
            audit.same("train-only physical mean: " + key, model[key], expected, atol=tol)
            numerical.append(key)
    flatmean = truth_train.reshape(len(truth_train), -1).mean(axis=0, dtype=np.float64)
    if "cycle_mean" in model and model["cycle_mean"].shape == flatmean.shape:
        audit.same("train-only physical whole-cycle mean", model["cycle_mean"], flatmean, atol=2e-8)
        numerical.append("cycle_mean")
        if "cycle_basis" in model and model["cycle_basis"].shape[1] == len(flatmean):
            basis = np.asarray(model["cycle_basis"], dtype=np.float64)
            audit.same("whole-cycle basis orthonormal", basis @ basis.T, np.eye(len(basis)), atol=2e-8)
    for key in model:
        if re.fullmatch(r"b\d+_mean", key):
            prefix = key[:-4]; index_key = prefix + "indices"
            if index_key in model:
                idx = model[index_key].astype(int)
                audit.same("train-only block mean: " + key, model[key], flatmean[idx], atol=2e-8)
                numerical.append(key)
    for key in ("spatial_basis", "basis"):
        if key in model and model[key].ndim == 2 and model[key].shape[0] == truth_train.shape[-1]:
            basis = np.asarray(model[key], dtype=np.float64)
            # A float64 container can hold the original float32 V1 basis.
            tol = 5e-6
            audit.same("POD basis orthonormal: " + key, basis.T @ basis, np.eye(basis.shape[1]), atol=tol)
    return {"numerically_checked_physical_means": numerical,
            "limitation": "No SVD or training is repeated. Source/cache/split/representation hashes and case IDs supply provenance; physical means and recognizable basis orthogonality provide additional numerical checks."}


def scaler_checks(model, metadata, c, audit):
    """Independent math-coordinate reconstruction for the declared V2 formats."""
    performed = []
    if metadata.get("branch") == "fitting_full60" and "feature_kind" not in model:
        model = dict(model, feature_kind=np.asarray("mixed"))
    if all(k in model for k in ("feature_kind", "current_scale", "feature_mean", "feature_scale")):
        kind = str(model["feature_kind"]); cs = float(model["current_scale"])
        r = c[:, 0] / cs; angle = np.deg2rad(c[:, 1])
        if kind.startswith("sat"):
            cap = float(kind[3:]); r = cap / cs * np.tanh(c[:, 0] / cap)
        x, y = r * np.cos(angle), r * np.sin(angle)
        features = {"native": np.column_stack((r, c[:, 1] / 90)),
                    "cartesian": np.column_stack((x, y)),
                    "cartesian2": np.column_stack((x, y, x*x, x*y, y*y)),
                    "mixed": np.column_stack((r, c[:, 1] / 90, x, y, x*x, x*y, y*y))}
        f = np.column_stack((x, y, x*x, x*y, y*y)) if kind.startswith("sat") else features.get(kind)
        if f is not None:
            audit.same("train-only neural input mean", model["feature_mean"], f.mean(0), atol=1e-12)
            audit.same("train-only neural input scale", model["feature_scale"], np.maximum(f.std(0), 1e-8), atol=1e-12)
            performed.append("neural feature_mean/feature_scale")
    if all(k in model for k in ("coordinate", "cap", "input_mean", "input_scale")):
        kind = str(model["coordinate"]); r = c[:, 0].copy(); angle = np.deg2rad(c[:, 1])
        if kind == "saturated":
            cap = float(model["cap"]); r = cap * np.tanh(r / cap)
        f = c if kind == "native" else np.column_stack((r*np.cos(angle), r*np.sin(angle)))
        audit.same("train-only kernel neural input mean", model["input_mean"], f.mean(0), atol=1e-12)
        audit.same("train-only kernel neural input scale", model["input_scale"], np.maximum(f.std(0), 1e-8), atol=1e-12)
        performed.append("neural frame kernel input_mean/input_scale")
    if metadata.get("format") == "new60_condition_kernel_v2":
        for i, block in enumerate(metadata["blocks"]):
            prefix = f"b{i}_"; kind = block["coordinate"]; r = c[:, 0] / block["current_scale"]
            if kind == "polar_xy" or kind.startswith("saturated_"):
                if kind.startswith("saturated_"):
                    factor = float(kind.split("_")[1]); r = factor*np.tanh(r/factor)
                angle = np.deg2rad(c[:, 1]); f = np.column_stack((r*np.cos(angle), r*np.sin(angle)))
            else:
                f = np.column_stack((r ** {"native": 1, "sqrt_native": .5, "square_native": 2}[kind], c[:, 1]/90))
            center, scale = f.mean(0), np.maximum(f.std(0), 1e-10)
            audit.same("train-only kernel input center: " + prefix, model[prefix+"center"], center, atol=1e-12)
            audit.same("train-only kernel input scale: " + prefix, model[prefix+"input_scale"], scale, atol=1e-12)
            z = (f-center)/scale*np.asarray([np.sqrt(block["anisotropy"]), 1/np.sqrt(block["anisotropy"])])
            audit.same("kernel centers are training cases: " + prefix, model[prefix+"centers"], z, atol=1e-12)
        performed.append("condition kernel block centers/scales/training centers")
    return {"performed": performed, "unrecognized_scalers_not_claimed_verified": True}


def saved_prediction_check(choice, partition, ids, replay, root, audit):
    matches = [s for s in choice.get("saved_predictions", []) if s["partition"] == partition]
    if partition == "validation" or choice.get("branch") == "fitting_full60":
        audit.check("saved prediction declared: " + partition, bool(matches))
    for spec in matches:
        path = root / spec["relative_path"]
        audit.artifact("saved prediction SHA: " + partition, path, spec["sha256"])
        key = spec.get("prediction_key", "prediction")
        if "truth" in key.lower() or "label" in key.lower():
            raise ValueError("Saved prediction key must not select a truth/label array")
        with np.load(path, allow_pickle=False) as archive:
            saved_ids = archive[spec.get("case_ids_key", "case_ids")]
            audit.same("saved prediction case IDs: " + partition, saved_ids, ids, fatal=True)
            saved = archive[key]
            if spec.get("saved_transform") == "hypot(Br,Bt)":
                if saved.shape[-1] % 2:
                    raise ValueError("Saved vector dimension is not even")
                saved = np.hypot(saved.astype(np.float64)[..., 0::2], saved.astype(np.float64)[..., 1::2])
            audit.same("saved prediction replay: " + partition, replay, saved,
                       atol=float(spec.get("atol", 2e-6 if saved.dtype.itemsize <= 4 else 2e-9)),
                       rtol=float(spec.get("rtol", 1e-9)))


def reported_metric_check(choice, partition, metrics, root, audit):
    for spec in choice.get("reported_metrics", []):
        if spec["partition"] != partition:
            continue
        path = root / spec["relative_path"]
        audit.artifact("reported metric source SHA: " + partition, path, spec["sha256"])
        expected = pointer(read_json(path), spec.get("pointer", ""))
        computed = dict(metrics["global_metrics"], worst_case_rmse_T=metrics["worst_case_rmse_T"])
        aliases = {"max_abs_component_error_T": "max_abs_T"}
        checked = []
        for key, value in expected.items():
            canonical = aliases.get(key, key)
            if canonical in {"rmse_T", "mae_T", "max_abs_T", "relative_L2_pct", "worst_case_rmse_T"}:
                audit.same("reported metric replay: " + partition + "/" + key, computed[canonical], value,
                           atol=float(spec.get("atol", 1e-9)), rtol=float(spec.get("rtol", 1e-7)))
                checked.append(key)
        audit.check("reported metric pointer recognizes fields", bool(checked), checked)


def run(choices_path, output, root=ROOT, branch="prediction", allow_test=False, bindings_path=None, export_scientific=False, scientific_output_root=None):
    choices_path, output, root = Path(choices_path).resolve(), Path(output).resolve(), Path(root).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite QA output: " + str(output))
    choices = read_json(choices_path)
    partitions = label_policy(choices, branch, allow_test, choices_path)
    binding_hash = None
    if bindings_path is not None:
        bindings = read_json(bindings_path)
        if bindings["choices_sha256"] != sha256(choices_path):
            raise ValueError("Supplementary bindings refer to a different frozen choices artifact")
        permitted = {"source_bindings", "provenance_files", "saved_predictions", "reported_metrics"}
        mapping = {row["id"]: row for row in choices["models"]}
        for item in bindings["models"]:
            if set(item) - permitted - {"id"} or item["id"] not in mapping:
                raise ValueError("Supplementary bindings cannot change selection, quantities, models, or inference rules")
            for key in permitted.intersection(item):
                mapping[item["id"]].setdefault(key, []).extend(copy.deepcopy(item[key]))
        binding_hash = sha256(bindings_path)
    study = (root / choices.get("source_study_relative_path", "../new60_20261014")).resolve()
    source = study / "source/combined/flux60_s60_merged.npz"
    split_path = study / "cache_v1/split.json"
    split = read_json(split_path)
    audit = Audit()
    if choices.get("frozen") or choices.get("frozen_utc"):
        sidecars = [choices_path.with_suffix(choices_path.suffix + ".sha256"), choices_path.with_suffix(".sha256")]
        audit.check("frozen choices SHA sidecar", any(p.exists() and p.read_text(encoding="ascii").split()[0] == sha256(choices_path) for p in sidecars), fatal=True)
    if "protocol_sha256" in choices:
        audit.artifact("global protocol SHA", root / "protocol.json", choices["protocol_sha256"])
    if "split_sha256" in choices:
        audit.artifact("global split SHA", split_path, choices["split_sha256"])
    audit.check("split exact disjoint 42/9/9", [len(split[k]) for k in ("train", "validation", "test")] == [42, 9, 9]
                and sorted(sum([split[k] for k in ("train", "validation", "test")], [])) == list(range(60)), fatal=True)
    audit.artifact("original merged source SHA", source, choices.get("source_sha256", SOURCE_SHA256))
    with np.load(source, allow_pickle=False) as z:
        case_ids = z["case_id"]
        conditions = np.column_stack((z["I_A"], z["Thet_deg"]))
        common = {"theta": z["theta_m_absolute_deg"], "solver_index": z["solver_index"], "time_s": z["time_s"]}
        for key in ("theta_m_relative_deg", "theta_e_deg", "initial_mechanical_deg", "speed_rpm"):
            if key in z.files:
                common[key] = z[key]
        geometries = {r: {"point_id": z[r + "_point_id"], "xy": np.column_stack((z[r + "_x_ref_mm"], z[r + "_y_ref_mm"])),
                           "theta": common["theta"]} for r in ("stator", "rotor", "airgap")}
    audit.same("source case IDs", case_ids, split["case_ids"], fatal=True)
    for key in ("train", "validation", "test"):
        audit.same("original split IDs: " + key, case_ids[split[key]], split["original_ids"][key], fatal=True)
    row_groups = {"training_fit": list(range(60))} if branch == "fitting_full60" else {
        "train": split["train"], "validation": split["validation"], **({"reused_test": split["test"]} if allow_test else {})}
    allowed = sorted(sum(row_groups.values(), [])); excluded = sorted(set(range(60)) - set(allowed))
    rows = choices.get("models", choices.get("choices", []))
    selected = [c for c in rows if c.get("branch", "prediction") == branch]
    audit.check("nonempty selected branch", bool(selected), fatal=True)
    ids = [c.get("id", str(i)) for i, c in enumerate(selected)]
    audit.check("unique model IDs", len(set(ids)) == len(ids), fatal=True)
    results = []
    output.mkdir(parents=True)
    with threadpool_limits(limits=2):
      for choice_index, choice in enumerate(selected):
        begin_check = len(audit.checks)
        result = {"id": ids[choice_index], "branch": branch, "region": choice["region"], "quantity": choice["quantity"]}
        results.append(result)
        try:
            region = choice["region"]; quantity = QUANTITIES[choice["quantity"]]
            source_quantity = QUANTITIES[choice.get("source_quantity", choice["quantity"])]
            derived = source_quantity != quantity
            audit.check("quantity transform declared consistently", not derived or (
                source_quantity == "vector" and quantity == "planar_magnitude" and choice.get("output_transform") == "hypot(Br,Bt)"
                and "model_source_module_relative_path" in choice), fatal=True)
            audit.check("quantity belongs to region", (region == "airgap") == (quantity == "Br"), fatal=True)
            signed = signed_truth(source, region, allowed)
            lookup = {i: j for j, i in enumerate(allowed)}
            truth = target_truth(signed, quantity)
            cache_path = study / "cache_v1" / (region + ".npz")
            cached = read_rows(cache_path, "field", allowed, return_masked=True)
            audit.check("excluded cache label rows remain NaN", np.isnan(cached[excluded]).all(), {"excluded_indices": excluded})
            audit.same("original signed field -> cache exact mapping", cached[allowed], signed)
            with np.load(cache_path, allow_pickle=False) as z:
                audit.same("cache case IDs", z["case_ids"], case_ids)
                audit.same("cache native conditions", z["conditions"], conditions)
                audit.same("cache POD training rows", z["pod_train_case_indices"], split["train"])
                for key, value in geometries[region].items():
                    audit.same("cache geometry: " + key, z[key], value)
                if set(split["train"]).issubset(allowed):
                    expected_mean = cached[split["train"]].reshape(-1, cached.shape[-1]).mean(axis=0, dtype=np.float64).astype(np.float32)
                    audit.same("V1 spatial POD mean train42 only", z["mean"], expected_mean)
            del cached
            model_path = root / choice["model_relative_path"]
            inference_path = root / choice["inference_module_relative_path"]
            result["model_sha256"] = audit.artifact("selected model SHA", model_path, choice["model_sha256"])
            result["inference_source_sha256"] = audit.artifact("selected inference SHA", inference_path, choice["inference_module_sha256"])
            for side in (model_path.with_suffix(model_path.suffix + ".sha256"),):
                if side.exists():
                    audit.check("model sidecar SHA", side.read_text(encoding="ascii").split()[0] == result["model_sha256"], fatal=True)
            model, metadata = load_model(model_path, audit)
            result["model_metadata"] = metadata
            if "quantity" in metadata:
                audit.check("model physical quantity matches source_quantity", QUANTITIES.get(metadata["quantity"]) == source_quantity,
                            {"recorded": metadata["quantity"], "declared_source": source_quantity}, fatal=True)
            train_indices = list(range(60)) if branch == "fitting_full60" else split["train"]
            result["provenance"] = source_and_provenance(choice, model, metadata, root, cache_path, split_path,
                case_ids[train_indices], case_ids[split["validation"]], audit)
            source_truth = target_truth(signed, source_quantity)
            train_truth = source_truth[[lookup[i] for i in train_indices]]
            result["representation_checks"] = representation_checks(model, metadata, train_truth, conditions[train_indices], geometries[region], audit)
            result["scaler_checks"] = scaler_checks(model, metadata, conditions[train_indices], audit)
            module = import_inference(inference_path, audit)
            base_predict = None
            if derived:
                base_module = import_inference(root / choice["model_source_module_relative_path"], audit)
                base_predict = getattr(base_module, "predict")
            predict = getattr(module, choice.get("inference_function", "predict"))
            argument = model if choice.get("model_argument", "path") == "dict" else model_path
            kwargs = choice.get("predict_kwargs", {})
            if "frame_index" in kwargs or any(kwargs.get(k) for k in ("return_coeff", "return_latent")):
                raise ValueError("Primary replay requires full physical fields for every frame")
            result["metrics"] = {}
            scientific_prediction = np.full(truth.shape, np.nan, dtype=np.float64) if export_scientific else None
            for partition in partitions:
                indices = row_groups[partition]
                y = truth[[lookup[i] for i in indices]]
                before = set(sys.modules)
                prediction = np.asarray(predict(argument, conditions[indices], **kwargs), dtype=np.float64)
                forbidden = sorted(x for x in set(sys.modules) - before if x.split(".")[0] in {"torch", "tensorflow", "sklearn", "jax"})
                audit.check("prediction uses no training framework", not forbidden, forbidden, fatal=True)
                audit.check("full output shape and finite: " + partition, prediction.shape == y.shape and np.isfinite(prediction).all(),
                            {"actual": list(prediction.shape), "expected": list(y.shape)}, fatal=True)
                saved_prediction_check(choice, partition, case_ids[indices], prediction, root, audit)
                result["metrics"][partition] = metric_report(prediction, y, case_ids[indices], geometries[region]["point_id"], quantity)
                reported_metric_check(choice, partition, result["metrics"][partition], root, audit)
                if partition == "validation" and choice.get("validation_selection_metrics"):
                    met = dict(result["metrics"][partition]["global_metrics"], worst_case_rmse_T=result["metrics"][partition]["worst_case_rmse_T"])
                    for key, value in choice["validation_selection_metrics"].items():
                        if key in {"rmse_T", "mae_T", "max_abs_T", "relative_L2_pct", "worst_case_rmse_T"}:
                            audit.same("frozen validation metric: " + key, met[key], value, atol=1e-9, rtol=1e-7)
                if export_scientific:
                    scientific_prediction[[lookup[i] for i in indices]] = prediction
                # Check public single-case and selected-frame contracts on two fixed frames.
                if partition in {"validation", "training_fit"}:
                    if derived:
                        vector = np.asarray(base_predict(argument, conditions[indices], **choice.get("model_source_predict_kwargs", {})), dtype=np.float64)
                        expected = np.hypot(vector[..., 0::2], vector[..., 1::2])
                        audit.same("derived magnitude wrapper equals independent hypot", prediction, expected, atol=2e-9, rtol=1e-9)
                    for frame in (0, y.shape[1] - 1):
                        one = np.asarray(predict(argument, conditions[indices[0]], frame_index=frame, **kwargs))
                        batch = np.asarray(predict(argument, conditions[indices[:2]], frame_index=frame, **kwargs))
                        audit.same("single/frame inference parity: " + str(frame), one, prediction[0, frame], atol=2e-9, rtol=1e-9)
                        audit.same("batch/frame inference parity: " + str(frame), batch, prediction[:2, frame], atol=2e-9, rtol=1e-9)
            audit.check("inference source unchanged during replay", sha256(inference_path) == result["inference_source_sha256"])
            audit.check("model unchanged during replay", sha256(model_path) == result["model_sha256"])
            if export_scientific and all(c["pass"] for c in audit.checks[begin_check:]):
                if not re.fullmatch(r"[A-Za-z0-9_-]+", result["id"]):
                    raise ValueError("Model id is not a safe scientific export filename")
                directory = (Path(scientific_output_root) if scientific_output_root else output / "evaluation") / result["id"]
                directory.mkdir(parents=True, exist_ok=True)
                export_meta = dict(branch=branch, quantity=quantity, source_quantity=source_quantity,
                    output_transform=choice.get("output_transform", "identity"), units="T", xy_units="mm",
                    choices_sha256=sha256(choices_path), model_sha256=result["model_sha256"],
                    inference_sha256=result["inference_source_sha256"], source_sha256=sha256(source),
                    inference_entry=choice["inference_module_relative_path"], fixed_stored_frames_only=True,
                    coordinate_frame="rotor material reference coordinates" if region == "rotor" else "fixed stator coordinates",
                    previous_test_already_seen=True, test_is_fresh_blind=False,
                    label_scope="training fit only" if branch == "fitting_full60" else "42 train / 9 validation / optionally 9 reused test")
                result["scientific_artifacts"] = {}
                for partition, indices in row_groups.items():
                    artifact = directory / (partition + "_predictions.npz")
                    if artifact.exists():
                        raise FileExistsError("Scientific prediction already exists: " + str(artifact))
                    positions = [lookup[i] for i in indices]
                    partition_common = dict(common)
                    if "speed_rpm" in partition_common and np.shape(partition_common["speed_rpm"]) == (60,):
                        partition_common["speed_rpm"] = partition_common["speed_rpm"][indices]
                    np.savez_compressed(artifact, prediction=scientific_prediction[positions], truth=truth[positions],
                        case_ids=case_ids[indices], original_row_indices=np.asarray(indices), partition=np.asarray(partition),
                        conditions=conditions[indices], point_id=geometries[region]["point_id"], xy=geometries[region]["xy"],
                        region=np.asarray(region), quantity=np.asarray(quantity), metadata_json=np.asarray(json.dumps(dict(export_meta, partition=partition))),
                        **partition_common)
                    result["scientific_artifacts"][partition] = {"path": str(artifact.resolve()), "sha256": sha256(artifact),
                                                                "shape": list(scientific_prediction[positions].shape), "dtype": "float64"}
        except Exception as exc:
            audit.check("replay exception", False, {"exception_type": type(exc).__name__, "message": str(exc)})
            result["error"] = str(exc)
        result["checks"] = audit.checks[begin_check:]
        result["pass"] = all(c["pass"] for c in result["checks"])
    report = dict(version="v2-independent-replay-1", created_utc=datetime.now(timezone.utc).isoformat(),
                  branch=branch, choices_sha256=sha256(choices_path), checker_sha256=sha256(Path(__file__)),
                  supplemental_bindings_sha256=binding_hash, scientific_exports_requested=export_scientific,
                  source_sha256=sha256(source), split_sha256=sha256(split_path), threads=2,
                  decoded_original_label_rows=allowed, excluded_label_rows=excluded,
                  evaluated_partitions=partitions, previous_test_already_seen=bool(choices.get("previous_test_already_seen")),
                  fresh_blind_test=False, test_labels_decoded_for_prediction=branch == "prediction" and allow_test,
                  full60_labels_used_only_for_training_fit_audit=branch == "fitting_full60",
                  no_model_selection_or_training_performed=True, checks=audit.checks, results=results,
                  all_checks_pass=all(c["pass"] for c in audit.checks),
                  limitations=["Known fixed 60 frames and measured spatial points only; no unseen time or mesh interpolation claim.",
                               "Equal point/component weights, not finite-element area weights.",
                               "Planar sqrt(Br^2+Bt^2) excludes unavailable Bz and is not signed-vector accuracy.",
                               "Old V1 test cases have been seen. Re-evaluation after freezing remains reused-test comparison.",
                               "Full60 training-fit scores do not estimate unseen-condition accuracy; do not feed them back into prediction selection.",
                               "POD/scaler identity and recognizable physical means are checked; no SVD/training rerun or causal proof of historical label exclusion."])
    (output / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# V2 独立数值回放", "", "状态：" + ("PASS" if report["all_checks_pass"] else "FAIL"),
             "", f"分支：`{branch}`；实际读取标签行：{allowed}；未数值解码行：{excluded}。", "",
             "|模型|区域/物理量|分区|RMSE (T)|全场 relative L2 (%)|最差工况 RMSE (T)|",
             "|---|---|---|---:|---:|---:|"]
    for result in results:
        for partition, met in result.get("metrics", {}).items():
            g = met["global_metrics"]
            lines.append(f"|{result['id']}|{result['region']}/{result['quantity']}|{partition}|{g['rmse_T']:.10g}|{g['relative_L2_pct']}|{met['worst_case_rmse_T']:.10g}|")
    failures = [c for c in audit.checks if not c["pass"]]
    lines += ["", f"共 {len(audit.checks)} 项检查，{len(failures)} 项失败。", "",
              "逐工况、逐帧、逐点、分量误差及原场能量见 JSON。所有误差使用原始 float32 数据转 float64 后计算，不使用网页 uint16 量化数据。", "",
              "二维磁密大小是 sqrt(Br²+Bt²)，不含未提供的 Bz；不能把它的误差与 Br/Bt 向量误差混比。近零场不做逐点百分比相除，零分母的 relative L2 为 null。", "",
              "V1 的旧测试已被查看；冻结后复评是复用留出集对比，不是新的盲测。全 60 工况版只报告 training_fit，不能称独立验证或预测精度。", "",
              "本脚本不训练、不调参、不按本次结果替换模型。哈希与训练 ID 证明交付物身份；数值检查不能单独证明历史训练从未违规读取标签。"]
    if failures:
        lines += ["", "失败项："] + ["- " + c["name"] + ": " + json.dumps(c.get("details"), ensure_ascii=False) for c in failures]
    (output / "replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choices", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--branch", choices=("prediction", "fitting_full60"), default="prediction")
    parser.add_argument("--bindings", type=Path, help="Hash-bound evidence supplement; cannot change frozen choices")
    parser.add_argument("--export-scientific", action="store_true", help="Write per-model float64 truth/prediction plus geometry and partitions only for passing models")
    parser.add_argument("--scientific-output-root", type=Path, help="Defaults to output/evaluation; individual model/partition files never overwritten")
    parser.add_argument("--allow-reused-test-evaluation", action="store_true",
                        help="Use only after explicit authorization, frozen choices and a valid SHA sidecar")
    args = parser.parse_args()
    report = run(args.choices, args.output, args.root, args.branch, args.allow_reused_test_evaluation, args.bindings, args.export_scientific, args.scientific_output_root)
    print(json.dumps({"all_checks_pass": report["all_checks_pass"], "checks": len(report["checks"]),
                      "results": len(report["results"]), "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
