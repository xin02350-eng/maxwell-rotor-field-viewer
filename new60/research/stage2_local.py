"""Validation-only NEW60 stage-two local controls; never evaluate test labels.

The seed-42 architecture winner is fixed before this stage. Trials: uniform
physical-field MSE, hotspot field MSE, hard-frame resampling, a frozen uniform
base plus hotspot-weighted residual, and a learning-rate-only control. Spatial
hotspots and hard frames are frozen from the seed-42 uniform model's TRAINING
errors. A recipe is selected using validation only, frozen before seed 17/73
checks, then retained only if it satisfies the same gates for all three seeds.

CLI: --cache --split --stage1-output --output --quantity [--max-epochs N]
No evaluate/test command is provided. All output directories are immutable.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

from experiment_core import REQUIRED, _validate_cache, predict, train

HERE = Path(__file__).resolve().parent
SEEDS = (42, 17, 73)
FAMILIES = ("hotspot_weighted", "hard_frame_resampling", "hotspot_residual", "learning_rate_0003")
SOURCE_FILES = ("stage2_local.py", "experiment_core.py")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes the stage-one directory")
    return path


def read_selected_label_rows(path, key, selected_cases, expected_cases):
    """Materialize only selected case rows; excluded labels stay NaN.

    ZIP decompression/seek may traverse compressed bytes, but no excluded case
    is converted to a numeric array. Raw-file hashing also does not decode labels.
    Prepared caches must store label arrays in C order so cases are contiguous.
    """
    selected = set(int(i) for i in selected_cases)
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if fortran or len(shape) != 3 or shape[0] != expected_cases or dtype.kind != "f":
            raise ValueError(f"{key} must be a C-order floating [case,time,dim] NPY array")
        result = np.full(shape, np.nan, dtype=dtype)
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        for case in range(shape[0]):
            if case in selected:
                raw = stream.read(row_bytes)
                if len(raw) != row_bytes:
                    raise ValueError(f"Truncated {key} case {case}")
                result[case] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
            else:
                stream.seek(row_bytes, 1)
    return result


def load_selection_cache(cache_path, split_path, quantity):
    """Inputs may be read for all cases; field/coeff values only for train/val."""
    split = read_json(split_path)
    if split.get("dataset") != "NEW60":
        raise ValueError("An explicit NEW60 split is required")
    ids = split.get("case_ids", [])
    if len(ids) != 60:
        raise ValueError("NEW60 must have 60 original case IDs")
    combined = []
    for role, count in (("train", 42), ("validation", 9), ("test", 9)):
        group = split.get(role, [])
        if len(group) != count or any(type(i) is not int or not 0 <= i < 60 for i in group):
            raise ValueError(f"Invalid {role} case-index partition")
        if split.get("original_ids", {}).get(role) != [ids[i] for i in group]:
            raise ValueError(f"Original IDs do not match {role} indices")
        combined.extend(group)
    if len(set(combined)) != 60:
        raise ValueError("Case partitions overlap or do not cover the dataset")
    wanted = (set(REQUIRED) - {"field", "coeff"}) | {
        "metadata_json", "pod_train_case_indices", "conditions", "theta",
        "feature_mean", "feature_scale", "harmonics",
    }
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {key: archive[key] for key in wanted if key in archive.files}
    if cache["case_ids"].dtype.kind not in "iuUS" or cache["case_ids"].tolist() != ids:
        raise ValueError("Original case IDs differ from the split")
    metadata = json.loads(str(cache["metadata_json"]))
    if metadata.get("dataset") != "NEW60" or metadata.get("basis_fit") != "train_cases_only":
        raise ValueError("NEW60 and train-only POD provenance are required")
    if metadata.get("quantity") != quantity or metadata.get("units") != "T" or metadata.get("xy_units") != "mm":
        raise ValueError("Quantity or unit metadata mismatch")
    component_count = metadata.get("component_count")
    if component_count not in (1, 2) or len(metadata.get("components", [])) != component_count:
        raise ValueError("An explicit one- or two-component field schema is required")
    expected_components = {"Br,Bt": ["Br", "Bt"], "Br": ["Br"], "|B|": ["|B|"]}.get(quantity)
    if expected_components is None or metadata["components"] != expected_components:
        raise ValueError("Quantity names and signed-component metadata differ")
    expected_layout = "interleaved" if component_count == 2 else "scalar"
    if metadata.get("layout") != expected_layout:
        raise ValueError("Expected point-major interleaved or scalar component layout")
    if not np.array_equal(np.sort(cache["pod_train_case_indices"]), np.sort(split["train"])):
        raise ValueError("POD training indices differ from the split")
    selected = split["train"] + split["validation"]
    for key in ("field", "coeff"):
        cache[key] = read_selected_label_rows(cache_path, key, selected, 60)
    _validate_cache(cache)
    if cache["field"].shape[-1] != len(cache["point_id"]) * component_count:
        raise ValueError("D differs from P times component_count")
    if len(cache["point_id"]) < 2:
        raise ValueError("At least two spatial points are needed for hotspot/ordinary gates")
    t = cache["features"].shape[1]
    if np.asarray(cache["theta"]).shape != (t,) or np.asarray(cache["conditions"]).shape != (60, 2):
        raise ValueError("Common time/condition dimensions differ")
    conditions_path = Path(cache_path).parent / "case_conditions.json"
    expected_conditions = dict(dataset="NEW60", case_ids=ids, columns=["current", "current_angle"],
                               conditions=np.asarray(cache["conditions"], dtype=np.float64).tolist())
    if (not conditions_path.exists() or digest(conditions_path) != metadata.get("case_conditions_sha256")
            or read_json(conditions_path) != expected_conditions):
        raise ValueError("Canonical case-condition identity differs")
    for key in ("field", "coeff"):
        if not np.isfinite(cache[key][selected]).all():
            raise ValueError(f"Nonfinite train/validation {key}")
        # These are synthetic NaN placeholders, not loaded test values.
        if not np.isnan(cache[key][split["test"]]).all():
            raise AssertionError("Excluded labels were unexpectedly materialized")
    cache["_source_path"] = str(Path(cache_path).resolve())
    return cache, split, metadata


def verify_stage1(stage1, cache_path, split_path, quantity):
    freeze_path = stage1 / "selection_frozen.json"
    if digest(freeze_path) != (stage1 / "selection_frozen.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Stage-one selection freeze has changed")
    frozen, identity = read_json(freeze_path), read_json(stage1 / "run_identity.json")
    if digest(stage1 / "run_identity.json") != frozen["identity_sha256"]:
        raise ValueError("Stage-one identity has changed")
    if digest(cache_path) != identity["cache_sha256"] or digest(split_path) != identity["split_sha256"]:
        raise ValueError("Stage two must use exactly the stage-one cache and split")
    if identity.get("input_metadata", {}).get("quantity") != quantity:
        raise ValueError("Quantity differs from stage one")
    if (stage1 / "final_evaluation").exists():
        raise ValueError("Stage-one test evaluation already exists; stage two must precede final testing")
    if frozen.get("candidate_test_predictions_used_for_selection") is not False:
        raise ValueError("Stage-one validation-only selection provenance is missing")
    for name, expected in identity["code_sha256"].items():
        if digest(inside(stage1 / "source_snapshot", name)) != expected:
            raise ValueError(f"Stage-one source snapshot changed: {name}")
    if digest(HERE / "experiment_core.py") != identity["code_sha256"]["experiment_core.py"]:
        raise ValueError("Training engine changed between stages")
    if digest(stage1 / "report_hotspots.npz") != frozen["hotspot_sha256"]:
        raise ValueError("Stage-one reporting hotspot artifact changed")
    winner = next(row for row in frozen["rows"] if row["name"] == frozen["architecture_winner"])
    for row in frozen["rows"]:
        if digest(inside(stage1, row["model_relative_path"])) != row["model_sha256"]:
            raise ValueError(f"Stage-one model changed: {row['name']}")
    if winner["seed"] != 42:
        raise ValueError("Architecture winner must come from seed42")
    return frozen, identity, winner


def metrics(prediction, truth):
    y = np.asarray(truth, dtype=np.float64)
    err = np.asarray(prediction, dtype=np.float64) - y
    den = float(np.sum(y * y))
    sse = float(np.sum(err * err))
    return dict(rmse_T=float(np.sqrt(np.mean(err * err))), mae_T=float(np.mean(np.abs(err))),
                max_abs_T=float(np.max(np.abs(err))), relative_L2_pct=100 * np.sqrt(sse / den) if den else None)


def group_metrics(prediction, truth, case_ids, hotspot, metadata):
    count = metadata["component_count"]
    mask = np.repeat(hotspot, count)
    per_case = {str(case): metrics(prediction[i], truth[i]) for i, case in enumerate(case_ids)}
    result = dict(quantity=metadata["quantity"], units="T", global_metrics=metrics(prediction, truth),
                  hotspot=metrics(prediction[..., mask], truth[..., mask]),
                  ordinary=metrics(prediction[..., ~mask], truth[..., ~mask]), per_case=per_case,
                  worst_case_rmse_T=max(row["rmse_T"] for row in per_case.values()))
    result["components"] = {name: metrics(prediction[..., j::count], truth[..., j::count])
                            for j, name in enumerate(metadata["components"])}
    if metadata["quantity"] == "|B|":
        result["negative_prediction_count"] = int(np.count_nonzero(prediction < 0))
    return result


def acceptance(candidate, control):
    names = ("global", "ordinary", "worst_case")
    c = [candidate["global_metrics"]["rmse_T"], candidate["ordinary"]["rmse_T"], candidate["worst_case_rmse_T"]]
    b = [control["global_metrics"]["rmse_T"], control["ordinary"]["rmse_T"], control["worst_case_rmse_T"]]
    details = {name: dict(candidate_rmse_T=x, control_rmse_T=y, maximum_allowed_rmse_T=1.02 * y,
                         ratio=x / y if y else None, passed=bool(x <= 1.02 * y))
               for name, x, y in zip(names, c, b)}
    details["hotspot"] = dict(candidate_rmse_T=candidate["hotspot"]["rmse_T"],
                               control_rmse_T=control["hotspot"]["rmse_T"],
                               passed=bool(candidate["hotspot"]["rmse_T"] < control["hotspot"]["rmse_T"]))
    return dict(passed=all(item["passed"] for item in details.values()), criteria=details)


def parameter_count(model):
    with np.load(model, allow_pickle=False) as archive:
        return int(sum(archive[key].size for key in archive.files if re.fullmatch(r"(?:base__)*[wb]\d+", key)))


def select(cache_path, split_path, stage1_output, output, quantity, max_epochs=None, patience=None):
    started = time.perf_counter()
    cache_path, split_path, stage1, output = [Path(p).resolve() for p in (cache_path, split_path, stage1_output, output)]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stage-two output: {output}")
    if output.is_relative_to(stage1) or stage1.is_relative_to(output):
        raise ValueError("Stage-one and stage-two outputs must be separate directories")
    prior, prior_identity, winner = verify_stage1(stage1, cache_path, split_path, quantity)
    cache, split, metadata = load_selection_cache(cache_path, split_path, quantity)
    if metadata.get("region") != prior["region"]:
        raise ValueError("Region differs from stage one")
    base_kwargs = dict(max_epochs=int(prior_identity["base_kwargs"]["max_epochs"] if max_epochs is None else max_epochs),
                       patience=int(prior_identity["base_kwargs"]["patience"] if patience is None else patience),
                       batch_size=256, learning_rate=1e-3, threads=4, monitor="field_rmse_T")
    if min(base_kwargs["max_epochs"], base_kwargs["patience"]) < 1:
        raise ValueError("Epoch and patience budgets must be positive")
    output.mkdir(parents=True, exist_ok=False)
    code_dir = output / "source_snapshot"
    code_dir.mkdir()
    code_hashes = {}
    for name in SOURCE_FILES:
        code_hashes[name] = digest(HERE / name)
        shutil.copyfile(HERE / name, code_dir / name)
    for name in ("selection_frozen.json", "run_identity.json"):
        shutil.copyfile(stage1 / name, output / ("stage1_" + name))
    prior_seconds = sum(float(row["elapsed_seconds"]) for row in prior["rows"])
    tiny_path = stage1 / "tiny_diagnostic" / "summary.json"
    if tiny_path.exists():
        prior_seconds += float(read_json(tiny_path)["elapsed_seconds"])
    identity = dict(schema_version=1, stage="NEW60_stage2_local", dataset="NEW60", region=metadata["region"],
                    created_utc=now(), quantity=quantity, input_metadata=metadata,
                    cache_path=str(cache_path), cache_sha256=digest(cache_path),
                    split_path=str(split_path), split_sha256=digest(split_path), split=split,
                    stage1_output=str(stage1), stage1_freeze_sha256=digest(stage1 / "selection_frozen.json"),
                    stage1_identity_sha256=digest(stage1 / "run_identity.json"),
                    code_sha256=code_hashes, base_kwargs=base_kwargs, hidden=winner["hidden"],
                    architecture_source=winner["name"], stage1_selected=prior["selected"], seeds=list(SEEDS),
                    stage1_training_seconds=prior_seconds, checkpoint="validation_field_rmse_T",
                    loaded_label_roles=["train", "validation"], test_label_rows="not materialized; NaN placeholders",
                    test_metrics_during_selection=False, hotspot_fraction=.1, hotspot_weight=4.,
                    hard_frame_fraction=.1, hard_frame_sampling_weight=4.,
                    selection_rule="strict validation hotspot RMSE decrease and global/ordinary/worst-case RMSE <= 1.02 control; recipe frozen before matched-seed checks; retain only if all three seeds pass",
                    caveats=["Residual versus uniform changes both spatial weighting and additive architecture; compare residual against direct hotspot weighting to isolate the architecture contrast.",
                             "Residual includes a separately trained frozen base and additional parameters/time; do not claim equal compute.",
                             "Hard-frame weighted sampling uses replacement and changes the training distribution; no importance correction.",
                             "No historical hybrid baseline, unseen-position, unseen-space, or test improvement claim."])
    write_new(output / "run_identity.json", identity)
    identity_hash = digest(output / "run_identity.json")
    rows, models = [], {}
    hotspot = point_rmse = hard_mask = hard_weights = None
    p = len(cache["point_id"])
    train_ids, val_ids = split["train"], split["validation"]
    val_truth = cache["field"][val_ids]
    val_original_ids = cache["case_ids"][val_ids].tolist()
    training_seconds = 0.

    def fit(family, seed, base_path=None):
        nonlocal training_seconds
        name = f"{family}_seed{seed}"
        kwargs = dict(base_kwargs)
        kwargs["point_weights"] = np.ones(p)
        comparison = "stage-one architecture winner: standardized coefficient MSE -> uniform physical-field MSE"
        if family in ("hotspot_weighted", "hotspot_residual"):
            kwargs["point_weights"] = np.where(hotspot, 4., 1.)
            comparison = "uniform field control: only spatial weights 1 -> hotspot4/ordinary1"
        if family == "hard_frame_resampling":
            kwargs["sample_weights"] = hard_weights
            comparison = "uniform field control: only train-frame sampling -> hard10% weighted4x, with replacement"
        if family == "learning_rate_0003":
            kwargs["learning_rate"] = .0003
            comparison = "uniform field control: only learning rate 0.001 -> 0.0003"
        if family == "hotspot_residual":
            if base_path is None:
                raise ValueError("Residual requires the matched-seed frozen uniform model")
            kwargs["base_model"] = base_path
            comparison = "direct hotspot-weighted model: additive residual architecture with matched-seed frozen uniform base; same hotspot weights, hidden width, optimizer and branch budget"
        expected_base_hash = digest(base_path) if base_path is not None else None
        result = train(cache, split, winner["hidden"], seed, output / name, variant=name, **kwargs)
        summary = result["summary"]
        if summary["stop_reason"] not in ("early_stopping", "max_epochs"):
            raise RuntimeError(f"Incomplete trial {name}; selection will not be frozen")
        if base_path is not None and digest(base_path) != expected_base_hash:
            raise RuntimeError("Frozen residual base changed during training")
        model_path = Path(result["model_path"])
        training_seconds += float(summary["elapsed_seconds"])
        row = dict(name=name, family=family, seed=seed, hidden=winner["hidden"],
                   parameter_count=summary["parameter_count"], inference_parameter_count=parameter_count(model_path),
                   best_epoch=summary["best_epoch"], final_epoch=summary["final_epoch"], stop_reason=summary["stop_reason"],
                   elapsed_seconds=summary["elapsed_seconds"], stage2_cumulative_training_seconds=training_seconds,
                   stage1_plus_stage2_training_seconds=prior_seconds + training_seconds,
                   learning_rate=kwargs["learning_rate"], objective="exact_weighted_field_MSE",
                   checkpoint="validation_field_rmse_T", controlled_contrast=comparison,
                   base_model_sha256=expected_base_hash, base_model_relative_path=base_path.relative_to(output).as_posix() if base_path else None,
                   model_relative_path=model_path.relative_to(output).as_posix(), model_sha256=digest(model_path),
                   train_original_case_ids=cache["case_ids"][train_ids].tolist(), validation_original_case_ids=val_original_ids)
        row["frozen_parameter_count"] = row["inference_parameter_count"] - row["parameter_count"]
        row["base_training_seconds"] = next((prior_row["elapsed_seconds"] for prior_row in rows
                                             if base_path is not None and models[prior_row["name"]] == base_path), 0.)
        row["recipe_training_seconds"] = row["elapsed_seconds"] + row["base_training_seconds"]
        rows.append(row)
        models[name] = model_path
        return row

    def assess(row):
        prediction = predict(models[row["name"]], cache["features"][val_ids])
        row["validation"] = group_metrics(prediction, val_truth, val_original_ids, hotspot, metadata)
        write_new(output / row["name"] / "validation_groups.json", row["validation"])
        print(json.dumps({"stage": "stage2_validation", "name": row["name"],
                          "global_rmse_T": row["validation"]["global_metrics"]["rmse_T"],
                          "hotspot_rmse_T": row["validation"]["hotspot"]["rmse_T"],
                          "elapsed_seconds": row["elapsed_seconds"]}, ensure_ascii=False), flush=True)

    with threadpool_limits(limits=4):
        control42 = fit("uniform_field_control", 42)
        truth = cache["field"][train_ids]
        prediction = predict(models[control42["name"]], cache["features"][train_ids])
        error = (prediction - truth).reshape(*truth.shape[:2], p, metadata["component_count"])
        point_rmse = np.sqrt(np.mean(error ** 2, axis=(0, 1, 3)))
        hotspot = np.zeros(p, dtype=bool)
        hotspot[np.argsort(-point_rmse, kind="stable")[:min(p - 1, max(1, int(np.ceil(.1 * p))))]] = True
        frame_rmse = np.sqrt(np.mean(error ** 2, axis=(2, 3)))
        hard_mask = np.zeros(frame_rmse.size, dtype=bool)
        hard_mask[np.argsort(-frame_rmse.ravel(), kind="stable")[:max(1, int(np.ceil(.1 * frame_rmse.size)))]] = True
        hard_mask = hard_mask.reshape(frame_rmse.shape)
        hard_weights = np.where(hard_mask, 4., 1.)
        np.savez_compressed(output / "fixed_training_masks.npz", hotspot_mask=hotspot, point_id=cache["point_id"],
                            training_point_rmse_T=point_rmse, hard_frame_mask=hard_mask,
                            training_frame_rmse_T=frame_rmse, sampling_weights=hard_weights,
                            train_case_indices=np.asarray(train_ids), train_case_ids=cache["case_ids"][train_ids])
        mask_hash = digest(output / "fixed_training_masks.npz")
        write_new(output / "mask_definition.json", dict(source_model=control42["name"], source_model_sha256=control42["model_sha256"],
                  labels_used="train only", hotspot_count=int(hotspot.sum()), point_count=p,
                  hard_frame_count=int(hard_mask.sum()), training_frame_count=int(hard_mask.size),
                  order="stable descending train RMS error; point order / split-train order then frame index break ties",
                  hotspot_fixed_across_all_seeds=True, hard_frame_mask_fixed_across_all_seeds=True, artifact_sha256=mask_hash))
        del truth, prediction, error
        assess(control42)
        candidates = []
        for family in FAMILIES:
            row = fit(family, 42, models[control42["name"]] if family == "hotspot_residual" else None)
            assess(row)
            row["acceptance_vs_matched_uniform"] = acceptance(row["validation"], control42["validation"])
            candidates.append(row)
        eligible = [row for row in candidates if row["acceptance_vs_matched_uniform"]["passed"]]
        chosen = min(eligible, key=lambda row: (row["validation"]["hotspot"]["rmse_T"],
                                               row["validation"]["global_metrics"]["rmse_T"],
                                               row["inference_parameter_count"], row["name"])) if eligible else control42
        recipe = dict(frozen_utc=now(), family=chosen["family"], seed42_candidate=chosen["name"],
                      eligible_seed42_candidates=[row["name"] for row in eligible],
                      reason="lowest validation hotspot RMSE among candidates passing all safeguards" if eligible else "no candidate passed all safeguards; uniform control retained",
                      fixed_masks_sha256=mask_hash, identity_sha256=identity_hash,
                      no_further_recipe_search=True, test_labels_used=False)
        write_new(output / "candidate_recipe_frozen.json", recipe)
        recipe_hash = digest(output / "candidate_recipe_frozen.json")
        controls, recipe_rows = [control42], [chosen]
        paired = []
        if eligible:
            paired.append(dict(seed=42, **chosen["acceptance_vs_matched_uniform"]))
        for seed in SEEDS[1:]:
            control = fit("uniform_field_control", seed)
            assess(control)
            controls.append(control)
            if chosen["family"] == "uniform_field_control":
                recipe_rows.append(control)
            else:
                row = fit(chosen["family"], seed, models[control["name"]] if chosen["family"] == "hotspot_residual" else None)
                assess(row)
                row["acceptance_vs_matched_uniform"] = acceptance(row["validation"], control["validation"])
                paired.append(dict(seed=seed, **row["acceptance_vs_matched_uniform"]))
                recipe_rows.append(row)
        stable = bool(eligible) and all(pair["passed"] for pair in paired)
        selected = chosen if stable else control42

    # Protect all inputs/protocols against changes during this run before freezing.
    if (digest(cache_path) != identity["cache_sha256"] or digest(split_path) != identity["split_sha256"]
            or digest(stage1 / "selection_frozen.json") != identity["stage1_freeze_sha256"]
            or digest(stage1 / "run_identity.json") != identity["stage1_identity_sha256"]
            or digest(output / "run_identity.json") != identity_hash
            or digest(output / "fixed_training_masks.npz") != mask_hash
            or digest(output / "candidate_recipe_frozen.json") != recipe_hash):
        raise RuntimeError("An input or frozen protocol changed during stage two")
    for name, expected in code_hashes.items():
        if digest(HERE / name) != expected or digest(code_dir / name) != expected:
            raise RuntimeError(f"Source changed during stage two: {name}")
    for row in rows:
        if digest(models[row["name"]]) != row["model_sha256"]:
            raise RuntimeError(f"Trial model changed: {row['name']}")
    seed_statistics = {}
    for family, group in (("uniform_control", controls), ("frozen_recipe", recipe_rows)):
        seed_statistics[family] = {}
        for part in ("global_metrics", "hotspot", "ordinary"):
            values = [row["validation"][part]["rmse_T"] for row in group]
            seed_statistics[family][part] = dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)),
                                                 worst=float(np.max(values)), values=values)
    frozen = dict(schema_version=1, stage="NEW60_stage2_local", dataset="NEW60", region=metadata["region"], quantity=quantity,
                  frozen_utc=now(), identity_sha256=identity_hash,
                  stage1_freeze_sha256=identity["stage1_freeze_sha256"], rows=rows,
                  architecture_winner=prior["architecture_winner"], architecture_hidden=winner["hidden"],
                  recipe=recipe, recipe_sha256=recipe_hash, matched_seed_checks=paired, seed_stats=seed_statistics,
                  stable_validation_improvement=stable, selected=selected["name"], selected_family=selected["family"], selected_seed=42,
                  selected_model_relative_path=selected["model_relative_path"], selected_model_sha256=selected["model_sha256"],
                  control_model_relative_path=control42["model_relative_path"], control_model_sha256=control42["model_sha256"],
                  selected_reason="frozen local recipe passed validation safeguards for seeds42/17/73" if stable else "no stable local improvement across all three validation seeds; uniform field control retained",
                  hotspot_sha256=mask_hash, hotspot_relative_path="fixed_training_masks.npz",
                  stage1_training_seconds=prior_seconds, stage2_training_seconds=training_seconds,
                  total_training_seconds=prior_seconds + training_seconds, stage2_wall_seconds=time.perf_counter() - started,
                  residual_contrast="Residual minus direct hotspot model is the architecture contrast; residual minus uniform also changes spatial weighting and adds base compute.",
                  validation_original_case_ids=val_original_ids, candidate_test_predictions_used_for_selection=False,
                  test_labels_materialized=False, final_test_evaluation_pending=True)
    write_new(output / "validation_results.json", rows)
    write_new(output / "selection_frozen.json", frozen)
    with (output / "selection_frozen.sha256").open("x", encoding="ascii") as stream:
        stream.write(digest(output / "selection_frozen.json") + "\n")
    with (output / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        keys = ["name", "family", "seed", "parameter_count", "inference_parameter_count", "frozen_parameter_count",
                "best_epoch", "final_epoch", "stop_reason", "elapsed_seconds", "base_training_seconds", "recipe_training_seconds", "stage2_cumulative_training_seconds",
                "global_rmse_T", "hotspot_rmse_T", "ordinary_rmse_T", "worst_case_rmse_T", "passes_gates"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in keys if key in row}
            flat.update(global_rmse_T=row["validation"]["global_metrics"]["rmse_T"],
                        hotspot_rmse_T=row["validation"]["hotspot"]["rmse_T"],
                        ordinary_rmse_T=row["validation"]["ordinary"]["rmse_T"],
                        worst_case_rmse_T=row["validation"]["worst_case_rmse_T"],
                        passes_gates=row.get("acceptance_vs_matched_uniform", {}).get("passed"))
            writer.writerow(flat)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cache", "split", "stage1-output", "output", "quantity"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    args = parser.parse_args()
    frozen = select(args.cache, args.split, args.stage1_output, args.output, args.quantity,
                    max_epochs=args.max_epochs, patience=args.patience)
    print(json.dumps({"selected": frozen["selected"], "stable_validation_improvement": frozen["stable_validation_improvement"],
                      "final_test_evaluation_pending": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
