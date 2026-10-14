"""NEW60 stage-one controlled MLP study; selection and test are separate commands.

select: reference 2x32, four architectures, matched winner/reference seeds,
dynamic tiny diagnostic, and immutable validation-only selection freeze.
evaluate_final: explicitly verify frozen identities, then evaluate test labels.
No historical hybrid baseline claim and no local-loss ablations in this stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

from experiment_core import ARCHITECTURES, load_cache, predict, train
from prepare_new60 import quantity_spec

HERE = Path(__file__).resolve().parent
SEEDS = (42, 17, 73)
SOURCE_FILES = ("run_new60.py", "experiment_core.py", "prepare_new60.py")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_inputs(cache_path, split_path, region, quantity):
    cache = load_cache(cache_path)
    metadata = json.loads(str(cache["metadata_json"]))
    split = read_json(split_path)
    c, t, _ = cache["features"].shape
    if c != 60 or metadata.get("dataset") != "NEW60" or split.get("dataset") != "NEW60":
        raise ValueError("This runner requires genuine NEW60 caches, not renamed OLD12 files")
    if metadata.get("region") != region or metadata.get("basis_fit") != "train_cases_only":
        raise ValueError("Region or train-only POD provenance mismatch")
    if metadata.get("units") != "T" or metadata.get("xy_units") != "mm":
        raise ValueError("Explicit T and mm units are required")
    specification = quantity_spec(quantity)
    for key in ("quantity", "components", "component_count", "layout"):
        if metadata.get(key) != specification[key]:
            raise ValueError(f"Explicit quantity/component metadata mismatch: {key}")
    if metadata.get("angle_kind") not in ("absolute mechanical", "relative mechanical"):
        raise ValueError("Verified mechanical degree angle convention is required")
    ids = cache["case_ids"].tolist()
    if cache["case_ids"].dtype.kind not in "iuUS" or split.get("case_ids") != ids:
        raise ValueError("Split must preserve the cache's canonical original case IDs")
    groups = []
    for key, count in (("train", 42), ("validation", 9), ("test", 9)):
        group = split.get(key, [])
        if len(group) != count or any(type(i) is not int or not 0 <= i < c for i in group):
            raise ValueError(f"Invalid {key} case-index group")
        if split.get("original_ids", {}).get(key) != [ids[i] for i in group]:
            raise ValueError(f"{key} index/original-ID mapping differs")
        groups.extend(group)
    if len(set(groups)) != 60:
        raise ValueError("Case splits overlap or are incomplete")
    if not np.array_equal(np.sort(cache.get("pod_train_case_indices", [])), np.sort(split["train"])):
        raise ValueError("POD training provenance differs from split")
    p = len(cache["point_id"])
    expected = p * specification["component_count"]
    if cache["field"].shape[-1] != expected or np.asarray(cache["theta"]).shape != (t,):
        raise ValueError("Field component layout or common time axis mismatch")
    if (np.asarray(cache["conditions"]).shape != (60, 2)
            or not np.isfinite(cache["conditions"]).all() or not np.isfinite(cache["theta"]).all()):
        raise ValueError("Invalid condition/time metadata")
    shared_conditions_path = Path(cache_path).parent / "case_conditions.json"
    if (not shared_conditions_path.exists()
            or digest(shared_conditions_path) != metadata.get("case_conditions_sha256")):
        raise ValueError("Canonical case/condition identity is missing or modified")
    expected_conditions = dict(dataset="NEW60", case_ids=ids,
                               columns=["current", "current_angle"],
                               conditions=np.asarray(cache["conditions"], dtype=np.float64).tolist())
    if read_json(shared_conditions_path) != expected_conditions:
        raise ValueError("Canonical case IDs map to different current/angle conditions")
    return cache, split, metadata


def metrics(prediction, truth):
    y = np.asarray(truth, dtype=np.float64)
    e = np.asarray(prediction, dtype=np.float64) - y
    denominator = float(np.sum(y ** 2))
    return dict(rmse_T=float(np.sqrt(np.mean(e ** 2))), mae_T=float(np.mean(np.abs(e))),
                max_abs_T=float(np.max(np.abs(e))),
                relative_L2_pct=float(100 * np.sqrt(np.sum(e ** 2) / denominator)) if denominator else None)


def group_metrics(prediction, truth, case_ids, hotspot, quantity):
    specification = quantity_spec(quantity)
    component_count = specification["component_count"]
    if truth.shape[-1] != len(hotspot) * component_count or prediction.shape != truth.shape:
        raise ValueError("Quantity/component dimensions do not match the reporting mask")
    mask = np.repeat(hotspot, component_count)
    per_case = {str(case): metrics(prediction[i], truth[i]) for i, case in enumerate(case_ids)}
    result = dict(quantity=quantity, units="T", global_metrics=metrics(prediction, truth),
                  hotspot=metrics(prediction[..., mask], truth[..., mask]),
                  ordinary=metrics(prediction[..., ~mask], truth[..., ~mask]) if np.any(~mask) else None,
                  per_case=per_case, worst_case_rmse_T=max(v["rmse_T"] for v in per_case.values()))
    if quantity == "Br,Bt":
        result["components"] = {"Br": metrics(prediction[..., 0::2], truth[..., 0::2]),
                                "Bt": metrics(prediction[..., 1::2], truth[..., 1::2])}
    elif quantity == "Br":
        result["components"] = {"Br": metrics(prediction, truth)}
    if quantity == "|B|":
        result["negative_prediction_count"] = int(np.count_nonzero(prediction < 0))
        result["minimum_prediction_T"] = float(np.min(prediction))
    return result


def select(cache_path, split_path, region, output, max_epochs=600, patience=80,
           tiny_epochs=2000, run_tiny=True, *, quantity):
    cache_path, split_path, output = map(lambda p: Path(p).resolve(), (cache_path, split_path, output))
    cache, split, metadata = validate_inputs(cache_path, split_path, region, quantity)
    # Any prior run, complete or interrupted, remains immutable. Use another folder.
    output.mkdir(parents=True, exist_ok=False)
    code_dir = output / "source_snapshot"
    code_dir.mkdir()
    code = {}
    for name in SOURCE_FILES:
        code[name] = digest(HERE / name)
        shutil.copyfile(HERE / name, code_dir / name)
    identity = dict(schema_version=1, dataset="NEW60", region=region, created_utc=now(),
                    cache_path=str(cache_path), cache_sha256=digest(cache_path),
                    split_path=str(split_path), split_sha256=digest(split_path), split=split,
                    code_sha256=code, input_metadata=metadata, baseline="controlled_MLP_2x32",
                    not_claimed=["historical_hybrid_improvement", "unseen_time", "unseen_space", "local_spatial_residual"],
                    base_kwargs=dict(max_epochs=max_epochs, patience=patience, batch_size=256,
                                     learning_rate=1e-3, threads=4), seeds=list(SEEDS),
                    checkpoint="validation_field_rmse_T", test_metrics_during_selection=False,
                    local_trials_included=False,
                    final_choice="retain reference unless winner beats matched reference on every validation seed",
                    selected_seed=42)
    write_new(output / "run_identity.json", identity)
    np.savez_compressed(output / "input_transform.npz", feature_mean=cache["feature_mean"],
                        feature_scale=cache["feature_scale"], harmonics=cache["harmonics"],
                        angle_kind=np.asarray(metadata["angle_kind"]))
    rows, models = [], {}

    def fit(name, hidden, seed):
        result = train(cache, split, hidden, seed, output / name, variant=name, **identity["base_kwargs"])
        summary = result["summary"]
        if summary["stop_reason"] not in ("max_epochs", "early_stopping"):
            raise RuntimeError(f"Incomplete run {name}: {summary['stop_reason']}; no selection is frozen")
        model = Path(result["model_path"])
        row = dict(name=name, hidden=list(hidden), seed=seed, parameter_count=summary["parameter_count"],
                   best_epoch=summary["best_epoch"], final_epoch=summary["final_epoch"],
                   stop_reason=summary["stop_reason"], elapsed_seconds=summary["elapsed_seconds"],
                   validation_rmse_T=summary["best_metrics"]["validation_field_rmse_T"],
                   validation_relative_L2_pct=summary["best_metrics"]["validation_relative_L2_pct"],
                   model_relative_path=model.relative_to(output).as_posix(), model_sha256=digest(model))
        rows.append(row)
        models[name] = model
        print(json.dumps({"stage": "validation", **row}, ensure_ascii=False), flush=True)
        return row

    if run_tiny:
        t = cache["features"].shape[1]
        frames = np.unique(np.linspace(0, t - 1, min(16, t), dtype=int)).tolist()
        samples = [(case, frame) for case in split["train"][:2] for frame in frames]
        tiny_split = dict(train=split["train"], validation=[], test=split["validation"] + split["test"])
        tiny = train(cache, tiny_split, (128, 128), 42, output / "tiny_diagnostic",
                     variant="tiny_training_only_diagnostic", max_epochs=tiny_epochs, patience=300,
                     batch_size=32, learning_rate=1e-3, threads=4, train_sample_indices=samples,
                     monitor="coefficient_loss")
        history = tiny["history"]
        best = tiny["summary"]["best_metrics"]["train_coefficient_loss"]
        initial, final = history[0]["train_coefficient_loss"], history[-1]["train_coefficient_loss"]
        write_new(output / "tiny_check.json", dict(original_case_ids=[cache["case_ids"][i].item() for i in split["train"][:2]],
                  frames=frames, initial_loss=initial, best_loss=best, final_loss=final,
                  best_epoch=tiny["summary"]["best_epoch"], final_epoch=tiny["summary"]["final_epoch"],
                  stop_reason=tiny["summary"]["stop_reason"], best_loss_ratio=best / initial if initial else None,
                  diagnostic_99pct_drop=bool(best < initial * .01),
                  note=f"Only selected training samples; POD fitted on all {len(split['train'])} training cases"))
    reference = fit("reference_2x32_seed42", (32, 32), 42)
    candidates = []
    for hidden in ARCHITECTURES:
        candidates.append(fit(f"architecture_{len(hidden)}x{hidden[0]}_seed42", hidden, 42))
    winner = min(candidates, key=lambda row: (row["validation_rmse_T"], row["parameter_count"]))
    family = winner["name"].removesuffix("_seed42")
    winning_rows, reference_rows = [winner], [reference]
    for seed in SEEDS[1:]:
        winning_rows.append(fit(f"{family}_seed{seed}", winner["hidden"], seed))
        reference_rows.append(fit(f"reference_2x32_seed{seed}", (32, 32), seed))
    paired = [dict(seed=seed, winner_validation_rmse_T=w["validation_rmse_T"],
                   reference_validation_rmse_T=r["validation_rmse_T"],
                   winner_better=w["validation_rmse_T"] < r["validation_rmse_T"])
              for seed, w, r in zip(SEEDS, winning_rows, reference_rows)]
    stable = all(pair["winner_better"] for pair in paired)
    selected = winner if stable else reference
    seed_stats = {}
    for key, group in (("winner", winning_rows), ("reference", reference_rows)):
        values = [row["validation_rmse_T"] for row in group]
        seed_stats[key] = dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)),
                               worst=float(np.max(values)), values=values)
    # Reporting mask is also frozen using training labels, never final test residuals.
    tr = split["train"]
    truth = cache["field"][tr]
    prediction = predict(models[reference["name"]], cache["features"][tr])
    p = len(cache["point_id"])
    error = (prediction - truth).reshape(*truth.shape[:2], p, -1)
    point_rmse = np.sqrt(np.mean(error ** 2, axis=(0, 1, 3)))
    hotspot = np.zeros(p, dtype=bool)
    hotspot[np.argsort(-point_rmse, kind="stable")[:max(1, int(np.ceil(.1 * p)))]] = True
    np.savez_compressed(output / "report_hotspots.npz", hotspot_mask=hotspot,
                        training_point_rmse_T=point_rmse, point_id=cache["point_id"])
    write_new(output / "validation_results.json", rows)
    frozen = dict(schema_version=1, dataset="NEW60", region=region, frozen_utc=now(),
                  identity_sha256=digest(output / "run_identity.json"), rows=rows,
                  architecture_winner=winner["name"], selected=selected["name"],
                  selected_model_relative_path=selected["model_relative_path"],
                  selected_model_sha256=selected["model_sha256"], matched_seeds=paired, seed_stats=seed_stats,
                  stable_validation_improvement=stable,
                  recommendation_status="stable_improvement_over_controlled_reference" if stable else "no_stable_improvement_reference_retained",
                  hotspot_sha256=digest(output / "report_hotspots.npz"),
                  scope="NEW60 first-stage controlled MLP comparison; no historical-hybrid or local-residual claims",
                  candidate_test_predictions_used_for_selection=False, final_test_evaluation_pending=True)
    write_new(output / "selection_frozen.json", frozen)
    (output / "selection_frozen.sha256").write_text(digest(output / "selection_frozen.json") + "\n", encoding="ascii")
    return frozen


def evaluate_final(output, cache_path=None, split_path=None):
    output = Path(output).resolve()
    freeze_path = output / "selection_frozen.json"
    expected_freeze = (output / "selection_frozen.sha256").read_text(encoding="ascii").strip()
    if digest(freeze_path) != expected_freeze:
        raise ValueError("Selection freeze was modified")
    frozen = read_json(freeze_path)
    identity = read_json(output / "run_identity.json")
    if digest(output / "run_identity.json") != frozen["identity_sha256"]:
        raise ValueError("Run identity was modified")
    cache_path = Path(cache_path or identity["cache_path"])
    split_path = Path(split_path or identity["split_path"])
    if digest(cache_path) != identity["cache_sha256"] or digest(split_path) != identity["split_sha256"]:
        raise ValueError("Cache/split differs from selection inputs")
    for name, expected in identity["code_sha256"].items():
        if digest(HERE / name) != expected or digest(output / "source_snapshot" / name) != expected:
            raise ValueError(f"Code changed since selection: {name}")
    for row in frozen["rows"]:
        path = (output / row["model_relative_path"]).resolve()
        if not path.is_relative_to(output) or digest(path) != row["model_sha256"]:
            raise ValueError(f"Model identity differs: {row['name']}")
    if digest(output / "report_hotspots.npz") != frozen["hotspot_sha256"]:
        raise ValueError("Reporting mask differs from freeze")
    cache, split, metadata = validate_inputs(cache_path, split_path, frozen["region"],
                                             identity["input_metadata"]["quantity"])
    final_dir = output / "final_evaluation"
    final_dir.mkdir(exist_ok=False)
    write_new(final_dir / "evaluation_started.json", dict(started_utc=now(), freeze_sha256=expected_freeze,
                                                        post_test_selection_permitted=False))
    with np.load(output / "report_hotspots.npz", allow_pickle=False) as masks:
        hotspot = masks["hotspot_mask"]
    cases = split["test"]
    truth = cache["field"][cases].astype(np.float64)
    original_ids = cache["case_ids"][cases]
    save_names = {frozen["selected"], frozen["architecture_winner"], "reference_2x32_seed42"}
    rows = []
    for row in frozen["rows"]:
        prediction = predict(output / row["model_relative_path"], cache["features"][cases])
        result = group_metrics(prediction, truth, original_ids, hotspot, metadata["quantity"])
        rows.append({**row, "test": result})
        if row["name"] in save_names:
            meta = {**metadata, "model_version": row["name"], "split_role": "test",
                    "layout": metadata["layout"],
                    "model_sha256": row["model_sha256"], "freeze_sha256": expected_freeze,
                    "prediction_dtype": "float64", "selected_before_test": row["name"] == frozen["selected"]}
            np.savez_compressed(final_dir / (row["name"] + "_predictions.npz"), truth=truth,
                                prediction=prediction, case_ids=original_ids, point_id=cache["point_id"],
                                xy=cache["xy"], quantity=np.asarray(metadata["quantity"]),
                                model_name=np.asarray(row["name"]), theta=cache["theta"],
                                conditions=cache["conditions"][cases], hotspot_mask=hotspot,
                                metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)))
    oracle = cache["mean"] + ((truth - cache["mean"]) @ cache["basis"]) @ cache["basis"].T
    oracle_metrics = group_metrics(oracle, truth, original_ids, hotspot, metadata["quantity"])
    report = dict(dataset="NEW60", region=frozen["region"], quantity=metadata["quantity"],
                  components=metadata["components"], units="T", layout=metadata["layout"], completed_utc=now(),
                  selection_frozen_sha256=expected_freeze, selected=frozen["selected"], rows=rows,
                  pod_oracle=oracle_metrics,
                  note="All choices frozen before this explicit test command. Test rows must not be used to reselect.")
    write_new(final_dir / "final_test_results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select", help="Train/select/freeze; does not evaluate test")
    selection.add_argument("--cache", type=Path, required=True)
    selection.add_argument("--split", type=Path, required=True)
    selection.add_argument("--region", choices=["stator", "rotor", "airgap"], required=True)
    selection.add_argument("--quantity", choices=["Br,Bt", "Br", "|B|"], required=True,
                           help="Must explicitly agree with the source-verified cache quantity")
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--max-epochs", type=int, default=600)
    selection.add_argument("--patience", type=int, default=80)
    selection.add_argument("--tiny-epochs", type=int, default=2000)
    selection.add_argument("--skip-tiny", action="store_true")
    evaluation = commands.add_parser("evaluate_final", aliases=["evaluate-final"], help="Explicit frozen test evaluation")
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--cache", type=Path)
    evaluation.add_argument("--split", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.command == "select":
            result = select(args.cache, args.split, args.region, args.output, args.max_epochs,
                            args.patience, args.tiny_epochs, not args.skip_tiny, quantity=args.quantity)
            print(json.dumps({"stage": "frozen_validation_selection", "selected": result["selected"],
                              "test_pending": True}, ensure_ascii=False))
        else:
            report = evaluate_final(args.output, args.cache, args.split)
            print(json.dumps({"stage": "final_test_completed", "selected_before_test": report["selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
