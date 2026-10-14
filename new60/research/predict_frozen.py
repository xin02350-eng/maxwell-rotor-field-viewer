"""NumPy-only prediction from a fully frozen REAL_NEW60 study.

Reads model weights and whitelisted input/geometry metadata, never field or
coefficient labels. It neither evaluates accuracy nor changes a frozen choice.
All outputs use the stored 60 time frames and fixed spatial sampling points.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_STUDY = Path(__file__).resolve().parent.parent
REGIONS = ("stator", "rotor", "airgap")
INPUT_KEYS = ("metadata_json", "case_ids", "conditions", "theta", "theta_m_relative_deg",
              "time_s", "solver_index", "initial_mechanical_deg", "point_id", "xy",
              "feature_mean", "feature_scale", "harmonics")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def located(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing file or path outside study: {relative}")
    return path


def verified(root, relative, expected):
    path = located(root, relative)
    if digest(path) != expected:
        raise ValueError(f"SHA256 mismatch: {relative}")
    return path


def verify_frozen_inputs(study, region, role):
    """Bind inference to the all-region pre-test freeze, without loading labels."""
    study = Path(study).resolve()
    whole_path = located(study, "all_regions_selection_frozen.json")
    sidecar = located(study, "all_regions_selection_frozen.sha256")
    if digest(whole_path) != sidecar.read_text(encoding="ascii").strip():
        raise ValueError("All-region selection freeze SHA256 mismatch")
    whole = read_json(whole_path)
    if whole.get("test_metrics_computed") is not False:
        raise ValueError("Expected choices frozen before final testing")
    verified(study, "study_protocol.json", whole["protocol_sha256"])
    for name, expected in whole["source_code_sha256"].items():
        verified(study, Path("code") / name, expected)
    # Imported inference modules must be the verified study copies, not an
    # unrelated implementation found on sys.path or supplied via --study.
    for name in ("experiment_core.py", "hybrid_cycle_rbf.py"):
        if digest(Path(__file__).resolve().parent / name) != whole["source_code_sha256"][name]:
            raise ValueError(f"The executing {name} differs from the frozen study")
    choice_path = verified(study, f"experiments/{region}/final_choice.json", whole["choices_sha256"][region])
    choice = read_json(choice_path)
    if choice != whole["plans"][region] or choice.get("region") != region:
        raise ValueError("Regional choice does not match all-region freeze")
    if choice.get("candidate_test_predictions_used_for_selection") is not False:
        raise ValueError("Choice does not explicitly exclude test-based selection")
    for relative, expected in choice["prerequisite_freezes"].items():
        verified(study, relative, expected)
    cache_path = verified(study, f"cache_v1/{region}.npz", choice["cache_sha256"])
    split_path = verified(study, "cache_v1/split.json", choice["split_sha256"])
    split = read_json(split_path)
    if split.get("original_ids") != choice["new_splitIDs"]:
        raise ValueError("Original case-ID split differs from frozen choice")
    model_path = verified(study, choice[f"{role}_model_relative_path"], choice[f"{role}_model_sha256"])
    return choice, split, cache_path, model_path, digest(whole_path)


def load_input_metadata(cache_path, region):
    """Explicit whitelist: field, coeff, features and POD labels are not loaded."""
    with np.load(cache_path, allow_pickle=False) as archive:
        missing = set(INPUT_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"Missing input metadata: {sorted(missing)}")
        values = {name: archive[name] for name in INPUT_KEYS}
    metadata = json.loads(str(values["metadata_json"]))
    if metadata.get("region") != region or metadata.get("units") != "T" or metadata.get("xy_units") != "mm":
        raise ValueError("Unexpected region or physical units in cache metadata")
    if metadata.get("angle_kind") != "absolute mechanical":
        raise ValueError("This CLI requires saved absolute mechanical angles")
    components = metadata.get("components")
    if components not in (["Br", "Bt"], ["Br"]):
        raise ValueError("Only signed Br/Bt or signed Br caches are supported")
    if region == "airgap" and components != ["Br"]:
        raise ValueError("Expected signed Br-only airgap cache")
    if int(metadata.get("component_count", 0)) != len(components):
        raise ValueError("Component count differs from component names")
    expected_layout = "interleaved" if len(components) == 2 else "scalar"
    if metadata.get("layout") != expected_layout:
        raise ValueError("Unexpected component flatten order")
    n = len(values["case_ids"])
    frames = len(values["theta"])
    points = len(values["point_id"])
    dimension = 2 + 2 * len(values["harmonics"])
    if frames != 60 or metadata.get("time_count") != frames:
        raise ValueError("Only the 60 saved NEW60 time frames are supported")
    shapes = {"conditions": (n, 2), "theta": (frames,), "theta_m_relative_deg": (frames,),
              "time_s": (frames,), "solver_index": (frames,), "point_id": (points,), "xy": (points, 2),
              "feature_mean": (dimension,), "feature_scale": (dimension,)}
    for name, shape in shapes.items():
        if values[name].shape != shape or not np.isfinite(values[name]).all():
            raise ValueError(f"Invalid shape/nonfinite input metadata: {name}")
    if len(np.unique(values["case_ids"])) != n or len(np.unique(values["point_id"])) != points:
        raise ValueError("Duplicate case or point identifiers")
    if np.any(values["feature_scale"] <= 0) or not np.isfinite(values["harmonics"]).all():
        raise ValueError("Invalid input transform")
    initial = float(np.asarray(values["initial_mechanical_deg"]).item())
    if not math.isfinite(initial) or not np.allclose(values["theta"], values["theta_m_relative_deg"] + initial, atol=1e-8, rtol=0):
        raise ValueError("Absolute mechanical angles do not equal relative + initial")
    return values, metadata


def resolve_condition(values, case_id=None, current_rms=None, thet_deg=None):
    if case_id is not None:
        if current_rms is not None or thet_deg is not None:
            raise ValueError("Use case ID or I_RMS/Thet, not both")
        ids = values["case_ids"]
        matches = [i for i, value in enumerate(ids) if str(value) == str(case_id)]
        if not matches and np.issubdtype(ids.dtype, np.integer):
            try:
                query = int(case_id)
            except ValueError:
                query = None
            matches = [i for i, value in enumerate(ids) if query is not None and int(value) == query]
        if len(matches) != 1:
            raise ValueError(f"Original case ID not found uniquely: {case_id}")
        index = matches[0]
        return values["conditions"][index].astype(np.float64), ids[index].item(), index
    if current_rms is None or thet_deg is None:
        raise ValueError("Supply --case-id or both --current-rms and --thet-deg")
    condition = np.asarray([current_rms, thet_deg], dtype=np.float64)
    if not np.isfinite(condition).all() or condition[0] < 0:
        raise ValueError("Current must be nonnegative and both native inputs must be finite")
    return condition, "custom", None


def prepared_features(values, condition, indices):
    phase = np.deg2rad(values["theta"][indices])[:, None] * values["harmonics"][None, :]
    raw = np.concatenate((np.broadcast_to(condition, (len(indices), 2)), np.sin(phase), np.cos(phase)), axis=-1)
    # Preserve the float32 preparation boundary used by training and hybrid.
    return ((raw - values["feature_mean"]) / values["feature_scale"]).astype(np.float32)


def training_domain(conditions, query):
    """Pure-NumPy monotone-chain hull, used only as an input-domain flag."""
    points = np.asarray(conditions, dtype=np.float64)
    origin = points.mean(axis=0)
    spread = np.ptp(points, axis=0)
    spread[spread == 0] = 1
    points = (points - origin) / spread
    target = (np.asarray(query, dtype=np.float64) - origin) / spread
    unique = sorted(set(map(tuple, points.tolist())))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    tolerance = 1e-10
    if len(unique) == 1:
        inside = bool(np.linalg.norm(target - unique[0]) <= tolerance)
    elif len(hull) < 3:
        a, b = np.asarray(unique[0]), np.asarray(unique[-1])
        projection = float(np.dot(target-a, b-a) / np.dot(b-a, b-a))
        inside = bool(-tolerance <= projection <= 1+tolerance and abs(cross(a, b, target)) <= tolerance)
    else:
        inside = all(cross(hull[i], hull[(i+1) % len(hull)], target) >= -tolerance for i in range(len(hull)))
    return dict(inside_training_condition_convex_hull=bool(inside),
                native_input_order=["I_RMS_A", "Thet_deg"],
                training_input_min=np.min(conditions, axis=0).tolist(),
                training_input_max=np.max(conditions, axis=0).tolist(),
                assessment="input_interpolation_domain" if inside else "outside_training_hull_unvalidated_extrapolation",
                uncertainty_estimate_available=False,
                note="Convex-hull membership checks only inputs. It does not certify this query's prediction accuracy; extrapolation has not been validated.")


def output_coordinates(xy, absolute_angles, region, coordinates):
    xy = np.asarray(xy, dtype=np.float64)
    output = np.broadcast_to(xy, (len(absolute_angles), *xy.shape)).copy()
    if coordinates == "world" and region == "rotor":
        angle = np.deg2rad(absolute_angles)
        c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
        output[..., 0] = c * xy[:, 0] - s * xy[:, 1]
        output[..., 1] = s * xy[:, 0] + c * xy[:, 1]
    return output


def predict_field(model_path, values, condition, indices, frame_index=None):
    with np.load(model_path, allow_pickle=False) as archive:
        kind = json.loads(str(archive["metadata_json"])).get("format")
    if kind == "pod_mlp_numpy_v1":
        from experiment_core import predict
        result = predict(model_path, prepared_features(values, condition, indices))
    elif kind == "new60_mlp_cycle_rbf_v1":
        from hybrid_cycle_rbf import load_model, predict
        model = load_model(model_path)
        for target, source in (("theta", "theta"), ("harmonics", "harmonics"),
                               ("outer_feature_mean", "feature_mean"), ("outer_feature_scale", "feature_scale")):
            if not np.array_equal(model[target], values[source]):
                raise ValueError(f"Hybrid input transform differs from frozen cache: {target}")
        result = predict(model, condition, frame_index=frame_index)
        if frame_index is not None:
            result = result[None, :]
    else:
        raise ValueError(f"Unsupported frozen model format: {kind}")
    return np.asarray(result, dtype=np.float64), kind


def write_outputs(output, field, values, metadata, condition, case_id, indices, xy_output, provenance):
    output = Path(output)
    sidecar = output.with_suffix(output.suffix + ".metadata.json")
    if output.suffix.lower() not in (".csv", ".npz"):
        raise ValueError("Output suffix must be .csv or .npz")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Refusing to overwrite prediction output or metadata sidecar")
    k = len(metadata["components"])
    expected_shape = (len(indices), len(values["point_id"]), k)
    if field.shape != (len(indices), len(values["point_id"]) * k) or not np.isfinite(field).all():
        raise ValueError("Prediction dimensions/nonfinite values disagree with frozen geometry")
    components = field.reshape(expected_shape)
    named = {"Br_T": components[..., 0]}
    if k == 2:
        named["Bt_T"] = components[..., 1]
        named["B_magnitude_T"] = np.hypot(components[..., 0], components[..., 1])
    provenance["output_quantities"] = list(named)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".npz":
        with output.open("xb") as stream:
            np.savez_compressed(stream, **named, field=field, components=np.asarray(metadata["components"]),
                                case_id=np.asarray(str(case_id)), conditions=condition,
                                point_id=values["point_id"], xy_reference_mm=values["xy"], xy_mm=xy_output,
                                frame_index=indices, solver_index=values["solver_index"][indices],
                                time_s=values["time_s"][indices], theta_absolute_deg=values["theta"][indices],
                                theta_relative_deg=values["theta_m_relative_deg"][indices],
                                metadata_json=np.asarray(json.dumps(provenance, ensure_ascii=False, allow_nan=False)))
    else:
        header = ["case_id", "I_RMS_A", "Thet_deg", "frame_index", "solver_index", "time_s",
                  "theta_relative_deg", "theta_absolute_deg", "point_id", "x_reference_mm", "y_reference_mm",
                  "x_mm", "y_mm", "coordinate_frame", *named]
        with output.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for j, frame in enumerate(indices):
                for p, point in enumerate(values["point_id"]):
                    writer.writerow([case_id, *condition.tolist(), int(frame), int(values["solver_index"][frame]),
                                     float(values["time_s"][frame]), float(values["theta_m_relative_deg"][frame]),
                                     float(values["theta"][frame]), point.item(), *values["xy"][p].tolist(),
                                     *xy_output[j, p].tolist(), provenance["output_coordinate_frame"],
                                     *[float(array[j, p]) for array in named.values()]])
    provenance["output_sha256"] = digest(output)
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(output=str(output.resolve()), metadata=str(sidecar.resolve()),
                frames=len(indices), points=len(values["point_id"]), quantities=list(named),
                field_labels_loaded=False, accuracy_metrics_computed=False)


def run(args):
    output = args.output.resolve()
    if output.suffix.lower() not in (".csv", ".npz"):
        raise ValueError("Output suffix must be .csv or .npz")
    if output.exists() or output.with_suffix(output.suffix + ".metadata.json").exists():
        raise FileExistsError("Refusing to overwrite prediction output or metadata sidecar")
    if args.case_id is not None and (args.current_rms is not None or args.thet_deg is not None):
        raise ValueError("Use --case-id or native current/Thet inputs, not both")
    choice, split, cache_path, model_path, freeze_hash = verify_frozen_inputs(args.study, args.region, args.model)
    values, metadata = load_input_metadata(cache_path, args.region)
    condition, case_id, case_index = resolve_condition(values, args.case_id, args.current_rms, args.thet_deg)
    if args.frame is not None and not 0 <= args.frame < len(values["theta"]):
        raise ValueError("--frame is a zero-based stored index from 0 to 59")
    indices = np.arange(len(values["theta"]), dtype=np.int64) if args.frame is None else np.asarray([args.frame], dtype=np.int64)
    training = np.asarray(split["train"], dtype=np.int64)
    if training.ndim != 1 or len(training) != 42 or len(set(training.tolist())) != 42 or np.any(training < 0) or np.any(training >= len(values["case_ids"])):
        raise ValueError("Unexpected frozen training indices")
    if values["case_ids"][training].tolist() != split["original_ids"]["train"]:
        raise ValueError("Training indices disagree with original case IDs")
    domain = training_domain(values["conditions"][training], condition)
    field, model_format = predict_field(model_path, values, condition, indices, args.frame)
    xy = output_coordinates(values["xy"], values["theta"][indices], args.region, args.coordinates)
    actual_frame = "world" if args.coordinates == "world" else metadata["frame"]
    provenance = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
                      dataset=choice["dataset"], region=args.region, model_role=args.model,
                      selected_name=choice["selected_name"] if args.model == "selected" else "uniform_field_control_seed42",
                      model_format=model_format, model_relative_path=choice[f"{args.model}_model_relative_path"],
                      model_sha256=choice[f"{args.model}_model_sha256"], cache_sha256=choice["cache_sha256"],
                      split_sha256=choice["split_sha256"], all_regions_selection_frozen_sha256=freeze_hash,
                      inference_code_sha256=digest(Path(__file__)), quantity=metadata["quantity"],
                      field_units="T", coordinate_units="mm", output_coordinate_frame=actual_frame,
                      component_basis="signed local polar Br/Bt; rotation changes point coordinates, not these polar values",
                      input_current_convention="native I_RMS in amperes; no additional division or peak conversion",
                      input_angle_convention="native Thet in degrees; not asserted to equal dq beta",
                      native_input=dict(I_RMS_A=float(condition[0]), Thet_deg=float(condition[1])),
                      original_case_id=case_id, known_case_index=case_index,
                      stored_frame_indices=indices.tolist(), initial_mechanical_deg=float(values["initial_mechanical_deg"]),
                      absolute_angle_rule="theta_absolute_deg = theta_relative_deg + initial_mechanical_deg",
                      coordinate_rule="rotor world xy = rotation(theta_absolute_deg) @ xy_reference; stator and airgap remain fixed",
                      field_labels_loaded=False, coefficient_labels_loaded=False, accuracy_metrics_computed=False,
                      training_or_selection_performed=False, candidate_test_predictions_used_for_selection=False,
                      input_domain=domain, historical_round4_reproduced=False,
                      limitations=["Only 60 saved time frames and original sampled points; no arbitrary time/space prediction.",
                                   "No demonstrated extrapolation validity; input-domain membership is not an accuracy guarantee.",
                                   "Br and Bt are signed polar components. Airgap has Br only, so no full |B| is supplied.",
                                   "First supplied electrical period; steady-state certification is not available."])
    return write_outputs(output, field, values, metadata, condition, case_id, indices, xy, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--region", choices=REGIONS, required=True)
    parser.add_argument("--model", choices=("selected", "control"), default="selected")
    parser.add_argument("--case-id", help="Original source case ID, e.g. 2 or 02; not a split row number")
    parser.add_argument("--current-rms", type=float, help="Native model current I_RMS in A")
    parser.add_argument("--thet-deg", type=float, help="Native Thet control angle in degrees")
    parser.add_argument("--frame", type=int, help="Stored zero-based frame 0..59; omitted = all 60 frames, solver steps 1..60")
    parser.add_argument("--coordinates", choices=("reference", "world"), default="reference")
    parser.add_argument("--output", type=Path, required=True, help="New .csv or .npz path; existing outputs are rejected")
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
