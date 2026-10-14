"""Strict NEW60 adapter: canonical original IDs, immutable split, train-only POD.

Input NPZ: field[60,T,D], conditions[60,2], theta[T] in mechanical degrees,
point_id[P], xy[P,2] in mm, case_ids[60]. Quantity is explicitly supplied;
Br/Bt is point-major interleaved, independent of the region name.
Original-ID split has 42/9/9 cases. No test metrics are computed here.
"""
from pathlib import Path
import argparse, hashlib, json, os
import numpy as np
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _immutable_json(path, expected):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8-sig")) != expected:
            raise ValueError(f"Existing shared identity differs: {path}; use a new version directory")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, ensure_ascii=False, indent=2, allow_nan=False)


def quantity_spec(quantity):
    """The quantity is a source-data declaration, never inferred from region/D."""
    components = {"Br,Bt": ["Br", "Bt"], "Br": ["Br"], "|B|": ["|B|"]}
    if quantity not in components:
        raise ValueError("Explicit quantity must be Br,Bt, Br, or |B|")
    names = components[quantity]
    return dict(quantity=quantity, components=names, component_count=len(names),
                layout="interleaved" if len(names) == 2 else "scalar",
                source_quantity="signed Br/Bt interleaved by point" if len(names) == 2
                else ("signed Br" if quantity == "Br" else "magnetic flux density magnitude |B|"))


def prepare(input_path, split_path, region, rank, harmonics, angle_kind, output, *, quantity):
    input_path, split_path, output = map(Path, (input_path, split_path, output))
    if region not in ("stator", "rotor", "airgap") or angle_kind not in ("relative", "absolute"):
        raise ValueError("Explicit region and verified mechanical angle convention required")
    specification = quantity_spec(quantity)
    if output.suffix.lower() != ".npz":
        raise ValueError("Output must be an explicit .npz path")
    with np.load(input_path, allow_pickle=False) as data:
        required = {"field", "conditions", "theta", "point_id", "xy", "case_ids"}
        if not required.issubset(data.files):
            raise ValueError(f"Missing {required-set(data.files)}")
        arrays = {key: data[key] for key in required}
        optional = {key: data[key] for key in (
            'dataset_hotspot', 'solver_index', 'time_s', 'theta_m_relative_deg',
            'theta_e_deg', 'initial_mechanical_deg') if key in data.files}
    field = np.asarray(arrays["field"], dtype=np.float32)
    id_array = arrays["case_ids"]
    if id_array.ndim != 1 or id_array.dtype.kind not in "iuUS":
        raise ValueError("Original case IDs must be 1D integers or strings")
    ids = id_array.tolist()
    if field.ndim != 3 or field.shape[0] != 60 or len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError("NEW60 requires 60 distinct real operating conditions")
    if id_array.dtype.kind in "US" and any(not str(i).strip() for i in ids):
        raise ValueError("Empty case ID")
    ntime, dimension = field.shape[1:]
    point_ids, xy, theta = arrays["point_id"], arrays["xy"], arrays["theta"]
    if ntime < 1 or point_ids.ndim != 1 or len(point_ids) == 0:
        raise ValueError("Nonempty time axis and 1D point IDs required")
    npoints = len(point_ids)
    for key, value in optional.items():
        expected = (npoints,) if key == 'dataset_hotspot' else (() if key == 'initial_mechanical_deg' else (ntime,))
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f'Invalid optional data axis: {key}')
    if len(np.unique(point_ids)) != npoints:
        raise ValueError("Point IDs must be unique")
    if arrays["conditions"].shape != (60, 2) or theta.shape != (ntime,):
        raise ValueError("conditions[60,2] and common theta[T] required")
    if dimension != npoints*specification["component_count"] or xy.shape != (npoints, 2):
        raise ValueError("Point-major field components or xy dimensions mismatch")
    for name in ("field", "conditions", "theta", "xy"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Nonfinite {name}")
    if not np.isfinite(field).all():
        raise ValueError("Field overflows float32")
    if not 1 <= rank <= min(42*ntime-1, dimension):
        raise ValueError("POD rank must be positive and within training matrix dimensions")
    if (not harmonics or any(not isinstance(h, (int, np.integer)) or h <= 0 for h in harmonics)
            or len(set(harmonics)) != len(harmonics)):
        raise ValueError("Distinct positive integer mechanical harmonics required")
    supplied_split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    combined = []
    for key, count in (("train", 42), ("validation", 9), ("test", 9)):
        group = supplied_split.get(key)
        if not isinstance(group, list) or len(group) != count or any(i not in ids for i in group):
            raise ValueError(f"{key} must have {count} existing original case IDs")
        combined.extend(group)
    if len(set(combined)) != 60:
        raise ValueError("Split overlaps or is incomplete")
    canonical_ids = sorted(ids)
    order = np.asarray([ids.index(i) for i in canonical_ids])
    field = field[order]
    conditions = np.asarray(arrays["conditions"], dtype=np.float64)[order]
    split = {"dataset": "NEW60", "case_ids": canonical_ids}
    original_groups = {"dataset": "NEW60"}
    for key in ("train", "validation", "test"):
        split[key] = sorted(canonical_ids.index(i) for i in supplied_split[key])
        original_groups[key] = [canonical_ids[i] for i in split[key]]
    split["original_ids"] = {key: original_groups[key] for key in ("train", "validation", "test")}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent/".prepare_new60.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    temporary = output.with_name(output.name+".building.npz")
    try:
        if output.exists() or temporary.exists():
            raise FileExistsError("Output/staging exists; use a fresh version path")
        _immutable_json(output.parent/"case_order.json", {"dataset": "NEW60", "case_ids": canonical_ids})
        condition_identity = {"dataset": "NEW60", "case_ids": canonical_ids,
                              "columns": ["current", "current_angle"],
                              "conditions": conditions.tolist()}
        _immutable_json(output.parent/"case_conditions.json", condition_identity)
        _immutable_json(output.parent/"split.json", split)
        _immutable_json(output.parent/"split_original_case_ids.json", original_groups)
        train = np.asarray(split["train"], dtype=np.int64)
        snapshots = field[train].reshape(-1, dimension)
        mean = snapshots.mean(axis=0, dtype=np.float64).astype(np.float32)
        with threadpool_limits(limits=4):
            _, singular, vt = randomized_svd(snapshots-mean, n_components=rank, n_iter=5, random_state=42)
            basis = np.ascontiguousarray(vt.T, dtype=np.float32)
            coeff = (field-mean) @ basis
        raw_scale = coeff[train].reshape(-1, rank).std(axis=0, dtype=np.float64)
        scale_floor = float(max(raw_scale.max()*1e-5, 1e-8))
        scale = np.maximum(raw_scale, scale_floor).astype(np.float32)
        phase = np.deg2rad(theta)[:, None]*np.asarray(harmonics)[None, :]
        periodic = np.concatenate((np.sin(phase), np.cos(phase)), axis=-1)
        raw_features = np.concatenate((np.broadcast_to(conditions[:, None, :], (60, ntime, 2)),
                                       np.broadcast_to(periodic, (60, *periodic.shape))), axis=-1)
        feature_mean = raw_features[train].mean(axis=(0, 1))
        feature_scale = np.maximum(raw_features[train].std(axis=(0, 1)), 1e-8)
        features = ((raw_features-feature_mean)/feature_scale).astype(np.float32)
        oracle = {}
        with threadpool_limits(limits=4):
            for key in ("train", "validation"):
                truth = field[split[key]].astype(np.float64)
                prediction = mean+coeff[split[key]].astype(np.float64) @ basis.astype(np.float64).T
                error = prediction-truth
                norm = np.linalg.norm(truth)
                oracle[key] = {"rmse_T": float(np.sqrt(np.mean(error**2))),
                               "relative_L2_pct": float(100*np.linalg.norm(error)/norm) if norm else None}
        metadata = dict(dataset="NEW60", region=region, **specification, units="T", xy_units="mm",
                        frame="rotor_reference" if region == "rotor" else "stator_fixed",
                        case_count=60, time_count=ntime, rank=rank, harmonics=list(harmonics),
                        angle_kind=angle_kind+" mechanical", seed=42, basis_fit="train_cases_only",
                        source_case_ids=ids, case_ids=canonical_ids, source_reordered_to_canonical_ids=True,
                        generalization=f"unseen operating conditions; {ntime} supplied angle/time frames shared by training cases",
                        not_claimed=["unseen_time", "unseen_space", "physical_periodicity_without_external_verification"],
                        condition_columns=["current", "current_angle"],
                        condition_definition="Must be verified from source; current amplitude/RMS is not inferred",
                        case_conditions_sha256=sha256(output.parent/"case_conditions.json"),
                        scale_floor=scale_floor, oracle=oracle, test_metrics_computed=False,
                        source_sha256=sha256(input_path), source_split_sha256=sha256(split_path),
                        adapter_sha256=sha256(Path(__file__)),
                        basis_orthogonality_max=float(np.abs(basis.T @ basis-np.eye(rank)).max()))
        np.savez_compressed(temporary, features=features, field=field, basis=basis, mean=mean,
                            coeff=coeff.astype(np.float32), scale=scale, case_ids=np.asarray(canonical_ids),
                            point_id=point_ids, xy=xy, pod_train_case_indices=train, conditions=conditions,
                            theta=theta, feature_mean=feature_mean, feature_scale=feature_scale,
                            harmonics=np.asarray(harmonics), singular_values=singular,
                            metadata_json=json.dumps(metadata, ensure_ascii=False, allow_nan=False), **optional)
        if output.exists():
            raise FileExistsError("Output appeared while building; refusing overwrite")
        temporary.rename(output)
        return metadata
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--region", choices=["stator", "rotor", "airgap"], required=True)
    parser.add_argument("--quantity", choices=["Br,Bt", "Br", "|B|"], required=True,
                        help="Verified source quantity; Br,Bt requires D=2P, scalar quantities D=P")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--harmonics", type=int, nargs="+", required=True)
    parser.add_argument("--angle-kind", choices=["relative", "absolute"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = prepare(args.input, args.split, args.region, args.rank, args.harmonics, args.angle_kind,
                   args.output, quantity=args.quantity)
    print(json.dumps({"output": str(args.output), "dataset": "NEW60", "time_count": meta["time_count"],
                      "test_metrics_computed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

