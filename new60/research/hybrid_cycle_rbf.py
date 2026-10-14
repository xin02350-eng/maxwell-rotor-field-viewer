"""Frozen MLP plus condition-RBF residuals for supplied whole-cycle frames.

fit reads field/coeff labels from training and validation cases only. Prediction
uses NumPy only and accepts conditions, never arbitrary time/space coordinates.
The same 42 training centers/kernel fit pure POD coefficients as an ablation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

LENGTHSCALES = (0.6, 1.2, 2.4, 4.8)
RIDGES = (1e-8, 1e-6, 1e-4)
FORMAT = "new60_mlp_cycle_rbf_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _read_rows(path, key, cases, total):
    """Skip excluded case payloads without converting them to numeric arrays."""
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if fortran or len(shape) != 3 or shape[0] != total or dtype.kind != "f":
            raise ValueError(f"{key} must be C-order floating [case,time,dim]")
        out = np.full(shape, np.nan, dtype=dtype)
        size = int(np.prod(shape[1:])) * dtype.itemsize
        for case in range(total):
            if case in cases:
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError("Truncated label row")
                out[case] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
            else:
                stream.seek(size, 1)
    return out


def _load_cache(cache, split):
    split = _read_json(split) if isinstance(split, (str, Path)) else split
    groups = {}
    for name, count in (("train", 42), ("validation", 9), ("test", 9)):
        row = split.get(name, [])
        if len(row) != count or any(type(i) is not int or not 0 <= i < 60 for i in row):
            raise ValueError("NEW60 requires a disjoint complete 42/9/9 case-index split")
        groups[name] = row
    if len(set(sum(groups.values(), []))) != 60 or split.get("dataset") != "NEW60":
        raise ValueError("Invalid NEW60 partition")
    needed = {"features", "basis", "mean", "scale", "case_ids", "point_id", "xy", "theta",
              "conditions", "feature_mean", "feature_scale", "harmonics", "metadata_json", "pod_train_case_indices"}
    selected = groups["train"] + groups["validation"]
    if isinstance(cache, (str, Path)):
        with np.load(cache, allow_pickle=False) as z:
            if not needed.issubset(z.files):
                raise ValueError(f"Cache fields missing: {needed-set(z.files)}")
            a = {k: z[k] for k in needed}
        for key in ("field", "coeff"):
            a[key] = _read_rows(cache, key, selected, 60)
    else:
        a = {k: np.asarray(cache[k]) for k in needed}
        for key in ("field", "coeff"):
            source = np.asarray(cache[key])
            a[key] = np.full(source.shape, np.nan, dtype=source.dtype)
            a[key][selected] = source[selected]
    meta = json.loads(str(a["metadata_json"]))
    schema = {"Br,Bt": (["Br", "Bt"], "interleaved"), "Br": (["Br"], "scalar"), "|B|": (["|B|"], "scalar")}
    if meta.get("quantity") not in schema:
        raise ValueError("Explicit physical quantity required")
    components, layout = schema[meta["quantity"]]
    if (meta.get("components") != components or meta.get("component_count") != len(components)
            or meta.get("layout") != layout or meta.get("units") != "T" or meta.get("xy_units") != "mm"
            or meta.get("dataset") != "NEW60" or meta.get("basis_fit") != "train_cases_only"):
        raise ValueError("Quantity, units, or training-only POD metadata mismatch")
    ids = a["case_ids"].tolist()
    if len(ids) != 60 or len(set(ids)) != 60 or split.get("case_ids") != ids:
        raise ValueError("Case identity mismatch")
    for name, rows in groups.items():
        if split.get("original_ids", {}).get(name) != [ids[i] for i in rows]:
            raise ValueError("Original case-ID split mismatch")
    if sorted(a["pod_train_case_indices"].tolist()) != sorted(groups["train"]):
        raise ValueError("POD training cases differ")
    c, t, f = a["features"].shape
    d, r = a["basis"].shape
    p = len(a["point_id"])
    if (c != 60 or t < 1 or a["field"].shape != (c,t,d) or a["coeff"].shape != (c,t,r)
            or a["mean"].shape != (d,) or a["scale"].shape != (r,) or d != p*len(components)
            or a["conditions"].shape != (60,2) or a["theta"].shape != (t,) or a["xy"].shape != (p,2)
            or a["feature_mean"].shape != (f,) or a["feature_scale"].shape != (f,)):
        raise ValueError("Cache dimension mismatch")
    for key in ("features", "conditions", "theta", "basis", "mean", "scale", "feature_mean", "feature_scale", "xy"):
        if not np.isfinite(a[key]).all():
            raise ValueError(f"Nonfinite {key}")
    if np.any(a["scale"] <= 0) or np.any(a["feature_scale"] <= 0):
        raise ValueError("Nonpositive normalization scale")
    if len(np.unique(a["point_id"])) != p or p < 2:
        raise ValueError("Unique point IDs and at least two points required")
    for key in ("field", "coeff"):
        if not np.isfinite(a[key][selected]).all() or not np.isnan(a[key][groups["test"]]).all():
            raise ValueError("Invalid selected labels or excluded-label isolation")
    if isinstance(cache, (str, Path)):
        condition_file = Path(cache).parent / "case_conditions.json"
        expected = dict(dataset="NEW60", case_ids=ids, columns=["current", "current_angle"], conditions=a["conditions"].tolist())
        if (not condition_file.is_file() or digest(condition_file) != meta.get("case_conditions_sha256")
                or _read_json(condition_file) != expected):
            raise ValueError("Canonical case-condition identity mismatch")
    return a, split, meta


def _mlp_coeff(model, x):
    y = (np.asarray(x, dtype=np.float64)-model["feature_mean"])/model["feature_scale"]
    layers = int(model["layer_count"])
    for i in range(layers):
        y = y @ model[f"w{i}"] + model[f"b{i}"]
        if i+1 < layers:
            y = np.tanh(y)
    y = y * model["scale"]
    if int(model.get("has_base", 0)):
        y += _mlp_coeff({k[6:]: v for k,v in model.items() if k.startswith("base__")}, x)
    return y


def _features(model, conditions, indices):
    theta = np.deg2rad(model["theta"][indices])[:,None] * model["harmonics"][None,:]
    periodic = np.concatenate((np.sin(theta), np.cos(theta)), axis=-1)
    raw = np.concatenate((np.broadcast_to(conditions[:,None,:], (len(conditions),len(indices),2)),
                          np.broadcast_to(periodic, (len(conditions),*periodic.shape))), axis=-1)
    return ((raw-model["outer_feature_mean"])/model["outer_feature_scale"]).astype(np.float32)


def _kernel(x, centers, lengthscale):
    squared = np.sum((x[:,None,:]-centers[None,:,:])**2, axis=-1)
    return np.exp(-0.5*squared/(lengthscale*lengthscale))


def load_model(path, expected_sha256=None):
    """A saved model must have its matching immutable SHA sidecar."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="ascii").strip()
    actual = digest(path)
    if actual != expected or (expected_sha256 is not None and actual != expected_sha256):
        raise ValueError("Model SHA256 mismatch")
    with np.load(path, allow_pickle=False) as z:
        model = {k:z[k] for k in z.files}
    if json.loads(str(model["metadata_json"])).get("format") != FORMAT:
        raise ValueError("Unsupported hybrid model format")
    return model


def predict(model, conditions, frame_index=None, mode=None, return_coeff=False):
    """[2] -> [T,D], [N,2] -> [N,T,D]; integer frame_index removes T.

    modes: hybrid, mlp_only, rbf_only. rbf_only uses the parameters stored in
    this artifact. Best-independent RBF parameters are in rbf_only_model.npz.
    Only saved time frames and fixed spatial points are supported.
    """
    model = load_model(model) if isinstance(model, (str,Path)) else model
    if mode is None:
        mode = json.loads(str(model.get("metadata_json", '{}'))).get("default_mode", "hybrid")
    x = np.asarray(conditions, dtype=np.float64)
    one = x.ndim == 1
    if one:
        x = x[None,:]
    if x.ndim != 2 or x.shape[-1] != 2 or not np.isfinite(x).all() or len(x)==0:
        raise ValueError("Expected finite [2] or [N,2] raw conditions")
    if mode not in ("hybrid", "mlp_only", "rbf_only"):
        raise ValueError("Unknown prediction mode")
    t = len(model["theta"])
    if frame_index is not None and (isinstance(frame_index, (bool,np.bool_)) or not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<t):
        raise ValueError("frame_index must identify one stored time frame")
    indices = np.arange(t) if frame_index is None else np.asarray([frame_index])
    kernel = _kernel((x-model["condition_mean"])/model["condition_scale"], model["centers"], float(model["lengthscale"]))
    r = model["basis"].shape[1]
    if mode == "rbf_only":
        coeff = (kernel @ model["rbf_only_weights"].reshape(len(model["centers"]),-1)).reshape(len(x),t,r)[:,indices]
    else:
        mlp = {k[5:]:v for k,v in model.items() if k.startswith("mlp__")}
        coeff = _mlp_coeff(mlp, _features(model,x,indices))
        if mode == "hybrid":
            coeff += (kernel @ model["residual_weights"].reshape(len(model["centers"]),-1)).reshape(len(x),t,r)[:,indices]
    output = coeff if return_coeff else model["mean"] + coeff @ model["basis"].T
    if frame_index is not None:
        output = output[:,0]
    return output[0] if one else output


def _metrics(prediction, truth):
    y = np.asarray(truth,dtype=np.float64)
    e = np.asarray(prediction,dtype=np.float64)-y
    den = float(np.sum(y*y))
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))), mae_T=float(np.mean(np.abs(e))),
                max_abs_T=float(np.max(np.abs(e))), relative_L2_pct=float(100*np.sqrt(np.sum(e*e)/den)) if den else None)


def group_metrics(prediction, truth, case_ids, hotspot, metadata):
    n = metadata["component_count"]
    mask = np.repeat(hotspot,n)
    rows = {str(case):_metrics(prediction[i],truth[i]) for i,case in enumerate(case_ids)}
    result = dict(quantity=metadata["quantity"], units="T", global_metrics=_metrics(prediction,truth),
                  hotspot=_metrics(prediction[...,mask],truth[...,mask]), ordinary=_metrics(prediction[...,~mask],truth[...,~mask]),
                  per_case=rows, worst_case_rmse_T=max(row["rmse_T"] for row in rows.values()))
    result["components"] = {component:_metrics(prediction[...,i::n],truth[...,i::n]) for i,component in enumerate(metadata["components"])}
    if metadata["quantity"] == "|B|":
        result.update(negative_prediction_count=int(np.count_nonzero(prediction<0)),minimum_prediction_T=float(np.min(prediction)))
    return result


def _save_model(path, model):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream,**model)
    sha = digest(path)
    with Path(str(path)+".sha256").open("x",encoding="ascii") as stream:
        stream.write(sha+"\n")
    return sha


def fit(cache, split, mlp_model_path, outdir, fixed_hotspot_mask, lengthscale=None, ridge=None):
    """Fit twelve validation candidates or one fixed pair for seed replication.

    cache/split may be paths or in-memory values. Paths enforce canonical
    condition-file identity and decode only train/validation label rows. The
    caller must freeze the reporting hotspot mask from training labels only.
    """
    if (lengthscale is None) != (ridge is None):
        raise ValueError("Provide both fixed lengthscale/ridge or neither")
    pairs = [(l,r) for l in LENGTHSCALES for r in RIDGES] if lengthscale is None else [(float(lengthscale),float(ridge))]
    if any(not np.isfinite(l+r) or l<=0 or r<=0 for l,r in pairs):
        raise ValueError("Positive finite RBF parameters required")
    outdir, mlp_model_path = Path(outdir),Path(mlp_model_path)
    if outdir.exists():
        raise FileExistsError("Immutable output already exists; use a fresh directory")
    source_hash = digest(mlp_model_path)
    cache_hash = digest(cache) if isinstance(cache,(str,Path)) else None
    split_hash = digest(split) if isinstance(split,(str,Path)) else _json_hash(split)
    a, split_data, meta = _load_cache(cache,split)
    tr, va = np.asarray(split_data["train"]), np.asarray(split_data["validation"])
    with np.load(mlp_model_path,allow_pickle=False) as z:
        mlp = {k:z[k] for k in z.files}
    neural = json.loads(str(mlp["metadata_json"]))
    for key in ("basis","mean","scale","point_id","xy"):
        if not np.array_equal(mlp[key],a[key]):
            raise ValueError(f"MLP/cache {key} differ")
    for key, expected in (("train_case_indices",tr.tolist()),("validation_case_indices",va.tolist()),("excluded_test_case_indices",split_data["test"])):
        if sorted(neural.get(key,[])) != sorted(expected):
            raise ValueError("MLP train/validation/test provenance mismatch")
    t = a["features"].shape[1]
    if (neural.get("test_labels_accessed") is not False or neural.get("checkpoint_partition")!="validation"
            or neural.get("selected_training_samples") != len(tr)*t or neural.get("selected_validation_samples")!=len(va)*t):
        raise ValueError("MLP must be a validation-selected full-frame model with test labels excluded")
    if isinstance(fixed_hotspot_mask,(str,Path)):
        mask_hash = digest(fixed_hotspot_mask)
        with np.load(fixed_hotspot_mask,allow_pickle=False) as z:
            key = "hotspot_mask" if "hotspot_mask" in z.files else "hotspot"
            hotspot = z[key]
            if "point_id" not in z.files or not np.array_equal(z["point_id"],a["point_id"]):
                raise ValueError("Fixed hotspot point identity mismatch")
            if "train_case_indices" in z.files and sorted(z["train_case_indices"].tolist())!=sorted(tr.tolist()):
                raise ValueError("Fixed hotspot training-case identity mismatch")
    else:
        hotspot = np.asarray(fixed_hotspot_mask)
        mask_hash = _array_hash(hotspot)
    if hotspot.dtype.kind!="b" or hotspot.shape!=(len(a["point_id"]),) or not hotspot.any() or hotspot.all():
        raise ValueError("A nonempty boolean hotspot and ordinary region are required")
    mean = a["conditions"][tr].mean(axis=0)
    std = a["conditions"][tr].std(axis=0)
    std = np.where(std>1e-12,std,1.0)
    centers = (a["conditions"][tr]-mean)/std
    if len(np.unique(centers,axis=0))!=len(tr):
        raise ValueError("RBF training conditions must be distinct")
    model = dict(condition_mean=mean,condition_scale=std,centers=centers,
                 basis=a["basis"].astype(np.float64),mean=a["mean"].astype(np.float64),
                 theta=a["theta"],harmonics=a["harmonics"],outer_feature_mean=a["feature_mean"],outer_feature_scale=a["feature_scale"],
                 point_id=a["point_id"],xy=a["xy"],train_case_ids=a["case_ids"][tr],train_case_indices=tr,hotspot_mask=hotspot)
    model.update({"mlp__"+k:v for k,v in mlp.items()})
    rebuilt = _features(model,a["conditions"][np.r_[tr,va]],np.arange(t))
    if not np.allclose(rebuilt,a["features"][np.r_[tr,va]],rtol=1e-6,atol=1e-7):
        raise ValueError("Stored feature transform does not reproduce MLP cache inputs")
    # Only the reviewed local engine is used to confirm exact neural inference.
    from experiment_core import predict as engine_predict
    neural_train = engine_predict(mlp,a["features"][tr],return_coeff=True)
    neural_val = engine_predict(mlp,a["features"][va],return_coeff=True)
    if not np.allclose(neural_train,_mlp_coeff(mlp,a["features"][tr]),rtol=1e-12,atol=1e-12):
        raise AssertionError("Embedded NumPy MLP differs from the engine")
    target = a["coeff"][tr].astype(np.float64)
    residual = target-neural_train
    truth = a["field"][va].astype(np.float64)
    val_ids = a["case_ids"][va].tolist()
    neural_validation = group_metrics(model["mean"]+neural_val@model["basis"].T,truth,val_ids,hotspot,meta)
    outdir.mkdir(parents=True,exist_ok=False)
    source = Path(__file__).resolve()
    shutil.copyfile(source,outdir/source.name)
    identity = dict(format=FORMAT,source_mlp_sha256=source_hash,cache_sha256=cache_hash,split_sha256=split_hash,
                    fixed_hotspot_sha256=mask_hash,code_sha256=digest(source),engine_sha256=digest(Path(__file__).with_name("experiment_core.py")),
                    source_mlp_seed=neural.get("seed"),train_case_ids=a["case_ids"][tr].tolist(),validation_case_ids=val_ids,
                    case_split=split_data,input_metadata=meta,test_labels_loaded=False,test_metrics_computed=False,
                    analytic_fit_epochs=0,mlp_weights_frozen=True,condition_normalization_fit="train42_only",
                    rbf_target_layout="training_case x supplied_frame x raw_POD_coefficient",
                    kernel="exp(-0.5*||normalized_condition-center||^2/lengthscale^2)",solver="(K+ridge*I) weights=targets; no intercept",
                    parameter_selection="validation full-field RMSE only" if lengthscale is None else "fixed pair; no hyperparameter search",
                    scope="fixed spatial point IDs and supplied time frames; unseen conditions only",
                    not_claimed=["unseen_time","unseen_space","steady_state_certification","Round4_reproduction"],
                    hotspot_source_requirement="caller-frozen training-only mask; no candidate-specific mask tuning")
    _write_new(outdir/"run_identity.json",identity)
    rows = []
    solutions = []
    try:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=4)
    except ImportError:
        context = nullcontext()
    with context:
        for l, regularizer in pairs:
            kernel = _kernel(centers,centers,l)
            lhs = kernel+regularizer*np.eye(len(tr))
            stacked = np.concatenate((residual.reshape(len(tr),-1),target.reshape(len(tr),-1)),axis=1)
            weights = np.linalg.solve(lhs,stacked)
            width = t*target.shape[-1]
            correction = weights[:,:width].reshape(target.shape)
            pure = weights[:,width:].reshape(target.shape)
            query = _kernel((a["conditions"][va]-mean)/std,centers,l)
            hybrid_coeff = neural_val+(query@correction.reshape(len(tr),-1)).reshape(neural_val.shape)
            pure_coeff = (query@pure.reshape(len(tr),-1)).reshape(neural_val.shape)
            hybrid_metrics = group_metrics(model["mean"]+hybrid_coeff@model["basis"].T,truth,val_ids,hotspot,meta)
            pure_metrics = group_metrics(model["mean"]+pure_coeff@model["basis"].T,truth,val_ids,hotspot,meta)
            rows.append(dict(lengthscale=l,ridge=regularizer,kernel_condition_number=float(np.linalg.cond(lhs)),
                             hybrid_validation=hybrid_metrics,rbf_only_validation=pure_metrics,analytic_epochs=0))
            solutions.append((correction,pure))
    chosen = min(range(len(rows)),key=lambda i:(rows[i]["hybrid_validation"]["global_metrics"]["rmse_T"],i))
    pure_chosen = min(range(len(rows)),key=lambda i:(rows[i]["rbf_only_validation"]["global_metrics"]["rmse_T"],i))
    if digest(mlp_model_path)!=source_hash or (cache_hash is not None and digest(cache)!=cache_hash):
        raise RuntimeError("An input changed during the fit")
    if isinstance(split,(str,Path)) and digest(split)!=split_hash:
        raise RuntimeError("Split changed during the fit")
    if isinstance(fixed_hotspot_mask,(str,Path)) and digest(fixed_hotspot_mask)!=mask_hash:
        raise RuntimeError("Frozen mask changed during the fit")
    artifact_hashes = {}
    for filename,index,default_mode in (("model.npz",chosen,"hybrid"),("rbf_only_model.npz",pure_chosen,"rbf_only")):
        arrays = dict(model,lengthscale=np.asarray(rows[index]["lengthscale"]),ridge=np.asarray(rows[index]["ridge"]),
                      residual_weights=solutions[index][0],rbf_only_weights=solutions[index][1])
        metadata = dict(identity,quantity=meta["quantity"],components=meta["components"],component_count=meta["component_count"],
                        layout=meta["layout"],units="T",xy_units="mm",frame_count=t,default_mode=default_mode,
                        selected_parameters=dict(lengthscale=rows[index]["lengthscale"],ridge=rows[index]["ridge"]))
        arrays["metadata_json"] = np.asarray(json.dumps(metadata,ensure_ascii=False,allow_nan=False))
        artifact_hashes[filename] = _save_model(outdir/filename,arrays)
        # Verify save/load and whole-cycle versus single-frame inference.
        replay = predict(outdir/filename,a["conditions"][va],mode=default_mode)
        expected = rows[index][default_mode+"_validation"]["global_metrics"]["rmse_T"]
        observed = _metrics(replay,truth)["rmse_T"]
        if not np.isclose(observed,expected,rtol=1e-8,atol=1e-10):
            raise AssertionError("Saved model metric replay differs")
        if not np.allclose(replay[:,0],predict(outdir/filename,a["conditions"][va],frame_index=0,mode=default_mode),rtol=1e-10,atol=1e-10):
            raise AssertionError("Stored-frame inference differs from cycle inference")
    hybrid_rmse=rows[chosen]["hybrid_validation"]["global_metrics"]["rmse_T"]
    same_rbf=rows[chosen]["rbf_only_validation"]["global_metrics"]["rmse_T"]
    best_rbf=rows[pure_chosen]["rbf_only_validation"]["global_metrics"]["rmse_T"]
    frozen = dict(format=FORMAT,identity_sha256=digest(outdir/"run_identity.json"),candidate_count=len(rows),
                  selected_hybrid=rows[chosen],best_rbf_only=rows[pure_chosen],mlp_only_validation=neural_validation,
                  neural_contribution=dict(hybrid_minus_same_parameter_rbf_RMSE_T=hybrid_rmse-same_rbf,
                                           hybrid_minus_best_rbf_RMSE_T=hybrid_rmse-best_rbf,
                                           improves_same_parameter_rbf=bool(hybrid_rmse<same_rbf),improves_best_rbf=bool(hybrid_rmse<best_rbf),
                                           interpretation="Negative RMSE difference supports a benefit of the frozen MLP on these validation cases; otherwise do not claim a neural contribution."),
                  rows=rows,artifact_sha256=artifact_hashes,test_metrics_computed=False,analytic_epochs=0,
                  selected_by="validation global field RMSE",seed=neural.get("seed"),scope=identity["scope"])
    _write_new(outdir/"selection_frozen.json",frozen)
    with (outdir/"selection_frozen.sha256").open("x",encoding="ascii") as stream:
        stream.write(digest(outdir/"selection_frozen.json")+"\n")
    return frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("cache","split","mlp-model","hotspot-mask","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--lengthscale",type=float)
    parser.add_argument("--ridge",type=float)
    args=parser.parse_args()
    result=fit(args.cache,args.split,args.mlp_model,args.output,args.hotspot_mask,args.lengthscale,args.ridge)
    print(json.dumps(dict(output=str(args.output),candidate_count=result["candidate_count"],
                          lengthscale=result["selected_hybrid"]["lengthscale"],ridge=result["selected_hybrid"]["ridge"],
                          test_metrics_computed=False),ensure_ascii=False))


if __name__=="__main__":
    main()
