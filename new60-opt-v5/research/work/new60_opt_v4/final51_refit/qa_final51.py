"""Final51 numerical/scaler/label-scope audit, without any test field labels."""
from pathlib import Path
import builtins
import csv
import hashlib
import importlib.util
import json
import numpy as np

HERE=Path(__file__).resolve().parent
PREPARED=HERE.parents[1]/"new60_opt_v3/joint_neural/point_prepared"
CACHE=HERE.parents[1]/"new60_20261014/cache_v1/rotor.npz"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    original_import=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.split(".")[0] in ("torch","sklearn"):raise ImportError("Torch/sklearn explicitly blocked during inference QA")
        return original_import(name,*args,**kwargs)
    builtins.__import__=blocked
    nn=module("final51_nn",HERE/"regime_final51.py");gp=module("final51_kernel",HERE/"kernel_final51.py")
    original_nn=module("original42_regime_readonly",HERE.parent/"point_regime_nn/regime_point_nn.py")
    protocol=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"));allowed=protocol["train_case_ids"];excluded=protocol["excluded_test_case_ids"]
    with np.load(PREPARED/"mean_data.npz",allow_pickle=False) as z:mean={k:z[k] for k in ("case_ids","conditions","signed_mean")}
    with np.load(PREPARED/"baseline_model.npz",allow_pickle=False) as z:base={k:z[k] for k in z.files}
    signed_truth=nn.read_selected_fields(CACHE,allowed);truth=np.hypot(signed_truth[...,0::2],signed_truth[...,1::2])
    checks=[]
    def check(name,value,detail=None):
        if not value:raise AssertionError(name+": "+str(detail))
        checks.append(dict(name=name,status="PASS",detail=detail))
    check("exact51 training labels and9 excluded",len(allowed)==51 and len(excluded)==9 and not(set(allowed)&set(excluded)) and mean["case_ids"].tolist()==allowed)
    check("protocol checksum unchanged",sha(HERE/"protocol.json")== (HERE/"protocol.json.sha256").read_text().strip())
    check("frozen source checksum",sha(HERE/"regime_final51.py")==protocol["neural_source_sha256"])
    for config in protocol["neural_candidates"]:
        folder=HERE/config["name"];m=nn.load_model(folder/"model.npz");meta=json.loads(str(m["metadata_json"]));result=json.loads((folder/"result.json").read_text(encoding="utf-8"));c=mean["conditions"]
        label=config["name"]
        check(label+" train51 scope",meta["train_case_ids"]==allowed and meta["validation_case_ids"]==[] and meta["excluded_case_ids"]==excluded and not meta["test_labels_loaded"] and not meta["full60_fit_labels_loaded"])
        check(label+" checkpoint is training only","TRAIN51" in meta["checkpoint"] and "no test" in meta["checkpoint"])
        check(label+" model/source/protocol hashes",result["model_sha256"]==sha(folder/"model.npz") and meta["source_sha256"]==sha(HERE/"regime_final51.py") and meta["protocol_sha256"]==sha(HERE/"protocol.json"))
        f=nn.features(c,"cartesian",float(c[:,0].max()));check(label+" scaler current is all51 max",float(m["current_scale"])==float(c[:,0].max()))
        check(label+" input mean/scale train51",np.array_equal(m["feature_mean"],f.mean(0)) and np.array_equal(m["feature_scale"],np.maximum(f.std(0),1e-8)))
        s=mean["signed_mean"].std(0);s=np.maximum(s,s.max()*.001)
        check(label+" output mean/scale train51",np.array_equal(m["output_mean"],mean["signed_mean"].mean(0)) and np.array_equal(m["output_scale"],s))
        check(label+" immutable train42 dynamic embedded",all(np.array_equal(m["baseline_"+k],v) for k,v in base.items()))
        subset=c[:3];p=nn.predict(m,subset);signed=nn.predict(m,subset,return_signed=True).reshape(3,60,1800,2)
        check(label+" magnitude formed after signed sum",np.array_equal(p,np.hypot(signed[...,0],signed[...,1])))
        check(label+" zero-mean dynamic",np.max(np.abs(signed.mean(1)-nn.predict_mean(m,subset)))<1e-12)
        check(label+" original equation exact full-field replay",np.array_equal(p,original_nn.predict(m,subset)))
        single_difference=float(np.max(np.abs(nn.predict(m,subset[0],frame_index=7)-p[0,7])))
        check(label+" single case/frame output",single_difference<1e-12,dict(max_abs_difference_T=single_difference))
        full=nn.predict(m,c);newmet=nn.metrics(full,truth,allowed,m["point_id"]);oldmet=result["train51_fitting_magnitude"]
        check(label+" all51×60×1800 raw metrics independently replay",all(abs(newmet[k]-oldmet[k])<1e-12 for k in ("relative_L2_pct","rmse_T","mae_T","max_abs_T")))
        with (folder/"history.csv").open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
        best=min(rows,key=lambda r:float(r["train51_signed_mean_relative_L2_pct"]))
        check(label+" trainloss checkpoint and complete budget",len(rows)==121 and int(rows[-1]["step"])==5200 and int(best["step"])==result["best_step"] and int(result["configuration"]["adam_steps"])==5000)
        check(label+" actual trained parameter count",result["new_trainable_parameters"]==1800*(6+config["gates"]*9))
        check(label+" Torch64/Numpy replay precision",result["numpy_torch_float64_replay_max_abs_difference_T"]<1e-10)
        for bad,title in (([[-1,50]],"negative native current"),([[np.nan,50]],"nonfinite native current")):
            try:nn.predict(m,bad)
            except ValueError:check(label+" rejects "+title,True)
            else:check(label+" rejects "+title,False)
        try:nn.predict(m,[240,55.5],frame_index=60)
        except ValueError:check(label+" rejects invalid frame",True)
        else:check(label+" rejects invalid frame",False)
        check(label+" NumPy-only novel input finite",np.isfinite(nn.predict(m,[240,55.5],frame_index=11)).all())
    folder=HERE/"kernel_rank24";m=gp.load_model(folder/"model.npz");meta=json.loads(str(m["metadata_json"]));result=json.loads((folder/"result.json").read_text(encoding="utf-8"));c=mean["conditions"]
    check("GP train51 only fixed recipe",meta["training_original_ids"]==allowed and meta["validation_original_ids"]==[] and meta["excluded_test_original_ids"]==excluded and meta["spec"]==protocol["kernel_fixed_spec"])
    check("GP model/source hash",result["model_sha256"]==sha(folder/"model.npz") and result["source_sha256"]==sha(HERE/"kernel_final51.py"))
    rawx=gp.coordinates(c,meta["spec"]["current_scale"])
    check("GP center/scales trained51 only",np.array_equal(m["center"],rawx.mean(0)) and np.array_equal(m["input_scale"],np.maximum(rawx.std(0),1e-10)))
    check("GP full-cycle signed mean trained51 only",np.array_equal(m["mean"],signed_truth.reshape(51,-1).mean(0)))
    check("GP POD rank24 orthonormal",m["basis"].shape==(216000,24) and np.max(np.abs(m["basis"].T@m["basis"]-np.eye(24)))<1e-11)
    p=gp.predict(m,c);s=gp.predict(m,c,return_signed=True)
    check("GP signed first then magnitude",np.array_equal(p,np.hypot(s[...,0::2],s[...,1::2])))
    fresh=gp.metric(p,truth,allowed);old=result["train51_fitting_magnitude"]
    check("GP all51 raw magnitude fitting metrics replay",all(abs(fresh[k]-old[k])<1e-12 for k in ("relative_L2_pct","rmse_T","mae_T","max_abs_T")))
    single_difference=float(np.max(np.abs(gp.predict(m,c[0],frame_index=8)-p[0,8])))
    check("GP single case/frame output",single_difference<1e-10,dict(max_abs_difference_T=single_difference,tolerance_T=1e-10,reason="BLAS matrix/single-row summation roundoff in kernel map"))
    check("GP NumPy-only novel input finite",np.isfinite(gp.predict(m,[240,55.5],frame_index=11)).all())
    for value in ([-1,50],[np.nan,50]):
        try:gp.predict(m,value)
        except ValueError:check("GP rejects malformed native condition "+str(value),True)
        else:check("GP rejects malformed native condition "+str(value),False)
    report=dict(status="PASS",passed=len(checks),failed=0,checks=checks,torch_and_sklearn_imports_blocked=True,numerical_labels_read=allowed,excluded_test_case_ids=excluded,test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,warning="All numerical field metrics in this audit use TRAIN51 only. No prediction/validation error is certified.",protocol_sha256=sha(HERE/"protocol.json"),source_sha256=sha(__file__))
    (HERE/"qa_final51.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(dict(status="PASS",passed=len(checks),failed=0,original_test_labels_loaded=False)),flush=True)

if __name__=="__main__":main()
