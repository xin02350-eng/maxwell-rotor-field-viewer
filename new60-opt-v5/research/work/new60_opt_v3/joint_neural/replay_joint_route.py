"""Independent saved-file validation replay; inference needs only NumPy."""
from pathlib import Path
import builtins
import importlib.util
import json
import sys
import numpy as np

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from joint_spatial_nn import sha

checks=[]
def check(label,condition,detail=None):
    checks.append(dict(label=label,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError(label)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

with np.load(root/"point_prepared/mean_data.npz",allow_pickle=False) as z:c=z["conditions"][42:];ids=z["case_ids"][42:];mean_target=z["signed_mean"];trainc=z["conditions"][:42]
with np.load(root.parents[1]/"new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared/validation_truth.npz",allow_pickle=False) as z:truth=z["truth"];truth_ids=z["case_ids"]
check("Validation truth9x60x1800",truth.shape==(9,60,1800))
check("Validation case IDs",np.array_equal(ids,truth_ids))
check("Validation-only excluded case IDs",set(ids.tolist()).isdisjoint({2,4,12,16,17,18,38,41,58}))
guarded_import=builtins.__import__
def import_guard(name,*args,**kwargs):
    if name.split(".")[0] in {"torch","sklearn"}:raise AssertionError("Training-only dependency attempted during NumPy inference")
    return guarded_import(name,*args,**kwargs)
builtins.__import__=import_guard
joint=load_module("joint_numpy_check",root/"joint_spatial_nn.py")
point=load_module("point_numpy_check",root/"point_mean_nn.py")
low=load_module("point_low_numpy_check",root/"point_mean_lowcap_nn.py")
portable=load_module("portable_numpy_check",root/"pure_neural_blend_portable_v3.py")
rows=[]
for directory in (root/"runs",root/"point_runs"):
    for resultpath in sorted(directory.glob("*/result.json")):
        result=json.loads(resultpath.read_text(encoding="utf-8"));modelpath=resultpath.with_name("model.npz");arrays=np.load(modelpath,allow_pickle=False)
        metadata=json.loads(str(arrays["metadata_json"]));config=result["configuration"]
        source="joint_spatial_nn.py" if directory.name=="runs" else "point_mean_lowcap_nn.py" if result["name"].startswith("point_low") else "point_mean_nn.py"
        module=joint if source.startswith("joint") else low if "lowcap" in source else point
        check(result["name"]+" model SHA",sha(modelpath)==result["model_sha256"]==modelpath.with_suffix(".npz.sha256").read_text().strip())
        check(result["name"]+" source SHA",metadata["source_sha256"]==sha(root/source))
        check(result["name"]+" test/all60-fit label exclusion",not any(result[k] for k in ("test_labels_loaded","test_metrics_computed","full60_fit_labels_loaded")))
        check(result["name"]+" train42 IDs",len(metadata["identity"]["train_case_ids"])==42)
        check(result["name"]+" validation9 IDs",metadata["identity"]["validation_case_ids"]==ids.tolist())
        loaded=[];original_load=np.load
        def load_guard(path,*args,**kwargs):
            p=Path(path).resolve();loaded.append(str(p));check(result["name"]+" inference weight-only load",p==modelpath.resolve());return original_load(path,*args,**kwargs)
        np.load=load_guard
        try:pred=module.predict(modelpath,c,nonnegative=False)
        finally:np.load=original_load
        check(result["name"]+" prediction shape",pred.shape==truth.shape)
        e=pred-truth
        measures=dict(relative_L2_pct=float(100*np.sqrt(np.sum(e*e)/np.sum(truth*truth))),rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))))
        for k,v in measures.items():check(result["name"]+" replay "+k,np.isclose(v,result["validation"][k],rtol=1e-11,atol=1e-12),dict(actual=v,recorded=result["validation"][k]))
        pos=np.maximum(pred,0);ep=pos-truth
        for k,v in dict(relative_L2_pct=float(100*np.sqrt(np.sum(ep*ep)/np.sum(truth*truth))),rmse_T=float(np.sqrt(np.mean(ep*ep))),mae_T=float(np.mean(np.abs(ep))),max_abs_T=float(np.max(np.abs(ep)))).items():
            check(result["name"]+" nonnegative "+k,np.isclose(v,result["validation_nonnegative"][k],rtol=1e-11,atol=1e-12))
        for i,(cid,reported) in enumerate(zip(ids,result["validation"]["per_case"])):
            check(result["name"]+f" case{cid} identity",int(cid)==reported["case_id"])
            check(result["name"]+f" case{cid} RMSE",np.isclose(np.sqrt(np.mean(e[i]*e[i])),reported["rmse_T"],rtol=1e-11,atol=1e-12))
        for frame in (0,17,59):check(result["name"]+f" frame{frame} single API",np.allclose(module.predict(modelpath,c[0],frame_index=frame),pred[0,frame],rtol=0,atol=1e-12))
        arrays.close();rows.append(dict(name=result["name"],model=modelpath.relative_to(root).as_posix(),source=source,metrics=measures))
check("Completed true NN runs=11",len(rows)==11)
check("Incomplete explicitly excluded",(root/"point_runs/point_sat150_8_seed42/aborted.json").exists() and not (root/"point_runs/point_sat150_8_seed42/result.json").exists())
blendpath=root/"pure_neural_blend/portable_model.npz";binding=json.loads((root/"pure_neural_blend/portable_inference_binding.json").read_text())
check("Portable model SHA",sha(blendpath)==binding["portable_model_sha256"])
check("Portable wrapper source SHA",sha(root/"pure_neural_blend_portable_v3.py")==binding["inference_source_sha256"])
check("Portable point source SHA",sha(root/"point_mean_lowcap_nn.py")==binding["point_source_sha256"])
check("Portable helper source SHA",sha(root/"joint_spatial_nn.py")==binding["joint_source_sha256"])
with np.load(root/"pure_neural_blend/validation_predictions.npz",allow_pickle=False) as z:expected=z["raw_prediction"]
p=portable.predict(blendpath,c)
check("Portable relocation adapter identical field",np.array_equal(p,expected))
check("Portable source loaded without torch/sklearn",True)
for frame in (0,17,59):check(f"Portable frame{frame} float64 tolerance1e-12T",np.allclose(portable.predict(blendpath,c[0],frame_index=frame),p[0,frame],rtol=0,atol=1e-12))
# Explicit pointwise loops independently check the batched einsum expert decoder.
with np.load(blendpath,allow_pickle=False) as z:m={k:z[k] for k in z.files}
condition=c[0];r=condition[0]/float(m["current_scale"]);a=np.deg2rad(condition[1]);x=r*np.cos(a);y=r*np.sin(a);f=np.array([x,y,x*x,x*y,y*y]);norm=(f-m["feature_mean"])/m["feature_scale"]
actualmean=low.predict_mean(m,condition[None,:])
for j in (0,1359,1751):
    v=norm
    for k in range(int(m["layer_count"])):
        v=v@m[f"w{k}"][j]+m[f"b{k}"][j]
        if k<int(m["layer_count"])-1:v=np.tanh(v)
    v=v+norm@m["skip_weight"][j]+m["skip_bias"][j];v=v*m["output_scale"][j]+m["output_mean"][j]
    check(f"Independent point{j} loop",np.allclose(v,actualmean[0,j],rtol=0,atol=1e-12))
trainf=np.column_stack((trainc[:,0]/420*np.cos(np.deg2rad(trainc[:,1])),trainc[:,0]/420*np.sin(np.deg2rad(trainc[:,1]))));trainf=np.column_stack((trainf,trainf[:,0]**2,trainf[:,0]*trainf[:,1],trainf[:,1]**2))
check("Point feature mean train42 only",np.array_equal(trainf.mean(0),m["feature_mean"]))
check("Point feature std train42 only",np.array_equal(np.maximum(trainf.std(0),1e-8),m["feature_scale"]))
check("Point output mean train42 only",np.array_equal(mean_target[:42].mean(0),m["output_mean"]))
sc=mean_target[:42].std(0);sc=np.maximum(sc,sc.max()*.001);check("Point output std train42 only",np.array_equal(sc,m["output_scale"]))
check("Low-capacity30 params per spatial point",int(json.loads(str(m["metadata_json"]))["configuration"]["hidden"][0])==2 and 54000//1800==30)
baseline={k[9:]:v for k,v in m.items() if k.startswith("baseline_")}
with np.load(root.parents[1]/"new60_opt_v2/neural_route/rotor/selected_model.npz",allow_pickle=False) as z:
    for k in z.files:check("Immutable baseline exact array "+k,np.array_equal(z[k],baseline[k]))
total=sum(p.stat().st_size for p in root.rglob("*") if p.is_file());check("Route file budget<=300MB",total<=300*1024*1024,total)
report=dict(version="v3-neural-independent-saved-file-validation-replay",status="passed",checks_passed=len(checks),checks=checks,completed_NN_training_runs=11,incomplete_NN_runs_excluded=1,pure_neural_global_blend_comparisons=10,
    rows=rows,selected_blend=binding["configuration"],
    original_point_weight_hash=binding["original_model_sha256"],portable_model_sha256=binding["portable_model_sha256"],portable_inference_binding_sha256=sha(root/"pure_neural_blend/portable_inference_binding.json"),
    source_sha256=sha(__file__),route_total_bytes=total,test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
(root/"numpy_validation_replay.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(dict(status=report["status"],checks_passed=report["checks_passed"],bytes=total,NN_runs=11)))
