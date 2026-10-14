"""Independent point neural condition maps plus zero-mean neural cycle residual.

All scalers/targets/weights use only42 training conditions. Validation selects a
raw full-field checkpoint. Test labels and full60 fitting labels stay excluded.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import time
import zipfile
import numpy as np
from joint_spatial_nn import sha,exact_metrics

VERSION="v3-low-capacity-independent-point-mean-neural-with-trainable-linear-skip"

def selected_rows(path,key,indices):
    mapping={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as archive,archive.open(key+".npy") as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or len(shape)!=3 or shape[0]!=60 or dtype.kind!="f":raise ValueError("Invalid source field array")
        out=np.empty((len(indices),*shape[1:]),dtype=dtype);block=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(shape[0]):
            if i in mapping:
                raw=f.read(block)
                if len(raw)!=block:raise ValueError("Short selected field row")
                out[mapping[i]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(block,1)
    if not np.isfinite(out).all():raise ValueError("Nonfinite selected labels")
    return out

def fvalues(c,kind,scale):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));r=c[:,0]/scale;a=np.deg2rad(c[:,1])
    if c.shape[1]!=2 or not np.isfinite(c).all():raise ValueError("Invalid native conditions")
    if kind=="native":return np.column_stack((r,c[:,1]/90))
    if kind.startswith("sat"):
        s=float(kind[3:]);r=s/scale*np.tanh(c[:,0]/s)
    x=r*np.cos(a);y=r*np.sin(a)
    if kind=="cartesian":return np.column_stack((x,y))
    if kind=="mixed":return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))
    return np.column_stack((x,y,x*x,x*y,y*y))

def baseline_predict(m,c):
    # Exact NumPy forward of immutable V2 signed-cycle neural model.
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));scale=float(m["current_scale"])
    f=fvalues(c,str(m["feature_kind"]),scale);v=(f-m["feature_mean"])/m["feature_scale"]
    for j in range(int(m["layer_count"])):
        v=v@m[f"w{j}"]+m[f"b{j}"]
        if j<int(m["layer_count"])-1:
            v=np.tanh(v) if str(m["activation"])=="tanh" else v/(1+np.exp(-np.clip(v,-700,700)))
    x=fvalues(c,"cartesian",scale);deg=int(m["base_degree"])
    poly=np.column_stack([x[:,0]**i*x[:,1]**j for total in range(deg+1) for i in range(total+1) for j in [total-i]])
    latent=v*m["output_scale"]+poly@m["base_weights"]
    coef=(m["cycle_mean"]+latent@m["cycle_basis"]).reshape(len(c),int(m["frame_count"]),-1)
    return m["spatial_mean"]+coef@m["spatial_basis"].T

def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if sha(path)!=path.with_suffix(path.suffix+".sha256").read_text().strip():raise ValueError("Point model SHA mismatch")
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def predict_mean(m,c):
    x=(fvalues(c,str(m["feature_kind"]),float(m["current_scale"]))-m["feature_mean"])/m["feature_scale"]
    v=np.broadcast_to(x,(int(m["point_count"]),*x.shape))
    for j in range(int(m["layer_count"])):
        v=np.einsum("pcf,pfh->pch",v,m[f"w{j}"])+m[f"b{j}"][:,None,:]
        if j<int(m["layer_count"])-1:v=np.tanh(v)
    if bool(m["has_skip"]):v=v+np.einsum("cf,pfh->pch",x,m["skip_weight"])+m["skip_bias"][:,None,:]
    return (v*m["output_scale"][:,None,:]+m["output_mean"][:,None,:]).transpose(1,0,2)

def predict(model,conditions,frame_index=None,nonnegative=False,return_mean=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    mean=predict_mean(m,c)
    if return_mean:return mean[0] if single else mean
    baseline={k[len("baseline_"):]:v for k,v in m.items() if k.startswith("baseline_")}
    raw=baseline_predict(baseline,c)
    if str(m["target_kind"])=="magnitude":
        raw=np.hypot(raw[...,0::2],raw[...,1::2]);out=mean[...,0][:,None,:]+float(m["dynamic_scale"])*(raw-raw.mean(1,keepdims=True))
    else:
        raw=raw.reshape(len(c),int(m["frame_count"]),int(m["point_count"]),2)
        signed=mean[:,None,:,:]+float(m["dynamic_scale"])*(raw-raw.mean(1,keepdims=True));out=np.hypot(signed[...,0],signed[...,1])
    if nonnegative:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m["frame_count"]):raise ValueError("Invalid saved frame index")
        out=out[:,frame_index]
    return out[0] if single else out

def prepare(cache,split_path,baseline_path,baseline_module_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);split=json.loads(Path(split_path).read_text(encoding="utf-8-sig"));keep=split["train"]+split["validation"]
    if [len(split[k]) for k in ("train","validation","test")]!=[42,9,9] or len(set(keep+split["test"]))!=60:raise ValueError("Split guard")
    with np.load(cache,allow_pickle=False) as z:g={k:z[k] for k in ("conditions","point_id","xy","theta","time_s","case_ids","metadata_json")}
    if g["case_ids"].tolist()!=split["case_ids"]:raise ValueError("Case identity guard")
    signed=selected_rows(cache,"field",keep).astype(float);y=np.hypot(signed[...,0::2],signed[...,1::2]);means=y.mean(1)[...,None];signedmean=signed.mean(1).reshape(51,-1,2)
    with np.load(baseline_path,allow_pickle=False) as z:base={k:z[k] for k in z.files}
    if sha(baseline_path)!=Path(baseline_path).with_suffix(".npz.sha256").read_text().strip():raise ValueError("Baseline SHA guard")
    inputs=g["conditions"][keep];predbase=baseline_predict(base,inputs)
    spec=importlib.util.spec_from_file_location("v2_cycle_neural_readonly",baseline_module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected=module.predict(baseline_path,inputs)
    if np.max(np.abs(predbase-expected))>1e-10:raise AssertionError("Baseline copied NumPy equations disagree")
    dynamic=np.hypot(predbase[...,0::2],predbase[...,1::2]);dynamic-=dynamic.mean(1,keepdims=True)
    dynvec=predbase.reshape(51,60,-1,2).copy();dynvec-=dynvec.mean(1,keepdims=True)
    train_dynamic_truth=y[:42]-y[:42].mean(1,keepdims=True)
    train_signed_dynamic_truth=signed[:42].reshape(42,60,-1,2)-signed[:42].reshape(42,60,-1,2).mean(1,keepdims=True)
    alpha_mag=float(np.clip(np.sum(dynamic[:42]*train_dynamic_truth)/np.sum(dynamic[:42]**2),0,1))
    alpha_signed=float(np.clip(np.sum(dynvec[:42]*train_signed_dynamic_truth)/np.sum(dynvec[:42]**2),0,1))
    # Train-only pointwise variability, optional diagnostic hotspot selection.
    np.savez_compressed(out/"mean_data.npz",conditions=inputs,case_ids=g["case_ids"][keep],magnitude_mean=means,signed_mean=signedmean,truth_energy=np.sum(y*y,axis=(1,2)),scalar_count=np.asarray(y.shape[1]*y.shape[2]),
        validation_dynamic=dynamic[42:],validation_signed_dynamic=dynvec[42:],magnitude_dynamic_scale=np.asarray(alpha_mag),signed_dynamic_scale=np.asarray(alpha_signed))
    np.savez_compressed(out/"baseline_model.npz",**base)
    np.savez_compressed(out/"geometry.npz",**{k:g[k] for k in ("point_id","xy","theta","time_s")})
    identity=dict(version=VERSION,train_case_ids=g["case_ids"][split["train"]].tolist(),validation_case_ids=g["case_ids"][split["validation"]].tolist(),excluded_case_ids=g["case_ids"][split["test"]].tolist(),
        cache_sha256=sha(cache),split_sha256=sha(split_path),baseline_model_sha256=sha(baseline_path),baseline_source_sha256=sha(baseline_module_path),
        point_count=int(len(g["point_id"])),frame_count=60,quantity="sqrt(Br^2+Bt^2), no Bz",units="T",xy_units="mm",label_access=keep,
        test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,source_sha256=sha(__file__),baseline_numpy_replay_max_abs_difference=float(np.max(np.abs(predbase-expected))),
        dynamic_scale_policy="one nonnegative scalar per quantity least-squares calibrated on train42 only; no validation choice of dynamic mixing weight",
        magnitude_dynamic_scale=alpha_mag,signed_dynamic_scale=alpha_signed,
        train_magnitude_dynamic_relative_L2_pct=float(100*np.sqrt(np.sum(train_dynamic_truth**2)/np.sum(y[:42]**2))),
        train_original_neural_dynamic_error_relative_L2_pct=float(100*np.sqrt(np.sum((dynamic[:42]-train_dynamic_truth)**2)/np.sum(y[:42]**2))),
        train_calibrated_neural_dynamic_error_relative_L2_pct=float(100*np.sqrt(np.sum((alpha_mag*dynamic[:42]-train_dynamic_truth)**2)/np.sum(y[:42]**2))))
    (out/"identity.json").write_text(json.dumps(identity,indent=2)+"\n",encoding="utf-8")

def train(prepared,truth_path,out,config):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);torch.manual_seed(config["seed"])
    prepared=Path(prepared);out=Path(out);out.mkdir(parents=True,exist_ok=False);identity=json.loads((prepared/"identity.json").read_text());start=time.perf_counter()
    with np.load(prepared/"mean_data.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(truth_path,allow_pickle=False) as z:truth=z["truth"];truth_ids=z["case_ids"]
    if truth_ids.tolist()!=identity["validation_case_ids"] or a["case_ids"][:42].tolist()!=identity["train_case_ids"]:raise ValueError("Allowed case identity guard")
    with np.load(prepared/"baseline_model.npz",allow_pickle=False) as z:base={k:z[k] for k in z.files}
    with np.load(prepared/"geometry.npz",allow_pickle=False) as z:geometry={k:z[k] for k in z.files}
    with threadpool_limits(limits=2):
        target=a["magnitude_mean"] if config["target_kind"]=="magnitude" else a["signed_mean"]
        alpha=float(a["magnitude_dynamic_scale"] if config["target_kind"]=="magnitude" else a["signed_dynamic_scale"])
        cs=max(float(a["conditions"][:42,0].max()),1);f=fvalues(a["conditions"],config["features"],cs);fm=f[:42].mean(0);fs=np.maximum(f[:42].std(0),1e-8);x=(f-fm)/fs
        cm=target[:42].mean(0);sc=target[:42].std(0);sc=np.maximum(sc,sc.max()*.001);points=target.shape[1];outputs=target.shape[2]
        dims=[x.shape[-1],*config["hidden"],outputs]
        weights=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(points,ni,no,dtype=torch.float32)*np.sqrt(2/(ni+no))) for ni,no in zip(dims[:-1],dims[1:])])
        biases=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(points,1,no,dtype=torch.float32)) for no in dims[1:]])
        with torch.no_grad():weights[-1].zero_()
        skipw=torch.nn.Parameter(torch.zeros(points,dims[0],outputs,dtype=torch.float32));skipb=torch.nn.Parameter(torch.zeros(points,1,outputs,dtype=torch.float32))
        params=list(weights)+list(biases)+[skipw,skipb];xt=torch.tensor(np.broadcast_to(x,(points,*x.shape)).copy(),dtype=torch.float32)
        tt=torch.tensor(target[:42].transpose(1,0,2),dtype=torch.float32);mt=torch.tensor(cm[:,None,:],dtype=torch.float32);st=torch.tensor(sc[:,None,:],dtype=torch.float32)
        denominator=float(a["truth_energy"][:42].sum());history=[];best=float("inf");best_state=None;best_step=0;closures=0
        def forward(inp):
            v=inp
            for j,(w,b) in enumerate(zip(weights,biases)):
                v=torch.bmm(v,w)+b
                if j<len(weights)-1:v=torch.tanh(v)
            return (v+torch.bmm(inp,skipw)+skipb)*st+mt
        def objective():
            e=forward(xt[:,:42])-tt
            return 60*torch.sum(e*e)/denominator+config["regularization"]*sum(torch.mean(w*w) for w in weights)
        def val_field(means):
            if config["target_kind"]=="magnitude":return means[...,0][:,None,:]+alpha*a["validation_dynamic"]
            v=means[:,None,:,:]+alpha*a["validation_signed_dynamic"];return np.hypot(v[...,0],v[...,1])
        def checkpoint(step,stage,loss):
            nonlocal best,best_state,best_step
            with torch.no_grad():pred=forward(xt).numpy().transpose(1,0,2)
            v=val_field(pred[42:]);sse=np.sum((v-truth)**2);rmse=float(np.sqrt(sse/truth.size));rel=float(100*np.sqrt(sse/np.sum(truth*truth)));tr=float(100*np.sqrt(60*np.sum((pred[:42]-target[:42])**2)/denominator))
            row=dict(step=step,stage=stage,loss=float(loss),train_mean_relative_L2_pct=tr,validation_relative_L2_pct=rel,validation_rmse_T=rmse,elapsed_seconds=time.perf_counter()-start);history.append(row)
            if rmse<best:best=rmse;best_state=[p.detach().clone() for p in params];best_step=step
            return row
        def restore():
            with torch.no_grad():
                for p,b in zip(params,best_state):p.copy_(b)
        opt=torch.optim.Adam(params,lr=config["learning_rate"]);checkpoint(0,"initial",objective().detach())
        for step in range(1,config["adam_steps"]+1):
            opt.zero_grad();loss=objective();loss.backward();torch.nn.utils.clip_grad_norm_(params,10);opt.step()
            if step%50==0:
                row=checkpoint(step,"Adam",loss.detach())
                if step%1000==0:print(json.dumps(dict(name=config["name"],progress=row)),flush=True)
        final_adam=history[-1].copy();restore()
        # Use Float64 for real-T LBFGS refinement and exact NumPy replay.
        with torch.no_grad():
            for p in params:p.data=p.data.double()
        xt=torch.tensor(np.broadcast_to(x,(points,*x.shape)).copy(),dtype=torch.float64)
        tt=torch.tensor(target[:42].transpose(1,0,2),dtype=torch.float64)
        mt=torch.tensor(cm[:,None,:],dtype=torch.float64);st=torch.tensor(sc[:,None,:],dtype=torch.float64)
        opt=torch.optim.LBFGS(params,lr=1.,max_iter=10,history_size=10,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-14)
        for outer in range(1,config["lbfgs_blocks"]+1):
            def closure():
                nonlocal closures
                opt.zero_grad();loss=objective();loss.backward();closures+=1;return loss
            opt.step(closure);row=checkpoint(config["adam_steps"]+outer*10,"LBFGS",objective().detach())
            if outer%20==0:print(json.dumps(dict(name=config["name"],progress=row)),flush=True)
        final_optimizer=history[-1].copy();restore()
        arrays=dict(feature_kind=np.asarray(config["features"]),current_scale=np.asarray(cs),feature_mean=fm,feature_scale=fs,output_mean=cm,output_scale=sc,point_count=np.asarray(points),frame_count=np.asarray(60),layer_count=np.asarray(len(weights)),target_kind=np.asarray(config["target_kind"]),dynamic_scale=np.asarray(alpha),has_skip=np.asarray(True),skip_weight=skipw.detach().numpy(),skip_bias=skipb.detach().numpy()[:,0,:],**geometry)
        for j,(w,b) in enumerate(zip(weights,biases)):arrays[f"w{j}"]=w.detach().numpy();arrays[f"b{j}"]=b.detach().numpy()[:,0,:]
        arrays.update({"baseline_"+k:v for k,v in base.items()})
        metadata=dict(identity=identity,configuration=config,source_sha256=sha(__file__),checkpoint="minimum full raw validation field RMSE",best_step=best_step,
            mean_model="independent one-hidden-layer tanh MLP plus trainable linear skip per fixed spatial point, fewer than42 parameters per point, all42 condition targets",dynamic_model="immutable V2 signed pure NN, zero temporal mean applied per new input condition",
            train_scalers="train42 only",test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
        arrays["metadata_json"]=np.asarray(json.dumps(metadata,ensure_ascii=False));np.savez_compressed(out/"model.npz",**arrays);(out/"model.npz.sha256").write_text(sha(out/"model.npz")+"\n")
        actualmean=predict_mean(arrays,a["conditions"])
        with torch.no_grad():expectedmean=forward(xt).numpy().transpose(1,0,2)
        replay=float(np.max(np.abs(actualmean-expectedmean)))
        if replay>1e-10:raise AssertionError("Independent NumPy point mean replay mismatch")
        raw=predict(out/"model.npz",a["conditions"][42:]);positive=np.maximum(raw,0)
        alternate_zero=dict(arrays,dynamic_scale=np.asarray(0.));alternate_unit=dict(arrays,dynamic_scale=np.asarray(1.))
        if not np.isclose(np.sqrt(np.mean((raw-truth)**2)),best,rtol=2e-4,atol=1e-7):raise AssertionError("Full original field disagrees with checkpoint")
        with (out/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        result=dict(name=config["name"],configuration=config,validation=exact_metrics(raw,truth,a["case_ids"][42:]),validation_nonnegative=exact_metrics(positive,truth,a["case_ids"][42:]),
            best_step=best_step,raw_negative_count=int(np.sum(raw<0)),raw_min_T=float(raw.min()),parameter_count=sum(p.numel() for p in params),elapsed_seconds=time.perf_counter()-start,
            final_adam=final_adam,final_optimizer=final_optimizer,min_train_mean_relative_L2_pct=min(h["train_mean_relative_L2_pct"] for h in history),numpy_point_replay_max_abs_difference=replay,
            model_sha256=sha(out/"model.npz"),lbfgs_closure_evaluations=closures,dynamic_scale=alpha,
            validation_dynamic_zero_diagnostic=exact_metrics(predict(alternate_zero,a["conditions"][42:]),truth,a["case_ids"][42:]),
            validation_dynamic_unit_diagnostic=exact_metrics(predict(alternate_unit,a["conditions"][42:]),truth,a["case_ids"][42:]),
            test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
        (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,ensure_ascii=False),flush=True)
        return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="mode",required=True)
    q=sub.add_parser("prepare");q.add_argument("--cache",type=Path,required=True);q.add_argument("--split",type=Path,required=True);q.add_argument("--baseline",type=Path,required=True);q.add_argument("--baseline-module",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    q=sub.add_parser("train");q.add_argument("--prepared",type=Path,required=True);q.add_argument("--truth",type=Path,required=True);q.add_argument("--output",type=Path,required=True);q.add_argument("--config",type=Path,required=True)
    a=p.parse_args()
    if a.mode=="prepare":prepare(a.cache,a.split,a.baseline,a.baseline_module,a.output)
    else:train(a.prepared,a.truth,a.output,json.loads(a.config.read_text(encoding="utf-8")))
