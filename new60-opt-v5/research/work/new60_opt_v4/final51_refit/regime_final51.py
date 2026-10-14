"""Final rotor neural refit on original train42 plus original validation9.

Original test9 never enter labels, scalers, initialization, or checkpoints.
The signed temporal mean has 2/3 learned sigmoid affine experts per point;
the immutable train42 neural cycle supplies its zero-mean signed dynamics.
Inference uses NumPy only and is self-contained in this source + model.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import zipfile
import numpy as np

VERSION = "v4-final51-point-signed-sigmoid-mean-with-frozen-train42-dynamics"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def features(c,kind,scale):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64))
    if c.shape[1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError("Expected finite native [I_RMS>=0, Thet_deg]")
    r=c[:,0]/scale;a=np.deg2rad(c[:,1]);x=r*np.cos(a);y=r*np.sin(a)
    if kind=="cartesian":return np.column_stack((x,y))
    if kind=="mixed":return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))
    if kind=="cartesian2":return np.column_stack((x,y,x*x,x*y,y*y))
    raise ValueError("Unknown feature specification")

def sigmoid(v):
    z=np.exp(-np.abs(v));return np.where(v>=0,1/(1+z),z/(1+z))

def baseline_predict(m,c):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));scale=float(m["current_scale"])
    v=(features(c,str(m["feature_kind"]),scale)-m["feature_mean"])/m["feature_scale"]
    for j in range(int(m["layer_count"])):
        v=v@m[f"w{j}"]+m[f"b{j}"]
        if j<int(m["layer_count"])-1:v=np.tanh(v) if str(m["activation"])=="tanh" else v*sigmoid(v)
    x=features(c,"cartesian",scale);degree=int(m["base_degree"])
    poly=np.column_stack([x[:,0]**i*x[:,1]**j for total in range(degree+1) for i in range(total+1) for j in [total-i]])
    latent=v*m["output_scale"]+poly@m["base_weights"]
    coef=(m["cycle_mean"]+latent@m["cycle_basis"]).reshape(len(c),int(m["frame_count"]),-1)
    return m["spatial_mean"]+coef@m["spatial_basis"].T

def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if sha(path)!=path.with_suffix(path.suffix+".sha256").read_text().strip():raise ValueError("Final51 model SHA mismatch")
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def predict_mean(m,c):
    x=(features(c,"cartesian",float(m["current_scale"]))-m["feature_mean"])/m["feature_scale"]
    v=np.einsum("cf,pfh->pch",x,m["linear_weight"])+m["linear_bias"][:,None,:]
    z=np.einsum("cf,pfk->pck",x,m["gate_weight"])+m["gate_bias"][:,None,:]
    aug=np.column_stack((np.ones(len(x)),x));expert=np.einsum("cf,pkfh->pckh",aug,m["expert_weight"])
    v=v+np.sum(sigmoid(z)[...,None]*expert,axis=2)
    return (v*m["output_scale"][:,None,:]+m["output_mean"][:,None,:]).transpose(1,0,2)

def predict(model,conditions,frame_index=None,nonnegative=False,return_signed=False,return_mean=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c);mean=predict_mean(m,c)
    if return_mean:return mean[0] if single else mean
    base={k[9:]:v for k,v in m.items() if k.startswith("baseline_")};old=baseline_predict(base,c).reshape(len(c),int(m["frame_count"]),int(m["point_count"]),2)
    signed=mean[:,None,:,:]+float(m["dynamic_scale"])*(old-old.mean(1,keepdims=True))
    out=signed.reshape(len(c),int(m["frame_count"]),-1) if return_signed else np.hypot(signed[...,0],signed[...,1])
    if nonnegative and not return_signed:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m["frame_count"]):raise ValueError("Invalid saved frame index")
        out=out[:,frame_index]
    return out[0] if single else out

def read_selected_fields(path,indices):
    """Stream the NPZ field member and materialize only the explicit train51 rows."""
    indices=list(indices);locations={v:i for i,v in enumerate(indices)}
    if len(indices)!=51 or len(set(indices))!=51:raise ValueError("Expected exactly 51 unique selected rows")
    with zipfile.ZipFile(path) as z,z.open("field.npy") as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or shape!=(60,60,3600) or dtype.kind!="f":raise ValueError("Unexpected rotor field storage")
        size=int(np.prod(shape[1:]))*dtype.itemsize;out=np.empty((51,*shape[1:]),dtype=np.float64)
        for case in range(shape[0]):
            if case in locations:
                raw=f.read(size)
                if len(raw)!=size:raise ValueError("Truncated selected field row")
                out[locations[case]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(size,1)
    if not np.isfinite(out).all():raise ValueError("Nonfinite train51 fields")
    return out

def metrics(p,y,ids,point_ids):
    e=p-y;se=np.sum(e*e,axis=(1,2));energy=np.sum(y*y,axis=(1,2));rm=np.sqrt(se/np.prod(y.shape[1:]));worst=int(np.argmax(rm));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape)
    return dict(relative_L2_pct=float(100*np.sqrt(se.sum()/energy.sum())),rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        worst_case_id=int(ids[worst]),worst_case_rmse_T=float(rm[worst]),max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),point_id=int(point_ids[loc[2]]),prediction_T=float(p[loc]),truth_T=float(y[loc]),signed_error_T=float(e[loc])),
        per_case=[dict(case_id=int(k),rmse_T=float(r),relative_L2_pct=float(100*np.sqrt(s/den))) for k,r,s,den in zip(ids,rm,se,energy)])

def train(prepared,cache,out,config,protocol):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);torch.manual_seed(config["seed"]);rng=np.random.default_rng(config["seed"])
    prepared=Path(prepared);cache=Path(cache);out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    identity=json.loads((prepared/"identity.json").read_text(encoding="utf-8"));allowed=identity["train_case_ids"]+identity["validation_case_ids"];excluded=identity["excluded_case_ids"]
    if identity["test_labels_loaded"] or len(allowed)!=51 or set(allowed)&set(excluded) or set(allowed)|set(excluded)!=set(range(60)):raise ValueError("Allowed train51 label identity guard")
    if sha(prepared/"mean_data.npz")!=protocol["prepared_means_sha256"] or sha(cache)!=identity["cache_sha256"]:raise ValueError("Training data hash guard")
    with np.load(prepared/"mean_data.npz",allow_pickle=False) as z:a={k:z[k] for k in ("conditions","case_ids","signed_mean","truth_energy","signed_dynamic_scale")}
    with np.load(prepared/"geometry.npz",allow_pickle=False) as z:geometry={k:z[k] for k in z.files}
    if sha(prepared/"baseline_model.npz")!=identity["baseline_model_sha256"]:raise ValueError("Frozen train42 neural dynamics hash guard")
    with np.load(prepared/"baseline_model.npz",allow_pickle=False) as z:base={k:z[k] for k in z.files}
    if a["case_ids"].tolist()!=allowed or len(a["conditions"])!=51 or not np.isfinite(a["signed_mean"]).all():raise ValueError("Only51 prepared training rows allowed")
    if config["expert_kind"]!="sigmoid_affine" or config["gates"] not in (2,3):raise ValueError("Fixed final51 architecture guard")
    with threadpool_limits(limits=2):
        n=51;c=a["conditions"];target=a["signed_mean"];cs=float(c[:,0].max());f=features(c,"cartesian",cs);fm=f.mean(0);fs=np.maximum(f.std(0),1e-8);x=(f-fm)/fs
        cm=target.mean(0);sc=target.std(0);sc=np.maximum(sc,sc.max()*.001);points=target.shape[1];K=config["gates"]
        yn=(target-cm)/sc;design=np.column_stack((np.ones(n),x));ridge=np.diag([0,1e-8,1e-8]);coef=np.linalg.solve(design.T@design+ridge,design.T@yn.reshape(n,-1)).reshape(3,points,2)
        linearw=coef[1:].transpose(1,0,2);linearb=coef[0]
        bzero=(linearb+np.einsum("f,pfh->ph",(np.zeros(2)-fm)/fs,linearw))*sc+cm
        direction=np.sum((linearw*sc[:,None,:])*bzero[:,None,:],axis=-1);norm=np.linalg.norm(direction,axis=-1,keepdims=True)
        direction=np.where(norm>1e-12,direction/np.maximum(norm,1e-12),np.array([1.,0.]));directions=np.broadcast_to(direction[:,None,:],(points,K,2)).copy()
        projection=np.einsum("cf,pkf->pck",x,directions);q=np.linspace(.25,.75,K);threshold=np.stack([np.quantile(projection[:,:,j],q[j],axis=1) for j in range(K)],axis=1)
        pscale=np.maximum(projection.std(1),1e-3);gwi=(directions*config["sharpness"]/pscale[...,None]).transpose(0,2,1);gbi=-threshold*config["sharpness"]/pscale
        lw=torch.nn.Parameter(torch.tensor(linearw,dtype=torch.float32));lb=torch.nn.Parameter(torch.tensor(linearb,dtype=torch.float32));gw=torch.nn.Parameter(torch.tensor(gwi,dtype=torch.float32));gb=torch.nn.Parameter(torch.tensor(gbi,dtype=torch.float32))
        ew=torch.nn.Parameter(torch.tensor(rng.normal(0,.001,(points,K,3,2)),dtype=torch.float32));params=[lw,lb,gw,gb,ew]
        xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(yn.transpose(1,0,2),dtype=torch.float32);st=torch.tensor(sc[:,None,:],dtype=torch.float32)
        gate0=torch.tensor(gwi,dtype=torch.float32);bias0=torch.tensor(gbi,dtype=torch.float32);den=float(a["truth_energy"].sum());alpha=float(a["signed_dynamic_scale"])
        def forward(inp):
            v=torch.einsum("cf,pfh->pch",inp,lw)+lb[:,None,:];z=torch.einsum("cf,pfk->pck",inp,gw)+gb[:,None,:]
            aug=torch.cat((torch.ones((len(inp),1),dtype=inp.dtype),inp),axis=1);expert=torch.einsum("cf,pkfh->pckh",aug,ew)
            return v+torch.sum(torch.sigmoid(z)[...,None]*expert,axis=2)
        def objective():
            e=(forward(xt)-yt)*st
            return 60*torch.sum(e*e)/den+config["gate_prior"]*(torch.mean((gw-gate0)**2)+torch.mean((gb-bias0)**2))+config["expert_regularization"]*torch.mean(ew*ew)
        history=[];best=float("inf");best_state=None;best_step=0;closures=0
        def checkpoint(step,stage,loss):
            nonlocal best,best_state,best_step
            with torch.no_grad():mean=forward(xt).numpy().transpose(1,0,2)*sc+cm
            sse=float(np.sum((mean-target)**2));tr=float(100*np.sqrt(60*sse/den));record=dict(step=step,stage=stage,objective=float(loss),train51_signed_mean_relative_L2_pct=tr,train51_signed_mean_rmse_T=float(np.sqrt(sse/target.size)),elapsed_seconds=time.perf_counter()-start);history.append(record)
            # These 51 rows are all training rows. There is no held-out validation.
            if sse<best:best=sse;best_step=step;best_state=[p.detach().clone() for p in params]
            return record
        def restore():
            with torch.no_grad():
                for p,b in zip(params,best_state):p.copy_(b)
        opt=torch.optim.Adam([dict(params=[lw,lb,ew],lr=config["learning_rate"]),dict(params=[gw,gb],lr=config["gate_learning_rate"])]);checkpoint(0,"train51_OLS_initial",objective().detach())
        for step in range(1,config["adam_steps"]+1):
            opt.zero_grad();loss=objective();loss.backward();torch.nn.utils.clip_grad_norm_(params,5);opt.step()
            if step%50==0:
                row=checkpoint(step,"Adam",loss.detach())
                if step%1000==0:print(json.dumps(dict(name=config["name"],progress=row)),flush=True)
        final_adam=history[-1].copy();restore()
        with torch.no_grad():
            for p in params:p.data=p.data.double().contiguous()
        xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(yn.transpose(1,0,2),dtype=torch.float64);st=torch.tensor(sc[:,None,:],dtype=torch.float64);gate0=torch.tensor(gwi,dtype=torch.float64);bias0=torch.tensor(gbi,dtype=torch.float64)
        opt=torch.optim.LBFGS(params,lr=1,max_iter=10,history_size=10,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-14)
        for outer in range(1,config["lbfgs_blocks"]+1):
            def closure():
                nonlocal closures
                opt.zero_grad();loss=objective();loss.backward()
                for p in params:
                    if p.grad is not None and not p.grad.is_contiguous():p.grad=p.grad.contiguous()
                closures+=1;return loss
            opt.step(closure);row=checkpoint(config["adam_steps"]+outer*10,"LBFGS",objective().detach())
            if outer%10==0:print(json.dumps(dict(name=config["name"],progress=row)),flush=True)
        final_optimizer=history[-1].copy();restore()
        meta=dict(version=VERSION,configuration=config,train_case_ids=allowed,validation_case_ids=[],former_validation_now_training_case_ids=identity["validation_case_ids"],excluded_case_ids=excluded,
            prepared_identity_sha256=sha(prepared/"identity.json"),prepared_means_sha256=sha(prepared/"mean_data.npz"),baseline_model_sha256=identity["baseline_model_sha256"],source_sha256=sha(__file__),protocol_sha256=protocol["protocol_sha256"],
            checkpoint="minimum TRAIN51 signed temporal-mean raw-T squared loss; no independent validation; no test labels",new_network="trained per-point signed affine skip plus2/3 learned sigmoid affine experts",initialization="train51 signed mean linear ridge OLS; gate thresholds and all scalers fitted only to same51 training rows",
            point_weighting="uniform all1800",dynamic="immutable V2 train42 pure NN zero-temporal-mean signed output, train42-calibrated scale1.0",units="T",xy_units="mm",quantity="planar sqrt(Br²+Bt²), no Bz",frame_scope="all60 provided saved first-cycle frames",best_step=best_step,test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,
            label_scope="final51 refit: original validation9 become training; original9 test remain excluded; train51 metrics are fitting metrics")
        arrays=dict(expert_kind=np.asarray(config["expert_kind"]),initial_sharpness=np.asarray(config["sharpness"]),current_scale=np.asarray(cs),feature_mean=fm,feature_scale=fs,output_mean=cm,output_scale=sc,
            point_count=np.asarray(points),frame_count=np.asarray(60),dynamic_scale=np.asarray(alpha),linear_weight=lw.detach().numpy(),linear_bias=lb.detach().numpy(),gate_weight=gw.detach().numpy(),gate_bias=gb.detach().numpy(),expert_weight=ew.detach().numpy(),metadata_json=np.asarray(json.dumps(meta,ensure_ascii=False)),**geometry)
        arrays.update({"baseline_"+k:v for k,v in base.items()});np.savez_compressed(out/"model.npz",**arrays);(out/"model.npz.sha256").write_text(sha(out/"model.npz")+"\n")
        actual=predict_mean(arrays,c)
        with torch.no_grad():expected=forward(xt).numpy().transpose(1,0,2)*sc+cm
        replay=float(np.max(np.abs(actual-expected)))
        if replay>1e-10:raise AssertionError("NumPy and float64Torch mean replay mismatch")
        # Only after checkpoint and saved weights are fixed, score TRAIN51 full fields.
        truth_signed=read_selected_fields(cache,allowed);truth=np.hypot(truth_signed[...,0::2],truth_signed[...,1::2]);raw=predict(out/"model.npz",c)
        met=metrics(raw,truth,a["case_ids"],geometry["point_id"])
        with (out/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        result=dict(name=config["name"],configuration=config,train51_fitting_magnitude=met,train51_signed_mean_relative_L2_pct=float(100*np.sqrt(60*np.sum((actual-target)**2)/den)),
            new_trainable_parameters=sum(p.numel() for p in params),new_parameters_per_point=sum(p.numel() for p in params)//points,best_step=best_step,final_adam=final_adam,final_optimizer=final_optimizer,elapsed_seconds=time.perf_counter()-start,lbfgs_closure_evaluations=closures,
            numpy_torch_float64_replay_max_abs_difference_T=replay,model_sha256=sha(out/"model.npz"),source_sha256=sha(__file__),training_original_case_ids=allowed,excluded_original_case_ids=excluded,validation_case_ids=[],test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,
            warning="Original validation9 are included in training. These train51 full-field results are fitting errors, not prediction/validation errors.")
        (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,ensure_ascii=False),flush=True);return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepared",type=Path,required=True);p.add_argument("--cache",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--config",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text(encoding="utf-8"));protocol["protocol_sha256"]=sha(a.protocol);train(a.prepared,a.cache,a.output,json.loads(a.config.read_text(encoding="utf-8")),protocol)
