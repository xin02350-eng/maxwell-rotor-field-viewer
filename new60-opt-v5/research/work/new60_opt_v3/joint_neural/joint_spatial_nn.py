"""Train-only shared condition and saved-frame neural spatial-POD surrogate.

The spatial representation is reused read-only from the immutable V2 train-only
preparation. No excluded test labels or all-60 fitting labels are accessed.
Raw field SSE drives all checkpoints. NumPy-only deployable forward equations.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np

VERSION = "v3-shared-condition-saved-frame-spatial-pod-neural"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def condition_features(c,kind,current_scale):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64))
    if c.shape[1]!=2 or not np.isfinite(c).all():raise ValueError("Expected native [I_RMS, Thet_deg] condition rows")
    r=c[:,0]/current_scale;b=c[:,1]/90;a=np.deg2rad(c[:,1])
    if kind=="native":return np.column_stack((r,b,r*b,r*r,b*b))
    if kind.startswith("sat"):
        s=float(kind[3:]);r=s/current_scale*np.tanh(c[:,0]/s)
    x,y=r*np.cos(a),r*np.sin(a)
    if kind=="mixed":return np.column_stack((r,b,x,y,x*x,x*y,y*y,r*b))
    return np.column_stack((x,y,x*x,x*y,y*y))

def features(c,indices,frame_count,kind,current_scale,harmonics,interaction,transients):
    f=condition_features(c,kind,current_scale)
    # The supplied first cycle has no duplicate endpoint. Raw t and t^2 prevent
    # a forced periodic endpoint equality; sin/cos are mathematical features.
    t=(np.asarray(indices,dtype=np.float64)+1)/frame_count
    cols=[t,t*t]
    for k in range(1,harmonics+1):cols.extend((np.sin(2*np.pi*k*t),np.cos(2*np.pi*k*t)))
    if transients:cols.extend((np.exp(-2*t),np.exp(-8*t)))
    tf=np.column_stack(cols)
    cv=np.broadcast_to(f[:,None,:],(len(f),len(t),f.shape[-1]))
    tv=np.broadcast_to(tf[None,:,:],(len(f),len(t),tf.shape[-1]))
    pieces=[cv,tv]
    if interaction:
        nc=min(interaction,f.shape[-1]);pieces.append((cv[...,:nc,None]*tv[...,None,2:]).reshape(len(f),len(t),-1))
    return np.concatenate(pieces,axis=-1)

def load_model(path):
    if isinstance(path,dict):return path
    p=Path(path)
    if sha(p)!=p.with_suffix(p.suffix+".sha256").read_text().strip():raise ValueError("Model SHA mismatch")
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def predict(model,conditions,frame_index=None,nonnegative=False,return_coeff=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    n=int(m["frame_count"]);idx=np.arange(n) if frame_index is None else np.asarray([frame_index])
    if not np.issubdtype(idx.dtype,np.integer) or idx.min()<0 or idx.max()>=n:raise ValueError("Invalid saved frame index")
    v=(features(c,idx,n,str(m["feature_kind"]),float(m["current_scale"]),int(m["harmonics"]),int(m["interaction"]),bool(m["transients"]))-m["feature_mean"])/m["feature_scale"]
    skip=v@m["skip_weight"]+m["skip_bias"] if bool(m["has_skip"]) else 0.
    for i in range(int(m["layer_count"])):
        v=v@m[f"w{i}"]+m[f"b{i}"]
        if i<int(m["layer_count"])-1:v=np.tanh(v)
    coeff=(v+skip)*m["output_scale"]+m["coefficient_mean"][idx]
    out=coeff if return_coeff else m["spatial_mean"]+coeff@m["spatial_basis"].T
    if nonnegative and not return_coeff:out=np.maximum(out,0)
    if frame_index is not None:out=out[:,0]
    return out[0] if single else out

def exact_metrics(p,y,case_ids):
    e=p-y;se=np.sum(e*e,axis=(1,2));energy=np.sum(y*y,axis=(1,2));per=np.sqrt(se/np.prod(y.shape[1:]))
    worst=int(np.argmax(per));flat=int(np.argmax(np.abs(e)));index=np.unravel_index(flat,e.shape)
    return dict(relative_L2_pct=float(100*np.sqrt(se.sum()/energy.sum())),rmse_T=float(np.sqrt(np.mean(e*e))),
        mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        worst_case_id=int(case_ids[worst]),worst_case_rmse_T=float(per[worst]),
        max_abs_location=dict(case_id=int(case_ids[index[0]]),frame_index=int(index[1]),point_index=int(index[2]),
            prediction_T=float(p[index]),truth_T=float(y[index]),signed_error_T=float(e[index])),
        per_case=[dict(case_id=int(k),rmse_T=float(r),relative_L2_pct=float(100*np.sqrt(s/den))) for k,r,s,den in zip(case_ids,per,se,energy)])

def train(prepared,out,config):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);torch.manual_seed(config["seed"])
    prepared=Path(prepared).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    identity=json.loads((prepared/"identity.json").read_text(encoding="utf-8"))
    if identity["test_labels_loaded"] or len(identity["train_case_ids"])!=42 or len(identity["validation_case_ids"])!=9:raise ValueError("Prepared partition guard failed")
    if sha(prepared/"representation.npz")!=identity["representation_sha256"]:raise ValueError("Prepared representation SHA mismatch")
    with np.load(prepared/"train_validation_data.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(prepared/"representation.npz",allow_pickle=False) as z:rep={k:z[k] for k in z.files}
    if a["case_ids"][:42].tolist()!=identity["train_case_ids"] or a["case_ids"][42:].tolist()!=identity["validation_case_ids"]:raise ValueError("Prepared partition order failed")
    with threadpool_limits(limits=2):
        rank=config["rank"];target=a["coeff"][:,:,:rank];frames=target.shape[1];cs=max(float(a["conditions"][:42,0].max()),1.)
        f=features(a["conditions"],np.arange(frames),frames,config["features"],cs,config["harmonics"],config["interaction"],config["transients"])
        fm=f[:42].mean(axis=(0,1));fs=np.maximum(f[:42].std(axis=(0,1)),1e-8);x=(f-fm)/fs
        cm=target[:42].mean(axis=0);sc=np.std(target[:42]-cm,axis=(0,1));sc=np.maximum(sc,sc.max()*.001)
        oracle=np.maximum(a["centered_energy"]-np.sum(target*target,axis=(1,2)),0)
        xt=torch.tensor(x[:42].reshape(-1,x.shape[-1]),dtype=torch.float32);xall=torch.tensor(x.reshape(-1,x.shape[-1]),dtype=torch.float32)
        tt=torch.tensor(target[:42].reshape(-1,rank),dtype=torch.float32);mt=torch.tensor(np.broadcast_to(cm,(42,*cm.shape)).reshape(-1,rank).copy(),dtype=torch.float32);st=torch.tensor(sc,dtype=torch.float32)
        dims=[x.shape[-1],*config["hidden"],rank];layers=[]
        for j,(ni,no) in enumerate(zip(dims[:-1],dims[1:])):
            layer=torch.nn.Linear(ni,no);torch.nn.init.xavier_normal_(layer.weight);torch.nn.init.zeros_(layer.bias)
            if j==len(dims)-2:torch.nn.init.zeros_(layer.weight)
            layers.append(layer)
            if j<len(dims)-2:layers.append(torch.nn.Tanh())
        model=torch.nn.Sequential(*layers);skip=torch.nn.Linear(dims[0],rank) if config["skip"] else None
        if skip is not None:torch.nn.init.zeros_(skip.weight);torch.nn.init.zeros_(skip.bias)
        params=list(model.parameters())+(list(skip.parameters()) if skip is not None else [])
        denominator=float(a["truth_energy"][:42].sum());history=[];best=float("inf");best_state=None;best_step=0;start=time.perf_counter();closures=0
        def normalized(inp):return model(inp)+(skip(inp) if skip is not None else 0.)
        def objective():
            e=normalized(xt)*st+mt-tt
            return torch.sum(e*e)/denominator+config["regularization"]*sum(torch.mean(p*p) for p in params if p.ndim==2)
        def checkpoint(step,stage,loss):
            nonlocal best,best_state,best_step
            with torch.no_grad():p=normalized(xall).numpy().reshape(target.shape)*sc+cm
            sse=np.sum((p-target)**2,axis=(1,2))+oracle
            tr=float(100*np.sqrt(sse[:42].sum()/a["truth_energy"][:42].sum()));va=float(100*np.sqrt(sse[42:].sum()/a["truth_energy"][42:].sum()))
            record=dict(step=step,stage=stage,loss=float(loss),train_relative_L2_pct=tr,validation_relative_L2_pct=va,validation_rmse_T=float(np.sqrt(sse[42:].sum()/(9*int(a["scalar_count"])))),elapsed_seconds=time.perf_counter()-start)
            history.append(record)
            if record["validation_rmse_T"]<best:
                best=record["validation_rmse_T"];best_step=step;best_state=[p.detach().clone() for p in params]
            return record
        def restore():
            with torch.no_grad():
                for p,b in zip(params,best_state):p.copy_(b)
        optimizer=torch.optim.Adam(params,lr=config["learning_rate"]);checkpoint(0,"initial",objective().detach())
        for step in range(1,config["adam_steps"]+1):
            optimizer.zero_grad();loss=objective();loss.backward();torch.nn.utils.clip_grad_norm_(params,10);optimizer.step()
            if step%25==0:
                row=checkpoint(step,"Adam",loss.detach())
                if step%250==0:print(json.dumps(dict(name=config["name"],progress=row)),flush=True)
        final_adam=history[-1].copy();restore()
        if config["lbfgs_blocks"]:
            # Refinement is still train42-only; validation continues to choose a
            # checkpoint. Float64 makes small physical residual gradients usable.
            model.double()
            if skip is not None:skip.double()
            xt=xt.double();xall=xall.double();tt=tt.double();mt=mt.double();st=st.double()
            optimizer=torch.optim.LBFGS(params,lr=1.,max_iter=10,history_size=10,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-14)
            for outer in range(1,config["lbfgs_blocks"]+1):
                def closure():
                    nonlocal closures
                    optimizer.zero_grad();loss=objective();loss.backward();closures+=1;return loss
                optimizer.step(closure);row=checkpoint(config["adam_steps"]+outer*10,"LBFGS",objective().detach())
                if outer%10==0:print(json.dumps(dict(name=config["name"],progress=row)),flush=True)
        final_optimizer=history[-1].copy();restore();model.double()
        if skip is not None:skip.double()
        arrays=dict(feature_kind=np.asarray(config["features"]),current_scale=np.asarray(cs),harmonics=np.asarray(config["harmonics"]),interaction=np.asarray(config["interaction"]),transients=np.asarray(config["transients"]),
            feature_mean=fm,feature_scale=fs,coefficient_mean=cm,output_scale=sc,frame_count=np.asarray(frames),has_skip=np.asarray(config["skip"]),
            spatial_mean=rep["spatial_mean"],spatial_basis=rep["spatial_basis"][:,:rank],point_id=rep["point_id"],xy=rep["xy"],theta=rep["theta"],time_s=rep["time_s"])
        linear=[l for l in model if isinstance(l,torch.nn.Linear)];arrays["layer_count"]=np.asarray(len(linear))
        for j,l in enumerate(linear):arrays[f"w{j}"]=l.weight.detach().numpy().T;arrays[f"b{j}"]=l.bias.detach().numpy()
        if skip is not None:arrays["skip_weight"]=skip.weight.detach().numpy().T;arrays["skip_bias"]=skip.bias.detach().numpy()
        metadata=dict(version=VERSION,identity=identity,configuration=config,shared_weights_across_all_saved_frames=True,
            baseline="per-frame coefficient mean fitted to train42 only",feature_scalers="train42 only",output_scalers="train42 only",checkpoint="minimum raw full-validation field SSE",
            test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,frame_scope="the60 supplied first-cycle frames; raw time permits unequal endpoints",
            representation_sha256=sha(prepared/"representation.npz"),train_validation_data_sha256=sha(prepared/"train_validation_data.npz"),source_sha256=sha(__file__),best_step=best_step)
        arrays["metadata_json"]=np.asarray(json.dumps(metadata,ensure_ascii=False));np.savez_compressed(out/"model.npz",**arrays);(out/"model.npz.sha256").write_text(sha(out/"model.npz")+"\n")
        coef=predict(out/"model.npz",a["conditions"],return_coeff=True)
        with torch.no_grad():expected=normalized(torch.tensor(x.reshape(-1,x.shape[-1]),dtype=torch.float64)).numpy().reshape(target.shape)*sc+cm
        replay=float(np.max(np.abs(coef-expected)))
        if replay>1e-10:raise AssertionError(f"NumPy and Float64 Torch mismatch {replay}")
        raw=predict(out/"model.npz",a["conditions"][42:]);positive=np.maximum(raw,0)
        with np.load(prepared/"validation_truth.npz",allow_pickle=False) as z:y=z["truth"]
        if not np.isclose(np.sqrt(np.mean((raw-y)**2)),best,rtol=2e-4,atol=1e-7):raise AssertionError("Physical field reconstruction disagrees with coefficient checkpoint SSE")
        np.savez_compressed(out/"validation_predictions.npz",raw_prediction=raw,prediction=positive,case_ids=a["case_ids"][42:],quantity=np.asarray("|B|"),metadata_json=np.asarray(json.dumps(metadata,ensure_ascii=False)))
        with (out/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        tse=np.sum((coef[:42]-target[:42])**2)+oracle[:42].sum()
        result=dict(name=config["name"],configuration=config,validation=exact_metrics(raw,y,a["case_ids"][42:]),validation_nonnegative=exact_metrics(positive,y,a["case_ids"][42:]),
            train=dict(relative_L2_pct=float(100*np.sqrt(tse/a["truth_energy"][:42].sum())),rmse_T=float(np.sqrt(tse/(42*int(a["scalar_count"]))))),
            best_step=best_step,raw_negative_count=int(np.sum(raw<0)),raw_min_T=float(raw.min()),parameter_count=sum(p.numel() for p in params),elapsed_seconds=time.perf_counter()-start,
            final_adam=final_adam,final_optimizer=final_optimizer,min_train_relative_L2_pct=min(h["train_relative_L2_pct"] for h in history),numpy_torch_float64_replay_max_abs_difference=replay,
            model_sha256=sha(out/"model.npz"),validation_predictions_sha256=sha(out/"validation_predictions.npz"),lbfgs_closure_evaluations=closures,
            test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
        (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepared",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--config",type=Path,required=True)
    a=p.parse_args();train(a.prepared,a.output,json.loads(a.config.read_text(encoding="utf-8")))
