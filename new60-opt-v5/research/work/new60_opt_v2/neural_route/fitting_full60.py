"""Explicit all60 existing-case fitting branch; never a held-case prediction claim.

Run only after the independent prediction branch has been frozen. All60 labels
include cases formerly called validation/test; no information from this branch
may be used to change the independent prediction model or its selection.
"""
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np

VERSION="new60-all60-training-fit-v2.1"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def write_new(path,data):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def features(c,current_scale):
    c=np.atleast_2d(np.asarray(c,dtype=float))
    if c.shape[1]!=2 or not np.isfinite(c).all():raise ValueError("Expected [I_RMS, native Thet]")
    r=c[:,0]/current_scale;a=np.deg2rad(c[:,1]);x=r*np.cos(a);y=r*np.sin(a)
    return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))


def load_model(model):
    if not isinstance(model,(str,Path)):return model
    p=Path(model)
    if digest(p)!=p.with_suffix(p.suffix+".sha256").read_text().strip():raise ValueError("Model hash mismatch")
    with np.load(p,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    if "representation_relative_path" in m:
        rp=(p.parent/str(m["representation_relative_path"])).resolve()
        if digest(rp)!=str(m["representation_sha256"]):raise ValueError("Representation hash mismatch")
        with np.load(rp,allow_pickle=False) as z:m.update({k:z[k] for k in z.files if k not in m})
    return m


def predict(model,conditions,frame_index=None,nonnegative=True,return_latent=False):
    m=load_model(model);c=np.asarray(conditions,dtype=float);single=c.ndim==1;c=np.atleast_2d(c)
    x=(features(c,float(m["current_scale"]))-m["feature_mean"])/m["feature_scale"]
    for i in range(int(m["layer_count"])):
        x=x@m[f"w{i}"]+m[f"b{i}"]
        if i<int(m["layer_count"])-1:x=np.tanh(x)
    z=x*m["output_scale"]
    if return_latent:return z[0] if single else z
    out=(m["cycle_mean"]+z@m["cycle_basis"]).reshape(len(c),int(m["frame_count"]),-1)
    if nonnegative and str(m["quantity"])=="|B|":out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m["frame_count"]):raise ValueError("Invalid saved frame index")
        out=out[:,frame_index]
    return out[0] if single else out


def save_model(path,arrays):
    np.savez_compressed(path,**arrays);path=Path(path)
    path.with_suffix(path.suffix+".sha256").write_text(digest(path)+"\n",encoding="ascii")


def metrics(p,y):
    e=p-y;sse=np.sum(e*e)
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        relative_L2_pct=float(100*np.sqrt(sse/np.sum(y*y))))


def run(cache,out,prediction_freeze,quantity,seed=42):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(seed)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    # Require a completed independent prediction selection before reading any full60 numeric labels.
    frozen=json.loads(Path(prediction_freeze).read_text(encoding="utf-8-sig"))
    if frozen.get("test_labels_loaded") is not False or not frozen.get("selected_model_sha256"):raise ValueError("Prediction branch freeze missing or invalid")
    freeze_sha=digest(prediction_freeze)
    protocol=dict(version=VERSION,branch="fitting_full60",quantity=quantity,seed=seed,hidden=[256,256,256],threads=2,
        access="All60 case labels explicitly authorized for fitting; includes previous42train9validation9test",
        use="fitting error only on supplied60 cases*60frames*fixed points, no holdout or new-condition accuracy claim",
        independence="prediction branch was frozen before any full60 numeric label read; cannot modify its features/hyperparameters/selection",
        prediction_freeze_sha256=freeze_sha,source_sha256=digest(__file__),cache_sha256=digest(cache),
        representation="all60 centered whole-cycle SVD rank59, fitted solely in this full60 branch",
        loss="unregularized original-T whole-field MSE, optimized through orthonormal cycle coefficients",
        optimizer="Adam up to20000 steps, then LBFGS up to5000 iterations; training-only stopping at raw relativeL2<=0.05%",
        nonnegative_policy="optional max(0,raw) only for |B|; training/stopping/raw fitting error are unprojected")
    write_new(out/"protocol.json",protocol)
    with threadpool_limits(limits=2):
        with np.load(cache,allow_pickle=False) as z:
            a={k:z[k] for k in ("field","conditions","point_id","xy","theta","time_s","case_ids","metadata_json")}
        meta=json.loads(str(a["metadata_json"]))
        if a["field"].shape[0]!=60 or len(np.unique(a["case_ids"]))!=60 or meta["units"]!="T" or meta["xy_units"]!="mm":raise ValueError("Unexpected source identity/units")
        if quantity=="|B|":
            if meta["quantity"]!="Br,Bt" or meta["layout"]!="interleaved":raise ValueError("Magnitude requires signed Br/Bt source")
            field=np.hypot(a["field"][...,0::2].astype(float),a["field"][...,1::2].astype(float))
        elif quantity==meta["quantity"]=="Br":field=a["field"].astype(float)
        else:raise ValueError("Unsupported explicit target quantity")
        del a["field"]
        if not np.isfinite(field).all():raise ValueError("Nonfinite all60 field")
        n,t,d=field.shape;flat=field.reshape(n,-1);mean=flat.mean(0);center=flat-mean
        _,singular,vt=np.linalg.svd(center,full_matrices=False);basis=vt[:59];target=center@basis.T
        oracle_sse=np.maximum(np.sum(center*center,axis=1)-np.sum(target*target,axis=1),0)
        total_truth=float(np.sum(field*field));scalar_count=t*d;coef_scale=np.maximum(target.std(0),target.std(0).max()*1e-6)
        rep=dict(cycle_mean=mean,cycle_basis=basis,frame_count=np.asarray(t),point_id=a["point_id"],xy=a["xy"],theta=a["theta"],time_s=a["time_s"],quantity=np.asarray(quantity))
        save_model(out/"representation.npz",rep)
        cs=max(float(np.max(np.abs(a["conditions"][:,0]))),1);f=features(a["conditions"],cs);fm=f.mean(0);fs=np.maximum(f.std(0),1e-8);x=(f-fm)/fs
        dims=[x.shape[1],256,256,256,59]
        model=torch.nn.Sequential(*sum(([torch.nn.Linear(i,o),torch.nn.Tanh()] for i,o in zip(dims[:-2],dims[1:-1])),[]),torch.nn.Linear(dims[-2],dims[-1]))
        with torch.no_grad():model[-1].weight.zero_();model[-1].bias.zero_()
        xt=torch.tensor(x);yt=torch.tensor(target);st=torch.tensor(coef_scale)
        history=[];best=float("inf");best_state=None;best_step=0;closures=0;started=time.perf_counter()
        # A constant factor rescales original-T MSE for numerical optimization; no change in its minimizer.
        optimization_denominator=float(np.sum(center*center))
        def objective():return torch.sum((model(xt)*st-yt)**2)/optimization_denominator
        def checkpoint(step,stage,loss,grad):
            nonlocal best,best_state,best_step
            with torch.no_grad():prediction=(model(xt)*st).numpy()
            sse=float(np.sum((prediction-target)**2)+oracle_sse.sum());rmse=float(np.sqrt(sse/(n*scalar_count)));relative=float(100*np.sqrt(sse/total_truth))
            history.append(dict(step=step,stage=stage,objective=float(loss),gradient_norm=float(grad),fitting_rmse_T=rmse,fitting_relative_L2_pct=relative,closure_evaluations=closures))
            if sse<best:best=sse;best_step=step;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            return relative
        optimizer=torch.optim.Adam(model.parameters(),lr=.003);checkpoint(0,"initial",objective().detach(),0)
        for step in range(1,20001):
            optimizer.zero_grad();loss=objective();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            if step%25==0:
                rel=checkpoint(step,"Adam",loss.detach(),grad)
                if step%1000==0:print(json.dumps(dict(region=meta["region"],progress=history[-1])),flush=True)
                if rel<=.05:break
        adam_step=step;model.load_state_dict(best_state)
        if 100*np.sqrt(best/total_truth)>.05:
            optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=10,history_size=30,line_search_fn="strong_wolfe",tolerance_grad=1e-13,tolerance_change=1e-16)
            for outer in range(1,501):
                def closure():
                    nonlocal closures
                    optimizer.zero_grad();loss=objective();loss.backward();closures+=1;return loss
                optimizer.step(closure);loss=objective();grad=np.sqrt(sum(float(torch.sum(p.grad*p.grad)) for p in model.parameters() if p.grad is not None))
                rel=checkpoint(adam_step+outer*10,"LBFGS",loss.detach(),grad)
                if outer%50==0:print(json.dumps(dict(region=meta["region"],progress=history[-1])),flush=True)
                if rel<=.05:break
        model.load_state_dict(best_state)
        identity=dict(protocol,region=meta["region"],quantity=quantity,units="T",xy_units="mm",layout="scalar",components=[quantity],
            training_case_ids=a["case_ids"].tolist(),case_roles="all fitting; no validation/test within this branch",best_step=best_step,
            derivation="sqrt(Br^2+Bt^2), in-plane |B|" if quantity=="|B|" else "signed Br",model_version=VERSION,
            prediction_branch_unchanged=digest(prediction_freeze)==freeze_sha,representation_sha256=digest(out/"representation.npz"))
        arrays=dict(rep,feature_mean=fm,feature_scale=fs,current_scale=np.asarray(cs),output_scale=coef_scale,layer_count=np.asarray(4),metadata_json=np.asarray(json.dumps(identity,ensure_ascii=False)))
        layers=[l for l in model if isinstance(l,torch.nn.Linear)]
        for i,l in enumerate(layers):arrays[f"w{i}"]=l.weight.detach().numpy().T;arrays[f"b{i}"]=l.bias.detach().numpy()
        save_model(out/"model.npz",arrays)
        raw=predict(out/"model.npz",a["conditions"],nonnegative=False);projected=np.maximum(raw,0) if quantity=="|B|" else raw
        with torch.no_grad():expected=(model(xt)*st).numpy()
        replay=predict(out/"model.npz",a["conditions"],return_latent=True)
        if not np.allclose(replay,expected,rtol=1e-10,atol=1e-10):raise AssertionError("NumPy replay differs")
        np.savez_compressed(out/"fitting_predictions.npz",raw_prediction=raw,prediction=projected,case_ids=a["case_ids"],conditions=a["conditions"],quantity=np.asarray(quantity))
        with (out/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        result=dict(identity,fitting_raw=metrics(raw,field),fitting_nonnegative=metrics(projected,field),raw_negative_count=int(np.sum(raw<0)) if quantity=="|B|" else None,
            raw_min_T=float(raw.min()),parameter_count=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.perf_counter()-started,
            training_projection_floor_relative_L2_pct=float(100*np.sqrt(oracle_sse.sum()/total_truth)),
            model_sha256=digest(out/"model.npz"),prediction_sha256=digest(out/"fitting_predictions.npz"),numpy_replay_max_abs_latent_difference=float(np.max(np.abs(replay-expected))),
            target_reached=metrics(raw,field)["relative_L2_pct"]<=.05,heldout_metrics_computed=False)
        if digest(prediction_freeze)!=freeze_sha:raise AssertionError("Independent prediction freeze changed")
        write_new(out/"result.json",result);print(json.dumps(dict(region=meta["region"],fitting_raw=result["fitting_raw"],seconds=result["elapsed_seconds"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prediction-freeze",type=Path,required=True);p.add_argument("--quantity",choices=["|B|","Br"],required=True)
    a=p.parse_args();run(a.cache,a.output,a.prediction_freeze,a.quantity)
