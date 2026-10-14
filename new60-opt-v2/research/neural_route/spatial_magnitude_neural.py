"""Train-only spatial magnitude POD and independent condition MLP per saved frame.

Inference is NumPy-only. Selection uses raw, original-T field MSE on validation.
This source never numerically materializes excluded test label rows.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from magnitude_cycle_neural import read_json,digest,write_new,selected_rows,save_model

VERSION="new60-spatial-magnitude-heads-v2.2"


def features(c,kind,current_scale):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64))
    if c.shape[1]!=2 or not np.isfinite(c).all():raise ValueError("Expected native [I_RMS, Thet] conditions")
    r=c[:,0]/current_scale;a=np.deg2rad(c[:,1])
    if kind=="native":return np.column_stack((r,c[:,1]/90))
    if kind.startswith("sat"):
        scale=float(kind[3:]);r=scale/current_scale*np.tanh(c[:,0]/scale)
    x,y=r*np.cos(a),r*np.sin(a)
    return np.column_stack((x,y,x*x,x*y,y*y))


def load_model(model):
    if not isinstance(model,(str,Path)):return model
    path=Path(model)
    if digest(path)!=path.with_suffix(path.suffix+".sha256").read_text().strip():raise ValueError("Model hash mismatch")
    with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    if "representation_relative_path" in m:
        rep=(path.parent/str(m["representation_relative_path"])).resolve()
        if digest(rep)!=str(m["representation_sha256"]):raise ValueError("Representation hash mismatch")
        with np.load(rep,allow_pickle=False) as z:m.update({k:z[k] for k in z.files if k not in m})
    return m


def predict(model,conditions,frame_index=None,nonnegative=True,return_coeff=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    x=(features(c,str(m["feature_kind"]),float(m["current_scale"]))-m["feature_mean"])/m["feature_scale"]
    indices=np.arange(int(m["frame_count"])) if frame_index is None else np.asarray([frame_index])
    if not np.issubdtype(indices.dtype,np.integer) or indices.min()<0 or indices.max()>=int(m["frame_count"]):raise ValueError("Invalid saved frame index")
    v=np.broadcast_to(x,(len(indices),*x.shape))
    for i in range(int(m["layer_count"])):
        v=np.einsum("tcf,tfh->tch",v,m[f"w{i}"][indices])+m[f"b{i}"][indices]
        if i<int(m["layer_count"])-1:v=np.tanh(v)
    coeff=(v*m["output_scale"][indices,None,:]+m["coefficient_mean"][indices,None,:]).transpose(1,0,2)
    out=coeff if return_coeff else m["spatial_mean"]+coeff@m["spatial_basis"].T
    if nonnegative and not return_coeff:out=np.maximum(out,0)
    if frame_index is not None:out=out[:,0]
    return out[0] if single else out


def exact_metrics(pred,truth):
    e=pred-truth;sse=np.sum(e*e)
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        relative_L2_pct=float(100*np.sqrt(sse/np.sum(truth*truth))))


def coeff_metrics(pred,a,sl):
    sse=np.sum((pred-a["coeff"][sl])**2,axis=(1,2))+a["oracle_sse"][sl]
    return dict(rmse_T=float(np.sqrt(sse.sum()/(len(pred)*int(a["scalar_count"])))),
        relative_L2_pct=float(100*np.sqrt(sse.sum()/a["truth_energy"][sl].sum())),
        per_case_rmse_T=np.sqrt(sse/int(a["scalar_count"])).tolist())


def prepare(cache,split_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);split=read_json(split_path)
    keep=split["train"]+split["validation"]
    if list(map(len,(split["train"],split["validation"],split["test"])))!=[42,9,9] or len(set(keep+split["test"]))!=60:raise ValueError("Split mismatch")
    with np.load(cache,allow_pickle=False) as z:a={k:z[k] for k in ("conditions","point_id","xy","theta","time_s","case_ids","metadata_json")}
    meta=json.loads(str(a["metadata_json"]))
    if meta["quantity"]!="Br,Bt" or meta["layout"]!="interleaved" or meta["units"]!="T" or meta["xy_units"]!="mm":raise ValueError("Quantity/layout/units mismatch")
    if a["case_ids"].tolist()!=split["case_ids"]:raise ValueError("Case identity mismatch")
    raw=selected_rows(cache,"field",keep).astype(float);field=np.hypot(raw[...,0::2],raw[...,1::2]);del raw
    n,t,d=field.shape;flat=field[:42].reshape(-1,d);mean=flat.mean(0)
    _,singular,vt=np.linalg.svd(flat-mean,full_matrices=False);basis=vt[:512].T
    centered=field-mean;coeff=centered@basis
    energy=np.sum(centered*centered,axis=(1,2));truth_energy=np.sum(field*field,axis=(1,2))
    np.savez_compressed(out/"representation.npz",spatial_mean=mean,spatial_basis=basis,frame_count=np.asarray(t),
        point_id=a["point_id"],xy=a["xy"],theta=a["theta"],time_s=a["time_s"])
    np.savez_compressed(out/"train_validation_data.npz",conditions=a["conditions"][keep],coeff=coeff,centered_energy=energy,
        truth_energy=truth_energy,case_ids=a["case_ids"][keep],scalar_count=np.asarray(t*d))
    np.savez_compressed(out/"validation_truth.npz",truth=field[42:],case_ids=a["case_ids"][keep][42:])
    floors={}
    for rank in (384,512):
        sse=np.maximum(energy[42:]-np.sum(coeff[42:,:,:rank]**2,axis=(1,2)),0)
        floors[str(rank)]=dict(rmse_T=float(np.sqrt(sse.sum()/(9*t*d))),relative_L2_pct=float(100*np.sqrt(sse.sum()/truth_energy[42:].sum())))
    identity=dict(version=VERSION,region=meta["region"],quantity="|B|",components=["|B|"],layout="scalar",units="T",xy_units="mm",
        derivation="sqrt(Br^2+Bt^2), in-plane magnitude; Bz unavailable",representation_fit="spatial POD on train42 * all60 saved frames only",
        validation_projection_floors=floors,train_case_ids=a["case_ids"][split["train"]].tolist(),validation_case_ids=a["case_ids"][split["validation"]].tolist(),
        allowed_label_rows=keep,excluded_label_rows=split["test"],cache_sha256=digest(cache),split_sha256=digest(split_path),
        source_sha256=digest(__file__),support_source_sha256=digest(Path(__file__).with_name("magnitude_cycle_neural.py")),
        representation_sha256=digest(out/"representation.npz"),test_labels_loaded=False,test_metrics_computed=False,
        scope="condition inference at the60 supplied saved frames, no arbitrary-time claim",nonnegative_policy="max(0,raw) explicitly optional; checkpoint/recipe selection uses raw field MSE")
    write_new(out/"identity.json",identity);print(json.dumps(dict(spatial_projection_floors=floors)),flush=True)


def train(prepared,out,config):
    import torch
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(config["seed"])
    prepared,out=Path(prepared),Path(out);out.mkdir(parents=True,exist_ok=False);identity=read_json(prepared/"identity.json")
    with np.load(prepared/"train_validation_data.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
    q=config["rank"];a["coeff"]=a["coeff"][:,:,:q];a["oracle_sse"]=np.maximum(a["centered_energy"]-np.sum(a["coeff"]**2,axis=(1,2)),0)
    target=a["coeff"];c=a["conditions"];frames=target.shape[1];cs=max(float(c[:42,0].max()),1)
    f=features(c,config["features"],cs);fm=f[:42].mean(0);fs=np.maximum(f[:42].std(0),1e-8);x=(f-fm)/fs
    cm=target[:42].mean(0);sc=np.maximum(np.std(target[:42],axis=0),np.std(target[:42],axis=0).max(axis=-1,keepdims=True)*.001)
    dims=[x.shape[1],*config["hidden"],q]
    weights=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(frames,i,o)*np.sqrt(2/(i+o))) for i,o in zip(dims[:-1],dims[1:])])
    biases=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(frames,1,o)) for o in dims[1:]])
    with torch.no_grad():weights[-1].zero_()
    parameters=list(weights)+list(biases);xt=torch.tensor(np.broadcast_to(x,(frames,*x.shape)).copy())
    tt=torch.tensor(target[:42].transpose(1,0,2));mt=torch.tensor(cm[:,None,:]);st=torch.tensor(sc[:,None,:])
    # Original-T field MSE: orthonormal spatial POD means coefficient SSE plus fixed projection residual.
    denominator=float(a["truth_energy"][:42].sum());history=[];best=float("inf");best_state=None;best_step=0;closures=0;start=time.perf_counter()
    def forward(inputs):
        v=inputs
        for i,(w,b) in enumerate(zip(weights,biases)):
            v=torch.bmm(v,w)+b
            if i<len(weights)-1:v=torch.tanh(v)
        return v*st+mt
    def objective():
        e=forward(xt[:,:42])-tt
        return torch.sum(e*e)/denominator+config["regularization"]*sum(torch.mean(w*w) for w in weights)
    def checkpoint(step,stage,loss,grad):
        nonlocal best,best_state,best_step
        with torch.no_grad():pred=forward(xt).numpy().transpose(1,0,2)
        tr=coeff_metrics(pred[:42],a,slice(0,42));va=coeff_metrics(pred[42:],a,slice(42,None))
        history.append(dict(step=step,stage=stage,objective=float(loss),gradient_norm=float(grad),train_rmse_T=tr["rmse_T"],validation_rmse_T=va["rmse_T"],
            train_relative_L2_pct=tr["relative_L2_pct"],validation_relative_L2_pct=va["relative_L2_pct"],closure_evaluations=closures))
        if va["rmse_T"]<best:best=va["rmse_T"];best_step=step;best_state=[p.detach().clone() for p in parameters]
    def restore():
        with torch.no_grad():
            for p,b in zip(parameters,best_state):p.copy_(b)
    optimizer=torch.optim.Adam(parameters,lr=.003);checkpoint(0,"initial",objective().detach(),0)
    for step in range(1,config["adam_steps"]+1):
        optimizer.zero_grad();loss=objective();loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,10);optimizer.step()
        if step%25==0:checkpoint(step,"Adam",loss.detach(),grad)
        if step%1000==0:print(json.dumps(dict(name=config["name"],progress=history[-1])),flush=True)
    # Train fully for diagnostic minimum loss; evaluate all checkpoints using validation only.
    final_adam=history[-1].copy();restore()
    optimizer=torch.optim.LBFGS(parameters,lr=1,max_iter=10,history_size=10,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-14)
    for outer in range(1,config["lbfgs_blocks"]+1):
        def closure():
            nonlocal closures
            optimizer.zero_grad();loss=objective();loss.backward();closures+=1;return loss
        optimizer.step(closure);loss=objective();grad=np.sqrt(sum(float(torch.sum(p.grad*p.grad)) for p in parameters if p.grad is not None))
        checkpoint(config["adam_steps"]+outer*10,"LBFGS",loss.detach(),grad)
    final_optimizer=history[-1].copy();restore()
    arrays=dict(feature_kind=np.asarray(config["features"]),current_scale=np.asarray(cs),feature_mean=fm,feature_scale=fs,output_scale=sc,
        coefficient_mean=cm,layer_count=np.asarray(len(weights)),representation_relative_path=np.asarray("../../prepared/representation.npz"),
        representation_sha256=np.asarray(identity["representation_sha256"]))
    for i,(w,b) in enumerate(zip(weights,biases)):arrays[f"w{i}"]=w.detach().numpy();arrays[f"b{i}"]=b.detach().numpy()
    # A rank-specific basis is embedded, while geometry/mean still comes from the shared hash-checked representation.
    with np.load(prepared/"representation.npz",allow_pickle=False) as z:arrays["spatial_basis"]=z["spatial_basis"][:,:q]
    arrays["metadata_json"]=np.asarray(json.dumps(dict(identity,configuration=config,best_step=best_step,checkpoint_partition="validation",model_version=VERSION),ensure_ascii=False))
    save_model(out/"model.npz",arrays)
    with torch.no_grad():expected=forward(xt).numpy().transpose(1,0,2)
    actual=predict(out/"model.npz",c,return_coeff=True)
    if not np.allclose(actual,expected,rtol=1e-10,atol=1e-10):raise AssertionError("NumPy replay mismatch")
    raw=predict(out/"model.npz",c[42:],nonnegative=False);positive=np.maximum(raw,0)
    with np.load(prepared/"validation_truth.npz",allow_pickle=False) as z:truth=z["truth"]
    metrics=exact_metrics(raw,truth)
    if not np.isclose(metrics["rmse_T"],best,rtol=1e-8,atol=1e-9):raise AssertionError("Field and coefficient metrics mismatch")
    np.savez_compressed(out/"validation_predictions.npz",prediction=positive,raw_prediction=raw,case_ids=a["case_ids"][42:],quantity=np.asarray("|B|"))
    np.savez_compressed(out/"train_validation_coefficient_predictions.npz",prediction=actual,case_ids=a["case_ids"])
    with (out/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    result=dict(name=config["name"],configuration=config,validation=metrics,validation_nonnegative=exact_metrics(positive,truth),
        raw_negative_count=int(np.sum(raw<0)),raw_min_T=float(raw.min()),best_step=best_step,train=coeff_metrics(actual[:42],a,slice(0,42)),
        final_adam=final_adam,final_optimizer=final_optimizer,min_train_relative_L2_pct=min(h["train_relative_L2_pct"] for h in history),
        elapsed_seconds=time.perf_counter()-start,lbfgs_closure_evaluations=closures,parameter_count=sum(p.numel() for p in parameters),
        model_sha256=digest(out/"model.npz"),validation_prediction_sha256=digest(out/"validation_predictions.npz"),
        numpy_replay_max_abs_coefficient_difference=float(np.max(np.abs(actual-expected))),test_labels_loaded=False,test_metrics_computed=False)
    write_new(out/"result.json",result);print(json.dumps(dict(name=config["name"],validation=metrics,train=result["train"],seconds=result["elapsed_seconds"])),flush=True)
    return result


def run(cache,split,root):
    from threadpoolctl import threadpool_limits
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    recipes=[("cart32",384,"cartesian",[32,32]),("sat150_32",384,"sat150",[32,32]),("sat300_32",384,"sat300",[32,32]),
        ("native32",384,"native",[32,32]),("sat150_64",384,"sat150",[64,64]),("sat150_r512",512,"sat150",[32,32])]
    configs=[dict(name=name+"_seed42",rank=rank,features=f,hidden=h,seed=42,regularization=1e-7,adam_steps=3000,lbfgs_blocks=80) for name,rank,f,h in recipes]
    write_new(root/"candidate_protocol.json",dict(version=VERSION,quantity="|B|",threads=2,candidates=configs,confirmation_seeds=[17,73],
        label_access="only numeric42train+9validation rows; excluded test rows skipped",representation="POD512 fitted to train42*60 magnitude spatial snapshots",
        loss="original T squared field residual; coefficient SSE plus fixed POD projection residual; no standardized coefficient training objective",
        selection="minimum raw validation field RMSE checkpoints; top2 recipes rerun seeds17,73; best three-seed mean recipe, deployment seed42",
        nonnegative_policy="raw predictions drive selection; explicit max(0,raw) optional physical output with separate metrics",
        training="full3000 Adam steps plus800 LBFGS iterations budget, no early stop; retain final/minimum train metrics to assess underfitting"))
    with threadpool_limits(limits=2):
        prepare(cache,split,root/"prepared");rows=[]
        for config in configs:rows.append(train(root/"prepared",root/"runs"/config["name"],config))
        top=sorted(rows,key=lambda r:(r["validation"]["rmse_T"],r["parameter_count"]))[:2]
        for row in top:
            for seed in (17,73):
                cfg=dict(row["configuration"],seed=seed,name=row["name"].replace("seed42",f"seed{seed}"))
                rows.append(train(root/"prepared",root/"runs"/cfg["name"],cfg))
        groups=[]
        for row in top:
            stem=row["name"].replace("_seed42","");matched=[r for r in rows if r["name"].rsplit("_seed",1)[0]==stem]
            groups.append(dict(recipe=stem,mean_validation_rmse_T=float(np.mean([r["validation"]["rmse_T"] for r in matched])),runs=[r["name"] for r in matched]))
        choice=min(groups,key=lambda r:r["mean_validation_rmse_T"]);name=choice["recipe"]+"_seed42"
        model=load_model(root/"runs"/name/"model.npz");model.pop("representation_relative_path",None);save_model(root/"selected_model.npz",model)
        frozen=dict(version=VERSION,quantity="|B|",selected=name,selected_recipe=choice,recipes=groups,rows=rows,source_sha256=digest(__file__),
            support_source_sha256=digest(Path(__file__).with_name("magnitude_cycle_neural.py")),selected_model_sha256=digest(root/"selected_model.npz"),
            test_labels_loaded=False,test_metrics_computed=False)
        write_new(root/"selection_frozen.json",frozen);(root/"selection_frozen.sha256").write_text(digest(root/"selection_frozen.json")+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True);p.add_argument("--split",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.cache,a.split,a.output)
