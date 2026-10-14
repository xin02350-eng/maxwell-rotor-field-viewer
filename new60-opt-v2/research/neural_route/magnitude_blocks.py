"""Independent condition networks per saved time block, direct in-plane |B|.

Unlike one whole-cycle affine condition span, different blocks can use different
condition maps. Pure NumPy inference; all fitting uses train42 and validation9.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from magnitude_cycle_neural import (read_json,digest,write_new,selected_rows,
                                    feature_values,polynomial,save_model)

VERSION="new60-magnitude-block-neural-v2.1"


def load_model(model):
    if isinstance(model,(str,Path)):
        path=Path(model)
        if digest(path)!=path.with_suffix(path.suffix+".sha256").read_text().strip():raise ValueError("Model hash mismatch")
        with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
        if "representation_relative_path" in m:
            p=(path.parent/str(m["representation_relative_path"])).resolve()
            if digest(p)!=str(m["representation_sha256"]):raise ValueError("Representation hash mismatch")
            with np.load(p,allow_pickle=False) as z:m.update({k:z[k] for k in z.files if k not in m})
        return m
    return model


def predict(model,conditions,frame_index=None,nonnegative=True,return_latent=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    x=(feature_values(c,str(m["feature_kind"]),float(m["current_scale"]))-m["feature_mean"])/m["feature_scale"]
    v=np.broadcast_to(x,(int(m["block_count"]),*x.shape))
    for i in range(int(m["layer_count"])):
        v=np.einsum("ecf,efh->ech",v,m[f"w{i}"])+m[f"b{i}"]
        if i<int(m["layer_count"])-1:v=np.tanh(v)
    base=np.einsum("cf,efq->ceq",polynomial(c,int(m["base_degree"]),float(m["current_scale"])),m["base_weights"])
    latent=v.transpose(1,0,2)*m["output_scale"]+base
    if return_latent:return latent[0] if single else latent
    out=(m["cycle_mean"]+np.einsum("ceq,eqd->ced",latent,m["cycle_basis"])).reshape(len(c),int(m["frame_count"]),-1)
    if nonnegative:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m["frame_count"]):raise ValueError("Invalid saved frame index")
        out=out[:,frame_index]
    return out[0] if single else out


def metric(z,a,select=slice(None)):
    cross,energy=a["cross"][select],a["energy"][select]
    sse=np.einsum("ceq,eqr,cer->ce",z,a["gram"],z)-2*np.sum(z*cross,axis=-1)+energy
    if sse.min()<-1e-6:raise ValueError("SSE roundoff failure")
    sse=np.maximum(sse,0).sum(1)
    return dict(rmse_T=float(np.sqrt(sse.sum()/(len(z)*int(a["scalar_count"])))),
        relative_L2_pct=float(100*np.sqrt(sse.sum()/a["truth_energy"][select].sum())),
        per_case_rmse_T=np.sqrt(sse/int(a["scalar_count"])).tolist())


def full_metrics(p,y):
    e=p-y
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),
        max_abs_T=float(np.max(np.abs(e))),relative_L2_pct=float(100*np.sqrt(np.sum(e*e)/np.sum(y*y))))


def prepare(cache,split_path,out,block_frames):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    split=read_json(split_path);keep=split["train"]+split["validation"]
    if list(map(len,(split["train"],split["validation"],split["test"])))!=[42,9,9] or len(set(keep+split["test"]))!=60:raise ValueError("Split mismatch")
    with np.load(cache,allow_pickle=False) as z:
        a={k:z[k] for k in ("conditions","point_id","xy","theta","time_s","case_ids","metadata_json")}
    meta=json.loads(str(a["metadata_json"]))
    if meta["quantity"]!="Br,Bt" or meta["layout"]!="interleaved":raise ValueError("Direct magnitude requires Br/Bt interleaved source")
    if a["case_ids"].tolist()!=split["case_ids"] or meta["units"]!="T" or meta["xy_units"]!="mm":raise ValueError("Identity or units mismatch")
    raw=selected_rows(cache,"field",keep).astype(float)
    field=np.hypot(raw[...,0::2],raw[...,1::2]);del raw
    n,t,d=field.shape
    if t%block_frames:raise ValueError("Block frame count must divide supplied sequence")
    blocks=t//block_frames;q=41;y=field.reshape(n,blocks,block_frames*d)
    mean=y[:42].mean(0);center=y-mean
    basis=np.stack([np.linalg.svd(center[:42,e],full_matrices=False)[2][:q] for e in range(blocks)])
    latent=np.einsum("ced,eqd->ceq",center,basis)
    data=dict(conditions=a["conditions"][keep],latent=latent,gram=np.einsum("eqd,erd->eqr",basis,basis),cross=latent.copy(),
        energy=np.sum(center*center,axis=-1),truth_energy=np.sum(field*field,axis=(1,2)),scalar_count=np.asarray(t*d),
        case_ids=a["case_ids"][keep],quantity=np.asarray("|B|"))
    representation=dict(cycle_mean=mean,cycle_basis=basis,block_frames=np.asarray(block_frames),block_count=np.asarray(blocks),
        frame_count=np.asarray(t),point_id=a["point_id"],xy=a["xy"],theta=a["theta"],time_s=a["time_s"])
    np.savez_compressed(out/"representation.npz",**representation)
    np.savez_compressed(out/"train_validation_data.npz",**data)
    np.savez_compressed(out/"validation_truth.npz",truth=field[42:],case_ids=a["case_ids"][keep][42:])
    info=dict(version=VERSION,region=meta["region"],quantity="|B|",components=["|B|"],layout="scalar",units="T",xy_units="mm",
        derivation="sqrt(Br^2+Bt^2); in-plane magnitude, no Bz",block_frames=block_frames,block_count=blocks,rank_per_block=q,
        representation_fit="only training42 magnitude fields; independent condition POD per contiguous saved-time block",
        validation_projection_floor=metric(latent[42:],data,slice(42,None)),test_labels_loaded=False,test_metrics_computed=False,
        train_case_ids=a["case_ids"][split["train"]].tolist(),validation_case_ids=a["case_ids"][split["validation"]].tolist(),
        allowed_label_rows=keep,excluded_label_rows=split["test"],cache_sha256=digest(cache),split_sha256=digest(split_path),
        source_sha256=digest(__file__),support_source_sha256=digest(Path(__file__).with_name("magnitude_cycle_neural.py")),
        representation_sha256=digest(out/"representation.npz"),scope="condition inference at fixed 60 saved frames/points; no interpolation across arbitrary times",
        nonnegative_policy="max(0,raw) is explicit physical range projection; candidate/checkpoint selection uses raw field MSE")
    write_new(out/"identity.json",info)
    print(json.dumps(dict(block_frames=block_frames,validation_projection_floor=info["validation_projection_floor"])),flush=True)
    return info


def train(prepared,out,config):
    import torch
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(config["seed"])
    prepared,out=Path(prepared),Path(out);out.mkdir(parents=True,exist_ok=False)
    identity=read_json(prepared/"identity.json")
    with np.load(prepared/"train_validation_data.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
    conditions=a["conditions"];target=a["latent"];blocks=target.shape[1];q=target.shape[2]
    current_scale=max(float(np.max(np.abs(conditions[:42,0]))),1.)
    f=feature_values(conditions,config["features"],current_scale);fm=f[:42].mean(0);fs=np.maximum(f[:42].std(0),1e-8);x=(f-fm)/fs
    design=polynomial(conditions,config["base_degree"],current_scale)
    base=np.zeros((blocks,design.shape[1],q))
    if config["base_degree"]:
        flat=np.linalg.solve(design[:42].T@design[:42]+1e-6*np.eye(design.shape[1]),design[:42].T@target[:42].reshape(42,-1))
        base=flat.reshape(design.shape[1],blocks,q).transpose(1,0,2)
    baseline=np.einsum("cf,efq->ceq",design,base)
    output_scale=np.maximum(np.std(target[:42]-baseline[:42],axis=0),np.std(target[:42],axis=0).max(axis=-1,keepdims=True)*1e-4)
    dims=[x.shape[1],*config["hidden"],q]
    weights=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(blocks,i,o)*np.sqrt(2/(i+o))) for i,o in zip(dims[:-1],dims[1:])])
    biases=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(blocks,1,o)) for o in dims[1:]])
    with torch.no_grad():weights[-1].zero_()
    parameters=list(weights)+list(biases)
    xt=torch.tensor(np.broadcast_to(x,(blocks,*x.shape)).copy());bt=torch.tensor(baseline.transpose(1,0,2));sc=torch.tensor(output_scale[:,None,:])
    gram=torch.tensor(a["gram"]);cross=torch.tensor(a["cross"][:42].transpose(1,0,2));energy=torch.tensor(a["energy"][:42].T)
    denominator=float(a["truth_energy"][:42].sum());history=[];best_rmse=float("inf");best=None;best_step=0;closures=0;start=time.perf_counter()
    def forward(inputs):
        v=inputs
        for i,(w,b) in enumerate(zip(weights,biases)):
            v=torch.bmm(v,w)+b
            if i<len(weights)-1:v=torch.tanh(v)
        return v
    def objective():
        z=bt[:,:42]+forward(xt[:,:42])*sc
        loss=(torch.sum(torch.bmm(z,gram)*z)-2*torch.sum(z*cross)+energy.sum())/denominator
        return loss+config["regularization"]*sum(torch.mean(w*w) for w in weights)
    def checkpoint(step,stage,loss,grad):
        nonlocal best,best_rmse,best_step
        with torch.no_grad():z=(bt+forward(xt)*sc).numpy().transpose(1,0,2)
        tr=metric(z[:42],a,slice(0,42));va=metric(z[42:],a,slice(42,None))
        history.append(dict(step=step,stage=stage,objective=float(loss),gradient_norm=float(grad),train_rmse_T=tr["rmse_T"],validation_rmse_T=va["rmse_T"],
            train_relative_L2_pct=tr["relative_L2_pct"],validation_relative_L2_pct=va["relative_L2_pct"],closure_evaluations=closures))
        if va["rmse_T"]<best_rmse:best_rmse=va["rmse_T"];best_step=step;best=[p.detach().clone() for p in parameters]
    def restore():
        with torch.no_grad():
            for p,b in zip(parameters,best):p.copy_(b)
    opt=torch.optim.Adam(parameters,lr=.003);checkpoint(0,"initial",objective().detach(),0)
    for step in range(1,config["adam_steps"]+1):
        opt.zero_grad();loss=objective();loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,10);opt.step()
        if step%10==0:checkpoint(step,"Adam",loss.detach(),grad)
        if step-best_step>=600 and step>=1000:break
    adam_final=step;restore()
    opt=torch.optim.LBFGS(parameters,lr=1,max_iter=10,history_size=30,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-14)
    for outer in range(1,config["lbfgs_blocks"]+1):
        def closure():
            nonlocal closures
            opt.zero_grad();loss=objective();loss.backward();closures+=1;return loss
        opt.step(closure);loss=objective()
        grad=np.sqrt(sum(float(torch.sum(p.grad*p.grad)) for p in parameters if p.grad is not None))
        checkpoint(adam_final+outer*10,"LBFGS",loss.detach(),grad)
    restore()
    arrays=dict(feature_kind=np.asarray(config["features"]),current_scale=np.asarray(current_scale),feature_mean=fm,feature_scale=fs,
        base_degree=np.asarray(config["base_degree"]),base_weights=base,output_scale=output_scale,layer_count=np.asarray(len(weights)),
        representation_relative_path=np.asarray(f"../../prepared_b{identity['block_frames']}/representation.npz"),
        representation_sha256=np.asarray(identity["representation_sha256"]))
    for i,(w,b) in enumerate(zip(weights,biases)):arrays[f"w{i}"]=w.detach().numpy();arrays[f"b{i}"]=b.detach().numpy()
    arrays["metadata_json"]=np.asarray(json.dumps(dict(identity,configuration=config,best_step=best_step,checkpoint_partition="validation",model_version=VERSION),ensure_ascii=False))
    save_model(out/"model.npz",arrays)
    with torch.no_grad():expected=(bt+forward(xt)*sc).numpy().transpose(1,0,2)
    latent=predict(out/"model.npz",conditions,return_latent=True)
    if not np.allclose(latent,expected,rtol=1e-10,atol=1e-10):raise AssertionError("NumPy replay differs")
    raw=predict(out/"model.npz",conditions[42:],nonnegative=False);positive=np.maximum(raw,0)
    with np.load(prepared/"validation_truth.npz",allow_pickle=False) as z:truth=z["truth"]
    exact=full_metrics(raw,truth)
    if not np.isclose(exact["rmse_T"],best_rmse,rtol=1e-8,atol=1e-9):raise AssertionError("Quadratic/raw field metric differs")
    np.savez_compressed(out/"validation_predictions.npz",prediction=positive,raw_prediction=raw,case_ids=a["case_ids"][42:],quantity=np.asarray("|B|"))
    np.savez_compressed(out/"train_validation_latent_predictions.npz",prediction=latent,case_ids=a["case_ids"])
    with (out/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    result=dict(name=config["name"],configuration=config,validation=exact,validation_nonnegative=full_metrics(positive,truth),
        raw_negative_count=int(np.sum(raw<0)),raw_min_T=float(raw.min()),best_step=best_step,adam_final_step=adam_final,
        train=metric(latent[:42],a,slice(0,42)),elapsed_seconds=time.perf_counter()-start,lbfgs_closure_evaluations=closures,
        parameter_count=sum(p.numel() for p in parameters),model_sha256=digest(out/"model.npz"),
        numpy_replay_max_abs_latent_difference=float(np.max(np.abs(latent-expected))),test_labels_loaded=False,test_metrics_computed=False)
    write_new(out/"result.json",result)
    print(json.dumps(dict(name=config["name"],validation=exact,nonnegative=result["validation_nonnegative"],seconds=result["elapsed_seconds"])),flush=True)
    return result


def run(cache,split,root):
    from threadpoolctl import threadpool_limits
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    configurations=[]
    for frames in (1,3):
        for label,features,hidden,degree,reg in (("direct_small","cartesian2",[32,32],0,1e-6),
            ("direct_wide","mixed",[64,64],0,1e-6),("poly2_small","cartesian2",[32,32],2,1e-6),
            ("poly2_reg","cartesian",[64,64],2,1e-4)):
            configurations.append(dict(name=f"b{frames}_{label}_seed42",block_frames=frames,features=features,hidden=hidden,base_degree=degree,
                regularization=reg,seed=42,adam_steps=2500,lbfgs_blocks=30))
    write_new(root/"candidate_protocol.json",dict(version=VERSION,quantity="|B|",threads=2,candidates=configurations,
        label_access="Only42train+9validation, no test numeric arrays",representation="Independent train-only condition POD for each contiguous1or3frame block",
        selection="raw field-MSE validation checkpoint; compare matched seed mean for two best recipes; fixed seed42 export",nonnegative_policy="predict max(0,raw), both raw/projected metrics always reported",
        confirmation_seeds=[17,73],reason="whole-cycle magnitude condition-span validation floor1.07497% requires more expressive framewise condition maps"))
    with threadpool_limits(limits=2):
        for frames in (1,3):prepare(cache,split,root/f"prepared_b{frames}",frames)
        results=[]
        for config in configurations:results.append(train(root/f"prepared_b{config['block_frames']}",root/"runs"/config["name"],config))
        top=sorted(results,key=lambda r:(r["validation"]["rmse_T"],r["parameter_count"]))[:2]
        for row in top:
            for seed in (17,73):
                cfg=dict(row["configuration"],seed=seed,name=row["name"].replace("seed42",f"seed{seed}"))
                results.append(train(root/f"prepared_b{cfg['block_frames']}",root/"runs"/cfg["name"],cfg))
        groups=[]
        for row in top:
            stem=row["name"].replace("_seed42","");matched=[r for r in results if r["name"].rsplit("_seed",1)[0]==stem]
            groups.append(dict(recipe=stem,mean_validation_rmse_T=float(np.mean([r["validation"]["rmse_T"] for r in matched])),runs=[r["name"] for r in matched]))
        choice=min(groups,key=lambda r:r["mean_validation_rmse_T"]);name=choice["recipe"]+"_seed42"
        model=load_model(root/"runs"/name/"model.npz");model.pop("representation_relative_path",None);save_model(root/"selected_model.npz",model)
        frozen=dict(version=VERSION,quantity="|B|",selected=name,selected_recipe=choice,recipes=groups,rows=results,
            source_sha256=digest(__file__),support_source_sha256=digest(Path(__file__).with_name("magnitude_cycle_neural.py")),
            selected_model_sha256=digest(root/"selected_model.npz"),test_labels_loaded=False,test_metrics_computed=False)
        write_new(root/"selection_frozen.json",frozen);(root/"selection_frozen.sha256").write_text(digest(root/"selection_frozen.json")+"\n")
        return frozen


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True);p.add_argument("--split",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();run(args.cache,args.split,args.output)
