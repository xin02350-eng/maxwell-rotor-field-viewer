"""Condition -> whole supplied cycle, with train-only condition POD.

Inference imports NumPy only. Training/validation label rows are read explicitly
from C-order NPY members; excluded test rows are never converted to arrays.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np

VERSION = "new60-cycle-neural-v2.1"
REGIONS = ("stator", "rotor", "airgap")


def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def write_new(p, data):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def selected_rows(path,key,indices):
    """Seek past excluded case bytes without numeric materialization."""
    index_to_output={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as archive,archive.open(key+".npy") as f:
        version=np.lib.format.read_magic(f)
        shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or len(shape)!=3 or shape[0]!=60 or dtype.kind!="f":raise ValueError("Invalid case-major label array")
        out=np.empty((len(indices),*shape[1:]),dtype=dtype)
        block=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(shape[0]):
            if i in index_to_output:
                raw=f.read(block)
                if len(raw)!=block:raise ValueError("Truncated label row")
                out[index_to_output[i]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(block,1)
    if not np.isfinite(out).all():raise ValueError("Nonfinite permitted labels")
    return out


def feature_values(conditions,kind,current_scale):
    c=np.atleast_2d(np.asarray(conditions,dtype=np.float64))
    if c.shape[1]!=2 or not np.isfinite(c).all():raise ValueError("Expected raw [I_RMS, native Thet] conditions")
    r=c[:,0]/current_scale;a=np.deg2rad(c[:,1]);x=r*np.cos(a);y=r*np.sin(a)
    if kind=="cartesian":return np.column_stack((x,y))
    if kind=="cartesian2":return np.column_stack((x,y,x*x,x*y,y*y))
    if kind=="mixed":return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))
    if kind=="native":return np.column_stack((r,c[:,1]/90))
    raise ValueError(kind)


def polynomial(conditions,degree,current_scale):
    x=feature_values(conditions,"cartesian",current_scale)
    return np.column_stack([x[:,0]**i*x[:,1]**j for total in range(degree+1) for i in range(total+1) for j in [total-i]])


def _model_dict(model):
    if isinstance(model,(str,Path)):
        path=Path(model)
        side=path.with_suffix(path.suffix+".sha256")
        if not side.exists() or digest(path)!=side.read_text().strip():raise ValueError("Model SHA256 mismatch")
        with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
        if "representation_relative_path" in m:
            rp=(path.parent/str(m["representation_relative_path"])).resolve()
            if digest(rp)!=str(m["representation_sha256"]):raise ValueError("Representation SHA256 mismatch")
            with np.load(rp,allow_pickle=False) as z:m.update({k:z[k] for k in z.files if k not in m})
        return m
    return model


def predict(model,conditions,frame_index=None,return_coeff=False,return_latent=False):
    """Pure NumPy: conditions[N,2] -> field[N,60,D], or selected saved frame.

    The fixed-cycle model does not claim interpolation to arbitrary new times.
    A condition pair [2] returns [60,D] without the batch dimension.
    """
    m=_model_dict(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    f=feature_values(c,str(m["feature_kind"]),float(m["current_scale"]))
    a=(f-m["feature_mean"])/m["feature_scale"]
    for i in range(int(m["layer_count"])):
        a=a@m[f"w{i}"]+m[f"b{i}"]
        if i<int(m["layer_count"])-1:
            if str(m["activation"])=="tanh":a=np.tanh(a)
            else:a=a/(1+np.exp(-np.clip(a,-700,700)))
    latent=a*m["output_scale"]+polynomial(c,int(m["base_degree"]),float(m["current_scale"]))@m["base_weights"]
    if return_latent:return latent[0] if single else latent
    coeff=(m["cycle_mean"]+latent@m["cycle_basis"]).reshape(len(c),int(m["frame_count"]),-1)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m["frame_count"]):raise ValueError("Invalid supplied frame index")
        coeff=coeff[:,frame_index]
    out=coeff if return_coeff else m["spatial_mean"]+coeff@m["spatial_basis"].T
    return out[0] if single else out


def metrics_from_latent(z,gram,cross,energy,truth_energy,scalar_count):
    sse=np.einsum("ci,ij,cj->c",z,gram,z)-2*np.sum(z*cross,axis=1)+energy
    if np.min(sse)<-1e-6:raise ValueError("Negative reconstruction SSE beyond roundoff")
    sse=np.maximum(sse,0)
    return dict(rmse_T=float(np.sqrt(sse.sum()/(len(z)*scalar_count))),
                relative_L2_pct=float(100*np.sqrt(sse.sum()/truth_energy.sum())),
                per_case_rmse_T=np.sqrt(sse/scalar_count).tolist())


def prepare(cache_path,split_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    split=read_json(split_path);tr=split["train"];va=split["validation"]
    if list(map(len,(tr,va,split["test"])))!=[42,9,9] or len(set(tr+va+split["test"]))!=60:raise ValueError("Invalid split")
    keep=tr+va
    keys=["basis","mean","conditions","point_id","xy","theta","time_s","case_ids","metadata_json","pod_train_case_indices"]
    with np.load(cache_path,allow_pickle=False) as z:a={k:z[k] for k in keys}
    metadata=json.loads(str(a["metadata_json"]));region=metadata["region"]
    if sorted(a["pod_train_case_indices"].tolist())!=sorted(tr):raise ValueError("Spatial POD not train42")
    if metadata["units"]!="T" or metadata["xy_units"]!="mm":raise ValueError("Unknown units")
    if a["case_ids"].tolist()!=split["case_ids"]:raise ValueError("Split/cache case IDs differ")
    labels=selected_rows(cache_path,"field",keep).astype(np.float64)
    coeff=selected_rows(cache_path,"coeff",keep).astype(np.float64)
    c,t,d=labels.shape;r=coeff.shape[-1]
    flat=coeff.reshape(c,-1);mean=flat[:42].mean(axis=0)
    u,s,vt=np.linalg.svd(flat[:42]-mean,full_matrices=False)
    rank=min(41,int(np.sum(s>s[0]*1e-10)));basis=vt[:rank]
    latent=(flat-mean)@basis.T
    sb=a["basis"].astype(np.float64);sm=a["mean"].astype(np.float64)
    gram=np.zeros((rank,rank));cross=np.zeros((c,rank));energy=np.zeros(c)
    truth_energy=np.sum(labels*labels,axis=(1,2),dtype=np.float64)
    for frame in range(t):
        field_basis=basis.reshape(rank,t,r)[:,frame]@sb.T
        baseline=sm+mean.reshape(t,r)[frame]@sb.T
        residual=labels[:,frame]-baseline
        gram+=field_basis@field_basis.T;cross+=residual@field_basis.T;energy+=np.sum(residual*residual,axis=1)
    # Gram/cross use actual permitted fields, not an assumed orthonormal basis.
    oracle=metrics_from_latent(latent[42:],gram,cross[42:],energy[42:],truth_energy[42:],t*d)
    exact_latent_oracle=np.linalg.solve(gram,cross[42:].T).T
    exact_oracle=metrics_from_latent(exact_latent_oracle,gram,cross[42:],energy[42:],truth_energy[42:],t*d)
    representation=dict(cycle_mean=mean,cycle_basis=basis,spatial_basis=a["basis"],spatial_mean=a["mean"],
        frame_count=np.asarray(t),point_id=a["point_id"],xy=a["xy"],theta=a["theta"],time_s=a["time_s"])
    np.savez_compressed(out/"representation.npz",**representation)
    np.savez_compressed(out/"train_validation_data.npz",conditions=a["conditions"][keep],latent=latent,
        gram=gram,cross=cross,energy=energy,truth_energy=truth_energy,scalar_count=np.asarray(t*d),
        case_ids=a["case_ids"][keep],train_count=np.asarray(42),quantity=np.asarray(metadata["quantity"]))
    np.savez_compressed(out/"validation_truth.npz",truth=labels[42:],case_ids=a["case_ids"][va])
    info=dict(version=VERSION,region=region,quantity=metadata["quantity"],components=metadata["components"],layout=metadata["layout"],
        units="T",xy_units="mm",train_case_indices=tr,validation_case_indices=va,excluded_test_case_indices=split["test"],
        train_case_ids=a["case_ids"][tr].tolist(),validation_case_ids=a["case_ids"][va].tolist(),
        cache_sha256=digest(cache_path),split_sha256=digest(split_path),source_sha256=digest(__file__),
        representation_sha256=digest(out/"representation.npz"),condition_POD_fit="training42 whole-cycle coefficient vectors only",
        spatial_POD_fit="original training42 only",cycle_rank=rank,spatial_rank=r,frame_count=t,spatial_dimension=d,
        label_rows_materialized=keep,test_labels_materialized=False,test_metrics_computed=False,
        validation_condition_POD_projection=oracle,validation_best_field_projection=exact_oracle,
        singular_values=s.tolist(),scope="new conditions, fixed supplied 60 frames and points; no forced temporal endpoint equality",
        feature_note="I*cos(Thet), I*sin(Thet) are mathematical input coordinates; not a claim that native Thet is a dq current angle")
    write_new(out/"representation_identity.json",info)
    return info


def save_model(path,arrays):
    np.savez_compressed(path,**arrays)
    path.with_suffix(path.suffix+".sha256").write_text(digest(path)+"\n",encoding="ascii")


def train(prepared,out,config):
    import torch
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    prepared,out=Path(prepared),Path(out);out.mkdir(parents=True,exist_ok=False)
    identity=read_json(prepared/"representation_identity.json")
    with np.load(prepared/"train_validation_data.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
    torch.manual_seed(config["seed"]);rng=np.random.default_rng(config["seed"])
    ntr=int(a["train_count"]);conditions=a["conditions"];latent=a["latent"]
    current_scale=max(float(np.max(np.abs(conditions[:ntr,0]))),1.)
    raw=feature_values(conditions,config["features"],current_scale);fm=raw[:ntr].mean(0);fs=raw[:ntr].std(0);fs=np.maximum(fs,1e-8)
    x=(raw-fm)/fs
    design=polynomial(conditions,config["base_degree"],current_scale)
    if config["base_degree"]==0:base=np.zeros((1,latent.shape[1]))
    else:
        lhs=design[:ntr].T@design[:ntr]+config["base_ridge"]*np.eye(design.shape[1])
        base=np.linalg.solve(lhs,design[:ntr].T@latent[:ntr])
    baseline=design@base
    output_scale=np.maximum(np.std(latent[:ntr]-baseline[:ntr],axis=0),np.std(latent[:ntr],axis=0).max()*1e-4)
    layers=[];dims=[x.shape[1],*config["hidden"],latent.shape[1]]
    for i in range(len(dims)-1):
        layer=torch.nn.Linear(dims[i],dims[i+1]);torch.nn.init.xavier_normal_(layer.weight);torch.nn.init.zeros_(layer.bias);layers.append(layer)
        if i<len(dims)-2:layers.append(torch.nn.Tanh() if config["activation"]=="tanh" else torch.nn.SiLU())
    net=torch.nn.Sequential(*layers)
    with torch.no_grad():net[-1].weight.zero_();net[-1].bias.zero_()
    xt=torch.tensor(x);bt=torch.tensor(baseline);scale=torch.tensor(output_scale);gt=torch.tensor(a["gram"])
    ct=torch.tensor(a["cross"][:ntr]);et=torch.tensor(a["energy"][:ntr]);den=float(a["truth_energy"][:ntr].sum())
    best=None;best_rmse=float("inf");best_step=0;history=[];closures=0;start=time.perf_counter()
    def objective():
        z=bt[:ntr]+net(xt[:ntr])*scale
        fit=(torch.sum((z@gt)*z)-2*torch.sum(z*ct)+et.sum())/den
        regularizer=sum(torch.mean(p*p) for name,p in net.named_parameters() if name.endswith("weight"))*config["regularization"]
        return fit+regularizer,fit
    def checkpoint(step,stage,loss,grad,lr):
        nonlocal best,best_rmse,best_step
        with torch.no_grad():z=(bt+net(xt)*scale).numpy()
        train_metrics=metrics_from_latent(z[:ntr],a["gram"],a["cross"][:ntr],a["energy"][:ntr],a["truth_energy"][:ntr],int(a["scalar_count"]))
        val_metrics=metrics_from_latent(z[ntr:],a["gram"],a["cross"][ntr:],a["energy"][ntr:],a["truth_energy"][ntr:],int(a["scalar_count"]))
        row=dict(step=step,stage=stage,loss=float(loss),gradient_norm=float(grad),learning_rate=float(lr),closure_evaluations=closures,
            train_rmse_T=train_metrics["rmse_T"],validation_rmse_T=val_metrics["rmse_T"],train_relative_L2_pct=train_metrics["relative_L2_pct"],validation_relative_L2_pct=val_metrics["relative_L2_pct"])
        history.append(row)
        if val_metrics["rmse_T"]<best_rmse:
            best_rmse=val_metrics["rmse_T"];best_step=step;best={k:v.detach().clone() for k,v in net.state_dict().items()}
        return row
    optimizer=torch.optim.Adam(net.parameters(),lr=config["learning_rate"])
    initial,_=objective();checkpoint(0,"initial",initial.detach(),0,config["learning_rate"])
    final_step=0
    for step in range(1,config["adam_steps"]+1):
        optimizer.zero_grad();loss,_=objective();loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.0);optimizer.step()
        if step%5==0 or step==config["adam_steps"]:checkpoint(step,"Adam",loss.detach(),grad,config["learning_rate"])
        final_step=step
        if step-best_step>=config["patience"] and step>=config["min_adam_steps"]:break
    # LBFGS starts from the best validation checkpoint; its gradients use train only.
    net.load_state_dict(best)
    optimizer=torch.optim.LBFGS(net.parameters(),lr=1,max_iter=10,history_size=50,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-14)
    for outer in range(1,config["lbfgs_blocks"]+1):
        def closure():
            nonlocal closures
            optimizer.zero_grad();loss,_=objective();loss.backward();closures+=1;return loss
        optimizer.step(closure)
        loss,_=objective();gradient=np.sqrt(sum(float(torch.sum(p.grad*p.grad)) for p in net.parameters() if p.grad is not None))
        checkpoint(final_step+outer*10,"LBFGS",loss.detach(),gradient,1)
    net.load_state_dict(best)
    arrays=dict(feature_kind=np.asarray(config["features"]),current_scale=np.asarray(current_scale),feature_mean=fm,feature_scale=fs,
        output_scale=output_scale,base_degree=np.asarray(config["base_degree"]),base_weights=base,
        activation=np.asarray(config["activation"]),representation_relative_path=np.asarray("../../prepared/representation.npz"),
        representation_sha256=np.asarray(identity["representation_sha256"]))
    linear=[m for m in net if isinstance(m,torch.nn.Linear)];arrays["layer_count"]=np.asarray(len(linear))
    for i,layer in enumerate(linear):arrays[f"w{i}"]=layer.weight.detach().numpy().T;arrays[f"b{i}"]=layer.bias.detach().numpy()
    metadata=dict(version=VERSION,**{k:identity[k] for k in ("region","quantity","components","layout","units","xy_units","train_case_ids","validation_case_ids")},
        configuration=config,best_step=best_step,checkpoint_partition="validation",test_labels_loaded=False,test_metrics_computed=False,
        source_sha256=digest(__file__),representation_sha256=identity["representation_sha256"],scope=identity["scope"],feature_note=identity["feature_note"])
    arrays["metadata_json"]=np.asarray(json.dumps(metadata,ensure_ascii=False))
    save_model(out/"model.npz",arrays)
    with torch.no_grad():expected=(bt+net(xt)*scale).numpy()
    replay=predict(out/"model.npz",conditions,return_latent=True)
    if not np.allclose(replay,expected,rtol=1e-10,atol=1e-10):raise AssertionError("NumPy neural replay differs")
    validation=predict(out/"model.npz",conditions[ntr:])
    with np.load(prepared/"validation_truth.npz",allow_pickle=False) as z:truth=z["truth"]
    error=validation-truth
    exact=dict(rmse_T=float(np.sqrt(np.mean(error*error))),mae_T=float(np.mean(np.abs(error))),max_abs_T=float(np.max(np.abs(error))),relative_L2_pct=float(100*np.sqrt(np.sum(error*error)/np.sum(truth*truth))))
    if not np.isclose(exact["rmse_T"],best_rmse,rtol=1e-8,atol=1e-9):raise AssertionError("Quadratic/full-field validation mismatch")
    np.savez_compressed(out/"validation_predictions.npz",prediction=validation,case_ids=a["case_ids"][ntr:],quantity=a["quantity"])
    np.savez_compressed(out/"train_validation_latent_predictions.npz",prediction=replay,case_ids=a["case_ids"])
    with (out/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    result=dict(name=config["name"],configuration=config,validation=exact,best_step=best_step,adam_final_step=final_step,lbfgs_closures=closures,
        train=metrics_from_latent(replay[:ntr],a["gram"],a["cross"][:ntr],a["energy"][:ntr],a["truth_energy"][:ntr],int(a["scalar_count"])),
        elapsed_seconds=time.perf_counter()-start,parameter_count=sum(p.numel() for p in net.parameters()),
        model_sha256=digest(out/"model.npz"),validation_prediction_sha256=digest(out/"validation_predictions.npz"),
        numpy_replay_max_abs_latent_difference=float(np.max(np.abs(replay-expected))),test_labels_loaded=False,test_metrics_computed=False)
    write_new(out/"result.json",result)
    print(json.dumps(dict(name=config["name"],validation=exact,seconds=result["elapsed_seconds"])),flush=True)
    return result


def candidate_configs():
    choices=[("direct_cart", "cartesian",[64,64],0,1e-5),
        ("direct_cart2","cartesian2",[128,128],0,1e-5),
        ("direct_mixed","mixed",[64,64,64],0,1e-4),
        ("poly2_small","cartesian2",[64,64],2,1e-5),
        ("poly2_wide","cartesian2",[128,128],2,1e-5),
        ("poly3_small","cartesian",[64,64],3,1e-5),
        ("poly2_mixed","mixed",[64,64],2,1e-4),
        ("poly2_deep","cartesian2",[64,64,64],2,1e-4)]
    return [dict(name=name+"_seed42",features=features,hidden=hidden,base_degree=degree,base_ridge=1e-6,
        regularization=reg,seed=42,activation="tanh",learning_rate=.003,adam_steps=3000,min_adam_steps=1000,
        patience=600,lbfgs_blocks=40) for name,features,hidden,degree,reg in choices]


def run(cache,split,root):
    from threadpoolctl import threadpool_limits
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    protocol=root/"candidate_protocol.json"
    if protocol.exists():raise FileExistsError("Use a fresh route/region directory; no silent reruns")
    configs=candidate_configs()
    write_new(protocol,dict(version=VERSION,scope="independent whole-cycle neural route, validation only",label_access="42 train +9 validation only; seek past excluded test rows",threads=2,
        candidates=configs,representation="train-only SVD of 42 flattened whole-cycle spatial-POD coefficient vectors; rank<=41",
        selection="lowest full-field validation RMSE; checkpoints and candidate recipe use validation only; ties prefer fewer neural parameters",
        confirmation="repeat best two seed42 recipes with seeds17 and73; select recipe by mean validation RMSE across three matched seeds",
        allowed_followup="diagnose train-only condition-subspace floor and training convergence; any expanded grid gets a new written protocol before runs",
        not_claimed=["held-test performance","unseen time","unseen spatial points","physical dq meaning of I*cos(native Thet)"]))
    with threadpool_limits(limits=2):
        info=prepare(cache,split,root/"prepared")
        print(json.dumps(dict(representation=info["cycle_rank"],validation_projection_floor=info["validation_best_field_projection"])),flush=True)
        results=[]
        for config in configs:results.append(train(root/"prepared",root/"runs"/config["name"],config))
        top=sorted(results,key=lambda r:(r["validation"]["rmse_T"],r["parameter_count"]))[:2]
        for best in top:
            for seed in (17,73):
                config=dict(best["configuration"],seed=seed,name=best["name"].replace("seed42",f"seed{seed}"))
                results.append(train(root/"prepared",root/"runs"/config["name"],config))
        recipes=[]
        for seed42 in top:
            stem=seed42["name"].replace("_seed42","");group=[r for r in results if r["name"].rsplit("_seed",1)[0]==stem]
            recipes.append(dict(recipe=stem,seeds=[r["configuration"]["seed"] for r in group],mean_validation_rmse_T=float(np.mean([r["validation"]["rmse_T"] for r in group])),runs=[r["name"] for r in group]))
        selected=min(recipes,key=lambda r:r["mean_validation_rmse_T"])
        # Freeze seed42 as the reproducible single model; report every seed.
        name=selected["recipe"]+"_seed42"
        m=_model_dict(root/"runs"/name/"model.npz");m.pop("representation_relative_path",None)
        save_model(root/"selected_model.npz",m)
        final=dict(version=VERSION,region=info["region"],selected=name,selected_recipe=selected,recipe_comparisons=recipes,rows=results,
            model_sha256=digest(root/"selected_model.npz"),protocol_sha256=digest(protocol),representation_sha256=info["representation_sha256"],
            selected_by="validation only; matched seed recipe mean, fixed seed42 deployment",test_labels_loaded=False,test_metrics_computed=False)
        write_new(root/"selection_frozen.json",final)
        (root/"selection_frozen.sha256").write_text(digest(root/"selection_frozen.json")+"\n",encoding="ascii")
        return final


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True);parser.add_argument("--split",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.cache,args.split,args.output)
