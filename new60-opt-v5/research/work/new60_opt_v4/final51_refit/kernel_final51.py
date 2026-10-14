"""One prespecified analytic whole-cycle kernel/POD refit on train51 only.

This is NOT a neural network. No search or independent validation occurs.
Hyperparameters are inherited from the previous train42/val9 selected kernel.
Only explicit original train42+validation9 field rows are materialized.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import zipfile
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def coordinates(c,current_scale):
    c=np.asarray(c,dtype=np.float64)
    if c.shape[-1]!=2 or not np.isfinite(c).all() or np.any(c[...,0]<0):raise ValueError("Expected finite native I_RMS>=0,Thet")
    current=np.tanh(c[...,0]/current_scale);a=np.deg2rad(c[...,1]);return np.stack((current*np.cos(a),current*np.sin(a)),axis=-1)

def transformed(c,scale,center,input_scale,anisotropy):
    return (coordinates(c,scale)-center)/input_scale*np.asarray([np.sqrt(anisotropy),1/np.sqrt(anisotropy)])

def polynomial(x,degree):
    return np.stack([x[:,0]**i*x[:,1]**(total-i) for total in range(degree+1) for i in range(total+1)],axis=-1)

def kernel(x,y,length):
    r2=np.sum((x[:,None,:]-y[None,:,:])**2,axis=-1)/(length*length);return 1/np.sqrt(1+r2)

def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if sha(path)!=path.with_suffix(path.suffix+".sha256").read_text().strip():raise ValueError("Final51 kernel model SHA mismatch")
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def predict(model,conditions,frame_index=None,return_signed=False,return_coeff=False,nonnegative=False):
    m=load_model(model);metadata=json.loads(str(m["metadata_json"]));spec=metadata["spec"]
    x=np.asarray(conditions,dtype=np.float64);one=x.ndim==1;x=np.atleast_2d(x)
    if x.ndim!=2 or x.shape[1]!=2:raise ValueError("Expected [2] or [N,2] native condition inputs")
    z=transformed(x,spec["current_scale"],m["center"],m["input_scale"],spec["anisotropy"])
    coef=kernel(z,m["centers"],spec["lengthscale"])@m["weights"]+polynomial(z,spec["degree"])@m["trend"]
    if return_coeff:return coef[0] if one else coef
    signed=(m["mean"]+coef@m["basis"].T).reshape(len(x),60,3600)
    out=signed if return_signed else np.hypot(signed[...,0::2],signed[...,1::2])
    if nonnegative and not return_signed:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<60:raise ValueError("Invalid saved frame index")
        out=out[:,frame_index]
    return out[0] if one else out

def read_rows(path,indices):
    indices=list(indices);locations={v:i for i,v in enumerate(indices)}
    if len(indices)!=51 or len(set(indices))!=51:raise ValueError("Expected51 explicit unique training rows")
    with zipfile.ZipFile(path) as z,z.open("field.npy") as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or shape!=(60,60,3600) or dtype.kind!="f":raise ValueError("Unexpected rotor field storage")
        size=int(np.prod(shape[1:]))*dtype.itemsize;out=np.empty((51,*shape[1:]),dtype=np.float64)
        for case in range(shape[0]):
            if case in locations:
                raw=f.read(size)
                if len(raw)!=size:raise ValueError("Truncated selected row")
                out[locations[case]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(size,1)
    if not np.isfinite(out).all():raise ValueError("Nonfinite selected train51 field rows")
    return out

def metric(p,y,ids):
    e=p-y;sq=e*e;case=np.sqrt(np.mean(sq,axis=(1,2)));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape)
    return dict(relative_L2_pct=float(100*np.sqrt(np.sum(sq)/np.sum(y*y))),rmse_T=float(np.sqrt(np.mean(sq))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),worst_case_id=int(ids[np.argmax(case)]),worst_case_rmse_T=float(case.max()),
        max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),prediction_T=float(p[loc]),truth_T=float(y[loc]),signed_error_T=float(e[loc])),per_case=[dict(case_id=int(i),rmse_T=float(v)) for i,v in zip(ids,case)])

def train(cache,protocol_path,output):
    from threadpoolctl import threadpool_limits
    cache=Path(cache);protocol_path=Path(protocol_path);out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"));ids=protocol["train_case_ids"];excluded=protocol["excluded_test_case_ids"];spec=protocol["kernel_fixed_spec"]
    if len(ids)!=51 or set(ids)&set(excluded) or set(ids)|set(excluded)!=set(range(60)) or sha(cache)!=protocol["cache_sha256"]:raise ValueError("Train51 label identity/hash guard")
    if spec!={"coordinate":"saturated_1","anisotropy":2.8,"kernel":"inverse_multiquadric","lengthscale":3.72,"ridge":0.0001,"degree":2,"rank":24,"current_scale":237.57982517222536}:raise ValueError("Prespecified unique kernel guard")
    with np.load(cache,allow_pickle=False) as z:a={k:z[k] for k in ("conditions","case_ids","point_id","xy","theta","time_s")}
    if a["case_ids"].tolist()!=list(range(60)):raise ValueError("Expected original case IDs order")
    fields=read_rows(cache,ids);flat=fields.reshape(51,-1);c=a["conditions"][ids]
    with threadpool_limits(limits=2):
        mean=flat.mean(0);centered=flat-mean;eig,vec=np.linalg.eigh(centered@centered.T);order=np.argsort(eig)[::-1];eig=eig[order];vec=vec[:,order]
        if np.sum(eig>eig[0]*1e-12)<24:raise ValueError("Train51 POD insufficient rank24")
        basis=centered.T@(vec[:,:24]/np.sqrt(eig[:24]));basis,_=np.linalg.qr(basis,mode="reduced");coef=centered@basis
        rawx=coordinates(c,spec["current_scale"]);center=rawx.mean(0);input_scale=np.maximum(rawx.std(0),1e-10);x=transformed(c,spec["current_scale"],center,input_scale,spec["anisotropy"])
        p=polynomial(x,spec["degree"]);gram=kernel(x,x,spec["lengthscale"]);matrix=np.block([[gram+spec["ridge"]*np.eye(51),p],[p.T,np.zeros((p.shape[1],p.shape[1]))]])
        rhs=np.concatenate((coef,np.zeros((p.shape[1],24))),axis=0);sol=np.linalg.solve(matrix,rhs)
        meta=dict(format="new60_final51_kernel_v4",region="rotor",quantity="planar sqrt(Br²+Bt²), no Bz",units="T",spec=spec,training_original_ids=ids,former_validation_now_training_case_ids=protocol["former_validation_now_training_case_ids"],validation_original_ids=[],excluded_test_original_ids=excluded,
            cache_sha256=protocol["cache_sha256"],split_sha256=protocol["split_sha256"],protocol_sha256=sha(protocol_path),source_sha256=sha(__file__),representation="train51-only rank24 whole-cycle signed POD",checkpoint="no search; one fixed kernel/POD recipe inherited from train42/val9 selection",method="analytic inverse-multiquadric kernel with polynomial trend; NOT neural network",
            label_scope="original validation9 now training51; original9 test remain excluded; train51 metrics are fitting errors",test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,coordinate_caution="IcosThet and IsinThet are mathematical features, not verified dq-axis currents")
        model=dict(center=center,input_scale=input_scale,centers=x,weights=sol[:51],trend=sol[51:],mean=mean,basis=basis,metadata_json=np.asarray(json.dumps(meta,ensure_ascii=False)),point_id=a["point_id"],xy=a["xy"],theta=a["theta"],time_s=a["time_s"])
        np.savez_compressed(out/"model.npz",**model);(out/"model.npz.sha256").write_text(sha(out/"model.npz")+"\n")
        replay=predict(out/"model.npz",c,return_signed=True);expected=(mean+(gram@sol[:51]+p@sol[51:])@basis.T).reshape(51,60,3600);difference=float(np.max(np.abs(replay-expected)))
        if difference>1e-11:raise AssertionError("Final51 kernel equation replay mismatch")
        mag=np.hypot(replay[...,0::2],replay[...,1::2]);truth=np.hypot(fields[...,0::2],fields[...,1::2]);projection=mean+coef@basis.T
        result=dict(train51_fitting_magnitude=metric(mag,truth,ids),train51_fitting_signed_vector=metric(replay,fields,ids),signed_vector_train51_projection_relative_L2_pct=float(100*np.linalg.norm(flat-projection)/np.linalg.norm(flat)),train51_pod_orthogonality_max_abs=float(np.max(np.abs(basis.T@basis-np.eye(24)))),training_matrix_condition_number=float(np.linalg.cond(matrix)),
            prediction_equation_replay_max_abs_difference_T=difference,rank=24,spec=spec,model_sha256=sha(out/"model.npz"),source_sha256=sha(__file__),elapsed_seconds=time.perf_counter()-start,training_original_case_ids=ids,excluded_original_case_ids=excluded,validation_case_ids=[],test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,
            warning="Train51 includes former validation9; all reported errors here are training fitting errors, not held-out prediction errors; analytic kernel is not NN.")
    (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();train(a.cache,a.protocol,a.output)
