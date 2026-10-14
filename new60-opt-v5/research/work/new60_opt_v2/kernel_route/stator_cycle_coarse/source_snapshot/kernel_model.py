"""Pure-NumPy inference and fitting math for NEW60 condition-kernel studies."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def coordinates(conditions,kind,current_scale):
    x=np.asarray(conditions,dtype=np.float64)
    if x.shape[-1]!=2 or not np.isfinite(x).all() or np.any(x[...,0]<0):
        raise ValueError("Expected finite native I_RMS>=0, Thet condition pairs")
    current=x[...,0]/current_scale
    if kind=="polar_xy" or kind.startswith("saturated_"):
        if kind.startswith("saturated_"):
            factor=float(kind.split("_",1)[1]);current=factor*np.tanh(current/factor)
        phase=np.deg2rad(x[...,1]);return np.stack((current*np.cos(phase),current*np.sin(phase)),axis=-1)
    power={"native":1.,"sqrt_native":.5,"square_native":2.}[kind]
    return np.stack((current**power,x[...,1]/90.),axis=-1)


def transformed(conditions,kind,current_scale,center,scale,anisotropy):
    x=(coordinates(conditions,kind,current_scale)-center)/scale
    return x*np.asarray([np.sqrt(anisotropy),1/np.sqrt(anisotropy)])


def kernel(x,y,kind,lengthscale):
    r=np.sqrt(np.maximum(np.sum((x[:,None,:]-y[None,:,:])**2,axis=-1),0))/lengthscale
    if kind=="gaussian":return np.exp(-.5*r*r)
    if kind=="matern32":
        z=np.sqrt(3)*r;return (1+z)*np.exp(-z)
    if kind=="matern52":
        z=np.sqrt(5)*r;return (1+z+z*z/3)*np.exp(-z)
    if kind=="exponential":return np.exp(-r)
    if kind=="inverse_multiquadric":return 1/np.sqrt(1+r*r)
    if kind=="multiquadric":return -np.sqrt(1+r*r)
    if kind=="thin_plate":return r*r*np.log(np.maximum(r,1e-300))
    if kind=="cubic":return r**3
    raise ValueError(kind)


def polynomial(x,degree):
    if degree<0:return np.zeros((len(x),0))
    return np.stack([x[:,0]**i*x[:,1]**(total-i) for total in range(degree+1) for i in range(total+1)],axis=-1)


def solve_map(train_x,target,kind,lengthscale,ridge,degree):
    if kind=="polynomial_ridge":
        design=polynomial(train_x,degree)
        if ridge==0:
            weights=np.linalg.lstsq(design,target,rcond=1e-12)[0]
        else:
            weights=np.linalg.solve(design.T@design+ridge*np.eye(design.shape[1]),design.T@target)
        return weights,np.zeros((0,target.shape[1]))
    gram=kernel(train_x,train_x,kind,lengthscale)
    p=polynomial(train_x,degree)
    if p.shape[1]:
        matrix=np.block([[gram+ridge*np.eye(len(train_x)),p],[p.T,np.zeros((p.shape[1],p.shape[1]))]])
        rhs=np.concatenate((target,np.zeros((p.shape[1],target.shape[1]))),axis=0)
    else:
        matrix=gram+ridge*np.eye(len(train_x));rhs=target
    solution=np.linalg.solve(matrix,rhs)
    return solution[:len(train_x)],solution[len(train_x):]


def apply_map(x,train_x,weights,trend,kind,lengthscale,degree):
    if kind=="polynomial_ridge":return polynomial(x,degree)@weights
    return kernel(x,train_x,kind,lengthscale)@weights+polynomial(x,degree)@trend


def predict(model,conditions,frame_index=None,return_coeff=False):
    """Native [I_RMS,Thet] -> [60,D]; [N,2] -> [N,60,D]. NumPy only."""
    if isinstance(model,(str,Path)):
        with np.load(model,allow_pickle=False) as archive:
            model={key:archive[key] for key in archive.files}
    metadata=json.loads(str(model["metadata_json"]))
    if metadata["format"]!="new60_condition_kernel_v2":raise ValueError("Unknown model format")
    x=np.asarray(conditions,dtype=np.float64);one=x.ndim==1
    if one:x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=2:raise ValueError("Expected [2] or [N,2] native conditions")
    arrays=[]
    for block,spec in enumerate(metadata["blocks"]):
        prefix=f"b{block}_"
        z=transformed(x,spec["coordinate"],spec["current_scale"],model[prefix+"center"],model[prefix+"input_scale"],spec["anisotropy"])
        coeff=apply_map(z,model[prefix+"centers"],model[prefix+"weights"],model[prefix+"trend"],spec["kernel"],spec["lengthscale"],spec["degree"])
        if return_coeff:
            arrays.append(coeff);continue
        arrays.append(model[prefix+"mean"]+coeff@model[prefix+"basis"].T)
    if return_coeff:
        return arrays[0][0] if one and len(arrays)==1 else arrays[0] if len(arrays)==1 else arrays
    out=np.empty((len(x),metadata["time_count"]*metadata["field_dim"]),dtype=np.float64)
    for block,part in enumerate(arrays):out[:,model[f"b{block}_indices"]]=part
    out=out.reshape(len(x),metadata["time_count"],metadata["field_dim"])
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<metadata["time_count"]:raise ValueError("Invalid stored frame index")
        out=out[:,frame_index]
    return out[0] if one else out
