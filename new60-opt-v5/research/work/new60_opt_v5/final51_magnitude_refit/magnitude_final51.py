"""One prespecified V5 train51 magnetic magnitude-objective neural fine-tune.
No independent validation; original9 test/full60 fitting labels excluded.
Frozen actualtrain51 V4 meanNN scalers and train42 old neural dynamics.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import zipfile
import numpy as np

VERSION="v5-final51-magnitude-objective-fine-tuned-gate2-neural"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def selected_rows(path,key,indices):
    mapping={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or len(shape)!=3 or shape[0]!=60 or shape[1:]!=(60,3600) or dtype.kind!='f':raise ValueError('Invalid full source field geometry')
        out=np.empty((len(indices),*shape[1:]),dtype=dtype);block=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(shape[0]):
            if i in mapping:
                raw=f.read(block)
                if len(raw)!=block:raise ValueError('Short selected field row')
                out[mapping[i]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(block,1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite selected train labels')
    return out

def features(c,kind,scale):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64))
    if c.shape[1]!=2 or not np.isfinite(c).all():raise ValueError('Expected native [I_RMS_A, Thet_deg]')
    r=c[:,0]/scale;a=np.deg2rad(c[:,1]);x=r*np.cos(a);y=r*np.sin(a)
    if kind=='cartesian':return np.column_stack((x,y))
    if kind=='mixed':return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))
    if kind=='cartesian2':return np.column_stack((x,y,x*x,x*y,y*y))
    raise ValueError('Unknown feature specification')

def sigmoid(v):
    z=np.exp(-np.abs(v));return np.where(v>=0,1/(1+z),z/(1+z))

def baseline_predict(m,c):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));scale=float(m['current_scale'])
    v=(features(c,str(m['feature_kind']),scale)-m['feature_mean'])/m['feature_scale']
    for j in range(int(m['layer_count'])):
        v=v@m[f'w{j}']+m[f'b{j}']
        if j<int(m['layer_count'])-1:v=np.tanh(v) if str(m['activation'])=='tanh' else v*sigmoid(v)
    x=features(c,'cartesian',scale);degree=int(m['base_degree'])
    poly=np.column_stack([x[:,0]**i*x[:,1]**j for total in range(degree+1) for i in range(total+1) for j in [total-i]])
    latent=v*m['output_scale']+poly@m['base_weights'];coef=(m['cycle_mean']+latent@m['cycle_basis']).reshape(len(c),int(m['frame_count']),-1)
    return m['spatial_mean']+coef@m['spatial_basis'].T

def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if sha(path)!=path.with_suffix(path.suffix+'.sha256').read_text().strip():raise ValueError('Magnitude neural model SHA mismatch')
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def predict_mean(m,c):
    x=(features(c,'cartesian',float(m['current_scale']))-m['feature_mean'])/m['feature_scale']
    v=np.einsum('cf,pfh->pch',x,m['linear_weight'])+m['linear_bias'][:,None,:]
    z=np.einsum('cf,pfk->pck',x,m['gate_weight'])+m['gate_bias'][:,None,:]
    aug=np.column_stack((np.ones(len(x)),x));expert=np.einsum('cf,pkfh->pckh',aug,m['expert_weight'])
    v=v+np.sum(sigmoid(z)[...,None]*expert,axis=2)
    return (v*m['output_scale'][:,None,:]+m['output_mean'][:,None,:]).transpose(1,0,2)

def predict(model,conditions,frame_index=None,nonnegative=False,return_signed=False,return_mean=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c);mean=predict_mean(m,c)
    if return_mean:return mean[0] if single else mean
    base={k[9:]:v for k,v in m.items() if k.startswith('baseline_')};old=baseline_predict(base,c).reshape(len(c),int(m['frame_count']),int(m['point_count']),2)
    signed=mean[:,None,:,:]+float(m['dynamic_scale'])*(old-old.mean(1,keepdims=True))
    out=signed.reshape(len(c),int(m['frame_count']),-1) if return_signed else np.hypot(signed[...,0],signed[...,1])
    if nonnegative and not return_signed:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m['frame_count']):raise ValueError('Invalid saved frame index')
        out=out[:,frame_index]
    return out[0] if single else out

def metrics(p,y,ids,point_ids):
    e=p-y;se=np.sum(e*e,axis=(1,2));energy=np.sum(y*y,axis=(1,2));rm=np.sqrt(se/np.prod(y.shape[1:]));worst=int(np.argmax(rm));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape)
    return dict(relative_L2_pct=float(100*np.sqrt(se.sum()/energy.sum())),rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),worst_case_id=int(ids[worst]),worst_case_rmse_T=float(rm[worst]),
        max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),point_id=int(point_ids[loc[2]]),prediction_T=float(p[loc]),truth_T=float(y[loc]),signed_error_T=float(e[loc])),
        per_case=[dict(case_id=int(k),rmse_T=float(r),relative_L2_pct=float(100*np.sqrt(s/den))) for k,r,s,den in zip(ids,rm,se,energy)])

def mean_metrics(pred,target):
    e=pred-target;pm=np.hypot(pred[...,0],pred[...,1]);tm=np.hypot(target[...,0],target[...,1]);me=pm-tm
    return dict(signed_static_mean_relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(target)),signed_static_mean_component_rmse_T=float(np.sqrt(np.mean(e*e))),signed_static_mean_component_max_abs_T=float(np.abs(e).max()),
        norm_of_signed_mean_relative_L2_pct=float(100*np.linalg.norm(me)/np.linalg.norm(tm)),norm_of_signed_mean_rmse_T=float(np.sqrt(np.mean(me*me))),norm_of_signed_mean_max_abs_T=float(np.abs(me).max()),quantity_note='Static signed mean/norm(mean), distinct from fullframe |B| and mean(|B|)')
