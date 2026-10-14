"""Train-only local mean-field kernel + frozen pureNN dynamic control.

This is an analytic kernel control with a neural dynamic component, not a pure
neural network. No full60 fitting labels or test labels are read. Native current
coordinates are mathematical maps, not assumed physical dq currents.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,sys
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def coordinates(c,kind,current_scale):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));r=c[:,0]/current_scale;a=np.deg2rad(c[:,1])
    if kind=='native':return np.column_stack((r,c[:,1]/90))
    if kind=='sat05':r=.5*np.tanh(r/.5)
    if kind=='sat1':r=np.tanh(r)
    if kind=='sat2':r=2*np.tanh(r/2)
    return np.column_stack((r*np.cos(a),r*np.sin(a)))

def fit_transform(train,kind):
    nonzero=train[:,0][train[:,0]>0]
    current_scale=float(np.median(nonzero));x=coordinates(train,kind,current_scale)
    center=x.mean(0);scale=np.maximum(x.std(0),1e-12)
    return (x-center)/scale,current_scale,center,scale

def kernel(x,y,kind,length):
    r2=np.sum((x[:,None,:]-y[None,:,:])**2,axis=-1)/(length*length)
    if kind=='gaussian':return np.exp(-.5*r2)
    if kind=='imq':return 1/np.sqrt(1+r2)
    if kind=='matern32':
        r=np.sqrt(3*r2);return (1+r)*np.exp(-r)
    raise ValueError(kind)

def poly(x,degree):
    if degree==0:return np.ones((len(x),1))
    if degree==1:return np.column_stack((np.ones(len(x)),x))
    raise ValueError(degree)

def weights(train_c,query_c,spec):
    x,cs,center,scale=fit_transform(train_c,spec['coordinate'])
    q=(coordinates(query_c,spec['coordinate'],cs)-center)/scale
    k=kernel(x,x,spec['kernel'],spec['length']);p=poly(x,spec['degree'])
    a=np.block([[k+spec['ridge']*np.eye(len(x)),p],[p.T,np.zeros((p.shape[1],p.shape[1]))]])
    rhs=np.concatenate((np.eye(len(x)),np.zeros((p.shape[1],len(x)))),axis=0)
    sol=np.linalg.solve(a,rhs)
    return np.column_stack((kernel(q,x,spec['kernel'],spec['length']),poly(q,spec['degree'])))@sol

def predict(model,conditions,frame_index=None):
    if isinstance(model,(str,Path)):
        path=Path(model)
        with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
        root=path.parent
    else:
        m=model;root=Path(str(m['model_directory']))
    meta=json.loads(str(m['metadata_json']))
    if meta['format']!='new60-static-local-control-v3':raise ValueError('Unsupported format')
    c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    if c.shape[1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Expected native [I_RMS>=0,Thet]')
    static=np.empty((len(c),m['train_mean_fields'].shape[1]),dtype=np.float64)
    for spec_id in np.unique(m['spec_id_for_point']):
        mask=m['spec_id_for_point']==spec_id
        w=weights(m['train_conditions'],c,meta['specifications'][int(spec_id)])
        static[:,mask]=w@m['train_mean_fields'][:,mask]
    dynamic_path=(root/meta['dynamic_model_relative_path']).resolve()
    module_path=(root/meta['dynamic_module_relative_path']).resolve()
    if digest(dynamic_path)!=meta['dynamic_model_sha256'] or digest(module_path)!=meta['dynamic_module_sha256']:
        raise ValueError('Frozen neural dependency SHA mismatch')
    import importlib.util
    modspec=importlib.util.spec_from_file_location('_frozen_cycle_nn_v3_control',module_path)
    mod=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(mod)
    raw=mod.predict(dynamic_path,c)
    norm=np.hypot(raw[...,0::2],raw[...,1::2]);dynamic=norm-norm.mean(1,keepdims=True)
    out=static[:,None,:]+dynamic
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<60:raise ValueError('Invalid fixed frame')
        out=out[:,frame_index]
    return out[0] if single else out
