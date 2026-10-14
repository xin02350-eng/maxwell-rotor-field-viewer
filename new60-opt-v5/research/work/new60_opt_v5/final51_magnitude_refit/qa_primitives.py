"""Independent NumPy scalar/fullfield primitives copied from immutable42 QA.
No training, labels, model selection, or executable source import on load.
"""
from pathlib import Path
import builtins
import hashlib
import math
import numpy as np
checks=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def check(name,ok,detail=None):
    checks.append(dict(name=name,passed=bool(ok),detail=detail))
    if not ok:raise AssertionError(name+': '+str(detail))

def close(name,a,b,atol=1e-12,rtol=1e-12):
    a=np.asarray(a);b=np.asarray(b)
    check(name,a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol),float(np.max(np.abs(a-b))) if a.shape==b.shape else 'shape')

def static_forward(m,c,point):
    result=[]
    for current,theta in c:
        angle=float(theta)*math.pi/180;f=np.asarray([current/float(m['current_scale'])*math.cos(angle),current/float(m['current_scale'])*math.sin(angle)]);x=(f-m['feature_mean'])/m['feature_scale'];v=m['linear_bias'][point].copy()
        for j in range(2):v+=x[j]*m['linear_weight'][point,j]
        for gate in range(m['gate_bias'].shape[1]):
            z=float(m['gate_bias'][point,gate])
            for j in range(2):z+=float(x[j]*m['gate_weight'][point,j,gate])
            g=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z));e=m['expert_weight'][point,gate,0].copy()
            for j in range(2):e+=x[j]*m['expert_weight'][point,gate,j+1]
            v+=g*e
        result.append(v*m['output_scale'][point]+m['output_mean'][point])
    return np.asarray(result)

def field_metrics(p,y,ids,point_ids):
    e=p-y;se=np.square(e).sum(axis=(1,2));energy=np.square(y).sum(axis=(1,2));rm=np.sqrt(se/(60*1800));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape);worst=int(np.argmax(rm))
    return dict(relative_L2_pct=float(100*np.sqrt(se.sum()/energy.sum())),rmse_T=float(np.sqrt(np.square(e).mean())),mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),worst_case_id=int(ids[worst]),worst_case_rmse_T=float(rm[worst]),
        max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),point_id=int(point_ids[loc[2]]),prediction_T=float(p[loc]),truth_T=float(y[loc]),signed_error_T=float(e[loc])),
        per_case=[dict(case_id=int(cid),rmse_T=float(rr),relative_L2_pct=float(100*np.sqrt(ss/ee))) for cid,rr,ss,ee in zip(ids,rm,se,energy)])

def mean_metrics(p,y):
    e=p-y;pn=np.sqrt(np.sum(p*p,axis=-1));yn=np.sqrt(np.sum(y*y,axis=-1));me=pn-yn
    return dict(signed_static_mean_relative_L2_pct=float(100*np.sqrt(np.square(e).sum()/np.square(y).sum())),signed_static_mean_component_rmse_T=float(np.sqrt(np.square(e).mean())),signed_static_mean_component_max_abs_T=float(np.abs(e).max()),norm_of_signed_mean_relative_L2_pct=float(100*np.sqrt(np.square(me).sum()/np.square(yn).sum())),norm_of_signed_mean_rmse_T=float(np.sqrt(np.square(me).mean())),norm_of_signed_mean_max_abs_T=float(np.abs(me).max()))

def infer(mod,path,c,**kwargs):
    old_import=builtins.__import__;old_load=np.load;opened=[]
    def imp(name,*a,**k):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise ImportError('Training dependency forbidden: '+name)
        return old_import(name,*a,**k)
    def load(file,*a,**k):
        actual=Path(file).resolve();opened.append(str(actual))
        if actual!=path.resolve():raise AssertionError('Non-weight read during inference: '+str(actual))
        return old_load(file,*a,**k)
    builtins.__import__=imp;np.load=load
    try:result=mod.predict(path,c,**kwargs)
    finally:builtins.__import__=old_import;np.load=old_load
    check(path.parent.name+'/inference_only_own_weights',opened==[str(path.resolve())]);return result
