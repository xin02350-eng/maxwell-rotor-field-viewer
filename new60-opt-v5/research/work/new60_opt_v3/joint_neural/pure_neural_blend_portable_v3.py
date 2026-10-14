"""NumPy inference for a global two-model pure-neural convex blend.

The fixed old neural cycle and new independent neural mean experts are combined
with one global alpha. No per-point gate and no test-selected mixing weights.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if sha(path)!=path.with_suffix(path.suffix+".sha256").read_text().strip():raise ValueError("Blend model SHA mismatch")
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def point_module(m):
    p=Path(__file__).with_name("point_mean_lowcap_nn.py")
    if sha(p)!=str(m["point_module_sha256"]):raise ValueError("Point neural inference source SHA mismatch")
    helper=Path(__file__).with_name("joint_spatial_nn.py")
    if sha(helper)!=str(m["joint_module_sha256"]):raise ValueError("Train-only POD helper inference source SHA mismatch")
    # The helper imports NumPy/stdlib only. Load by checked sibling path so a
    # relocated CLI does not depend on the workspace's Python search path.
    hs=importlib.util.spec_from_file_location("v3_frozen_joint_helper",helper);hm=importlib.util.module_from_spec(hs);hs.loader.exec_module(hm)
    sentinel=object();prior=sys.modules.get("joint_spatial_nn",sentinel)
    try:
        sys.modules["joint_spatial_nn"]=hm
        spec=importlib.util.spec_from_file_location("v3_frozen_point_mean_neural",p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        if prior is sentinel:sys.modules.pop("joint_spatial_nn",None)
        else:sys.modules["joint_spatial_nn"]=prior
    return module

def signed_pair(model,conditions):
    m=load_model(model);module=point_module(m);c=np.atleast_2d(np.asarray(conditions,dtype=np.float64))
    if str(m["target_kind"])!="signed":raise ValueError("This blend expects a signed mean neural model")
    mean=module.predict_mean(m,c);base={k[9:]:v for k,v in m.items() if k.startswith("baseline_")};old=module.baseline_predict(base,c)
    vector=old.reshape(len(c),int(m["frame_count"]),int(m["point_count"]),2)
    new=(mean[:,None,:,:]+float(m["dynamic_scale"])*(vector-vector.mean(1,keepdims=True))).reshape(old.shape)
    return old,new

def predict(model,conditions,frame_index=None,nonnegative=False,return_signed=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c);old,new=signed_pair(m,c)
    alpha=float(m["blend_alpha"]);route=str(m["blend_route"])
    if not 0<=alpha<=1:raise ValueError("Global alpha out of range")
    if route=="signed_vector":
        vec=(1-alpha)*old+alpha*new
        out=vec if return_signed else np.hypot(vec[...,0::2],vec[...,1::2])
    elif route=="scalar_magnitude":
        if return_signed:raise ValueError("Magnitude blend does not imply unique signed components")
        out=(1-alpha)*np.hypot(old[...,0::2],old[...,1::2])+alpha*np.hypot(new[...,0::2],new[...,1::2])
    else:raise ValueError("Unknown neural blending route")
    if nonnegative and not return_signed:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m["frame_count"]):raise ValueError("Invalid saved frame index")
        out=out[:,frame_index]
    return out[0] if single else out
