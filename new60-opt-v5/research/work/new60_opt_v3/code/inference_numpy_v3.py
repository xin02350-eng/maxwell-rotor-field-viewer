"""SHA-bound numerical inference. Model/source modules need only NumPy.
Never opens scientific truth arrays or the label entries of source caches.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_freeze(root,name='both_versions_choices_frozen.json'):
    root=Path(root).resolve();p=root/name
    side=p.with_suffix(p.suffix+'.sha256')
    if not side.exists():side=p.with_suffix('.sha256')
    if digest(p)!=side.read_text().strip():raise ValueError('Frozen registry SHA mismatch')
    a=json.loads(p.read_text(encoding='utf8'))
    if digest(root/'protocol.json')!=a['protocol_sha256']:raise ValueError('Protocol SHA mismatch')
    if 'prediction_freeze_sha256' in a and digest(root/'prediction_choices_frozen.json')!=a['prediction_freeze_sha256']:raise ValueError('Prediction freeze SHA mismatch')
    for part in a.get('support_files',[]):
        if digest(root/part['relative_path'])!=part['sha256']:raise ValueError('Inference support SHA mismatch: '+part['relative_path'])
    return a

def predict(root,choice,conditions,frame_index=None):
    root=Path(root).resolve();c=np.asarray(conditions,dtype=float)
    if c.ndim not in (1,2) or c.shape[-1]!=2 or not np.isfinite(c).all() or np.any(c[...,0]<0):raise ValueError('Expected finite native [I_RMS>=0, Thet]')
    if frame_index is not None and (type(frame_index) is not int or not 0<=frame_index<60):raise ValueError('Expected stored frame_index 0..59')
    mp=(root/choice['model_relative_path']).resolve();sp=(root/choice['inference_module_relative_path']).resolve()
    if digest(mp)!=choice['model_sha256'] or digest(sp)!=choice['inference_module_sha256']:raise ValueError('Model/inference SHA mismatch')
    name='v3_bound_inference_'+choice['inference_module_sha256'][:16]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,sp);m=importlib.util.module_from_spec(spec)
        # Audited support modules may be adjacent to this hashed source.
        sys.path.insert(0,str(sp.parent))
        try:spec.loader.exec_module(m)
        finally:sys.path.pop(0)
        sys.modules[name]=m
    m=sys.modules[name]
    # Adjacent source imports can also be lazy inside predict functions.
    sys.path.insert(0,str(sp.parent))
    try:p=getattr(m,choice.get('inference_function','predict'))(mp,c,frame_index=frame_index,**choice.get('predict_kwargs',{}))
    finally:sys.path.pop(0)
    p=np.asarray(p,dtype=np.float64)
    if choice.get('output_conversion')=='hypot_Br_Bt':p=np.hypot(p[...,0::2],p[...,1::2])
    if not np.isfinite(p).all():raise ValueError('Nonfinite model output')
    return p
