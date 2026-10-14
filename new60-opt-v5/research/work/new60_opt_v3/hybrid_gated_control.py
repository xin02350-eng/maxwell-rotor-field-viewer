"""Last finite, global five-weight mixed control; validation only before freeze."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'new60_opt_v2'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load_source(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(p.parent))
    try:spec.loader.exec_module(m)
    finally:sys.path.pop(0)
    return m
def metrics(p,y):
    e=p-y
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(y)),
        mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()),
        per_case_rmse_T=np.sqrt(np.mean(e*e,axis=(1,2))).tolist())
def predict(model,conditions,frame_index=None):
    p=Path(model)
    with np.load(p,allow_pickle=False) as z:meta=json.loads(str(z['metadata_json']))
    fields=[]
    for i,b in enumerate(meta['components']):
        mp=(p.parent/b['model_relative_path']).resolve();sp=(p.parent/b['module_relative_path']).resolve()
        if digest(mp)!=b['model_sha256'] or digest(sp)!=b['module_sha256']:raise ValueError('Bound component SHA mismatch')
        m=load_source(sp,f'v3_gated_control_component_{i}');v=m.predict(mp,conditions,frame_index=frame_index)
        if b['quantity']=='Br,Bt':v=np.hypot(v[...,0::2],v[...,1::2])
        fields.append(v)
    return (1-meta['alpha'])*fields[0]+meta['alpha']*fields[1]
def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    parts=[(V2/'kernel_route/rotor_cycle_refined/model.npz',V2/'kernel_route/kernel_model.py','Br,Bt'),
           (HERE/'gated_neural_ensemble/selected_model.npz',HERE/'gated_neural_ensemble.py','|B|')]
    bindings=[dict(model_relative_path=os.path.relpath(mp,out).replace('\\','/'),module_relative_path=os.path.relpath(sp,out).replace('\\','/'),
        model_sha256=digest(mp),module_sha256=digest(sp),quantity=q) for mp,sp,q in parts]
    alphas=[0.,.25,.5,.75,1.]
    dump(out/'protocol.json',dict(alphas=alphas,source_sha256=digest(__file__),components=bindings,
        selection='Original-T complete validation9 RMSE, fixed common convex weight, no per-point validation mixing',test_labels_opened=False))
    with np.load(V2/'evaluation/prediction_rotor_magnitude/validation_predictions.npz',allow_pickle=False) as z:c=z['conditions'];truth=z['truth'];ids=z['case_ids']
    if ids.tolist()!=[1,9,14,15,24,27,35,40,51]:raise ValueError('Validation case IDs mismatch')
    fields=[]
    for i,(mp,sp,q) in enumerate(parts):
        v=load_source(sp,f'v3_hybrid_develop_{i}').predict(mp,c)
        fields.append(np.hypot(v[...,0::2],v[...,1::2]) if q=='Br,Bt' else v)
    rows=[];best=float('inf')
    for alpha in alphas:
        value=(1-alpha)*fields[0]+alpha*fields[1];row=dict(alpha=alpha,validation=metrics(value,truth));rows.append(row)
        if row['validation']['rmse_T']<best:
            best=row['validation']['rmse_T']
            meta=dict(format='v3-global-kernel-neural-control',alpha=alpha,components=bindings,method='mixed kernel and genuine neural models, not pure NN',source_sha256=digest(__file__),test_labels_opened=False)
            np.savez_compressed(out/'selected_model.npz',metadata_json=np.array(json.dumps(meta)))
            np.savez_compressed(out/'validation_predictions.npz',prediction=value,case_ids=ids)
            dump(out/'best.json',row)
    replay=predict(out/'selected_model.npz',c)
    with np.load(out/'validation_predictions.npz',allow_pickle=False) as z:
        if not np.allclose(replay,z['prediction'],rtol=1e-10,atol=1e-10):raise AssertionError('Mixed model NumPy replay differs')
    dump(out/'results.json',dict(completed=5,rows=rows,best=min(rows,key=lambda r:r['validation']['rmse_T']),model_sha256=digest(out/'selected_model.npz'),test_labels_opened=False))
    print(json.dumps(min(rows,key=lambda r:r['validation']['rmse_T'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(limits=2):run(a.output)
