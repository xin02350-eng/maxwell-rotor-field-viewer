"""Finite global validation-selected blends; no per-point validation tuning.

The signed-vector and magnitude blend routes are different and labelled. A
mixed neural/kernel ensemble is never described as a pure neural network.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'new60_opt_v2'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def metrics(p,y):
    e=p-y
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(y)),
        mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()),
        per_case_rmse_T=np.sqrt(np.mean(e*e,axis=(1,2))).tolist())

def predict(model,conditions,frame_index=None):
    if isinstance(model,(str,Path)):
        p=Path(model)
        with np.load(p,allow_pickle=False) as z:m={k:z[k] for k in z.files}
        meta=json.loads(str(m['metadata_json']));root=p.parent
    else:raise TypeError('Use saved model path for bound ensemble source')
    fields=[]
    for i,part in enumerate(meta['components']):
        mp=(root/part['model_relative_path']).resolve();sp=(root/part['module_relative_path']).resolve()
        if digest(mp)!=part['model_sha256'] or digest(sp)!=part['module_sha256']:raise ValueError('Ensemble source/model SHA mismatch')
        md=module(sp,f'v3_ensemble_component_{i}')
        pfield=md.predict(mp,conditions,frame_index=frame_index)
        if part['quantity']=='Br,Bt' and meta['blend_quantity']=='magnitude':pfield=np.hypot(pfield[...,0::2],pfield[...,1::2])
        fields.append(pfield)
    out=(1-meta['alpha'])*fields[0]+meta['alpha']*fields[1]
    if meta['blend_quantity']=='Br,Bt':out=np.hypot(out[...,0::2],out[...,1::2])
    return out

def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    gp=V2/'kernel_route/rotor_cycle_refined/model.npz';gs=V2/'kernel_route/kernel_model.py'
    nn=V2/'neural_route/rotor/selected_model.npz';ns=V2/'neural_route/cycle_neural.py'
    components=[dict(model_relative_path=str(p.relative_to(HERE.parent)),module_relative_path=str(s.relative_to(HERE.parent)),
        model_sha256=digest(p),module_sha256=digest(s),quantity='Br,Bt') for p,s in [(gp,gs),(nn,ns)]]
    alphas=np.linspace(0,1,21).tolist()
    dump(out/'protocol.json',dict(components=components,alphas=alphas,routes=['Br,Bt','magnitude'],
        method='finite global mixed kernel/neural ensemble control',selection='minimum original-T raw validation full-field RMSE, no per-point mixing',
        source_sha256=digest(__file__),test_labels_opened=False))
    # These files explicitly contain validation9 only, already scientifically checked.
    with np.load(V2/'evaluation/prediction_rotor_vector/validation_predictions.npz',allow_pickle=False) as z:
        c=z['conditions'];y=np.hypot(z['truth'][...,0::2],z['truth'][...,1::2]);ids=z['case_ids']
    expected=[1,9,14,15,24,27,35,40,51]
    if ids.tolist()!=expected:raise ValueError('Validation identity mismatch')
    pf=module(gs,'v3_gp_control').predict(gp,c);pn=module(ns,'v3_nn_control').predict(nn,c)
    rows=[];best=np.inf
    for quantity in ('Br,Bt','magnitude'):
        f,n=(pf,pn) if quantity=='Br,Bt' else (np.hypot(pf[...,0::2],pf[...,1::2]),np.hypot(pn[...,0::2],pn[...,1::2]))
        for alpha in alphas:
            value=(1-alpha)*f+alpha*n
            if quantity=='Br,Bt':value=np.hypot(value[...,0::2],value[...,1::2])
            row=dict(blend_quantity=quantity,alpha=alpha,validation=metrics(value,y));rows.append(row)
            if row['validation']['rmse_T']<best:
                best=row['validation']['rmse_T'];parts=[]
                import os
                for p,s in [(gp,gs),(nn,ns)]:parts.append(dict(model_relative_path=os.path.relpath(p,out),module_relative_path=os.path.relpath(s,out),model_sha256=digest(p),module_sha256=digest(s),quantity='Br,Bt'))
                meta=dict(format='v3-global-mixed-ensemble',blend_quantity=quantity,alpha=alpha,components=parts,region='rotor',quantity='|B|',units='T',
                    method='mixed kernel/neural model, not pure NN',validation=row['validation'],test_labels_opened=False,source_sha256=digest(__file__))
                np.savez_compressed(out/'best_model.npz',metadata_json=np.array(json.dumps(meta)))
                actual=predict(out/'best_model.npz',c)
                if not np.allclose(actual,value,rtol=1e-10,atol=1e-10):raise AssertionError('Bound ensemble NumPy replay differs')
                np.savez_compressed(out/'best_validation_predictions.npz',prediction=actual,case_ids=ids)
                dump(out/'best.json',row)
    dump(out/'ranking.json',sorted(rows,key=lambda r:r['validation']['rmse_T']))
    dump(out/'summary.json',dict(completed=len(rows),best=min(rows,key=lambda r:r['validation']['rmse_T']),model_sha256=digest(out/'best_model.npz'),test_labels_opened=False))
    print(json.dumps(min(rows,key=lambda r:r['validation']['rmse_T'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(limits=2):run(a.output)
