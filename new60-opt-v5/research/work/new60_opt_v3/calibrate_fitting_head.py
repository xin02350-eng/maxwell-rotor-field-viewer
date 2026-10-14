"""Independent all60 training-fit branch: least-squares NN output-layer update.

The tanh hidden layers were genuinely trained in V2. This updates the linear
readout on all60 known conditions, then checks the real raw fields in T.
It is supervised training inside an existing MLP, not unseen-case prediction.
Run only after V3 prediction choices are frozen. No results feed that branch.
"""
from __future__ import annotations
import argparse, hashlib, json, time, sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
V2=HERE.parent/'new60_opt_v2'
V1=HERE.parent/'new60_20261014'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def features(c,scale):
    c=np.atleast_2d(np.asarray(c,dtype=float));r=c[:,0]/scale;a=np.deg2rad(c[:,1]);x=r*np.cos(a);y=r*np.sin(a)
    return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))

def predict(model,conditions,frame_index=None,nonnegative=True):
    if isinstance(model,(str,Path)):
        with np.load(model,allow_pickle=False) as z:model={k:z[k] for k in z.files}
    c=np.asarray(conditions,dtype=float);one=c.ndim==1
    x=(features(c,float(model['current_scale']))-model['feature_mean'])/model['feature_scale']
    for i in range(int(model['layer_count'])):
        x=x@model[f'w{i}']+model[f'b{i}']
        if i<int(model['layer_count'])-1:x=np.tanh(x)
    p=(model['cycle_mean']+(x*model['output_scale'])@model['cycle_basis']).reshape(len(x),60,-1)
    if nonnegative and str(model['quantity'])=='|B|':p=np.maximum(p,0.)
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<60:raise ValueError('Expected fixed frame_index 0..59')
        p=p[:,frame_index]
    return p[0] if one else p

def metrics(p,y):
    e=p-y
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),
                relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(y)))

def run(region,prediction_freeze,output):
    start=time.perf_counter();freeze=Path(prediction_freeze)
    f=json.loads(freeze.read_text(encoding='utf-8-sig'))
    if f.get('test_labels_used_for_selection') is not False or not f.get('models'):raise ValueError('V3 prediction freeze missing')
    freeze_sha=digest(freeze)
    if freeze_sha!=freeze.with_suffix(freeze.suffix+'.sha256').read_text().strip():raise ValueError('Prediction freeze SHA mismatch')
    for item in f['models']:
        for path_key,hash_key in [('model_relative_path','model_sha256'),('inference_module_relative_path','inference_module_sha256')]:
            if digest((freeze.parent/item[path_key]).resolve())!=item[hash_key]:raise ValueError(f'Frozen prediction binding mismatch: {item[path_key]}')
    for item in f.get('support_files',[]):
        if digest((freeze.parent/item['relative_path']).resolve())!=item['sha256']:raise ValueError('Frozen supporting source binding mismatch')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=V2/'neural_route/fitting_full60'/region/'model.npz'
    if digest(source)!=source.with_suffix(source.suffix+'.sha256').read_text().strip():raise ValueError('Base fitting model SHA mismatch')
    with np.load(source,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    cache=V1/'cache_v1'/f'{region}.npz'
    old_meta=json.loads(str(m['metadata_json']))
    if digest(cache)!=old_meta['cache_sha256']:raise ValueError('Source cache SHA mismatch')
    with np.load(cache,allow_pickle=False) as z:
        c=z['conditions'];ids=z['case_ids'];cache_meta=json.loads(str(z['metadata_json']));field=z['field'].astype(float)
    if cache_meta['units']!='T' or cache_meta['xy_units']!='mm' or len(set(ids.tolist()))!=60 or field.shape[:2]!=(60,60):raise ValueError('Cache identity/unit/shape mismatch')
    if ids.tolist()!=old_meta['training_case_ids'] or not np.isfinite(c).all() or not np.isfinite(field).all():raise ValueError('Cache case identity/finiteness mismatch')
    if str(m['quantity'])=='|B|':field=np.hypot(field[...,0::2],field[...,1::2])
    flat=field.reshape(60,-1);target=(flat-m['cycle_mean'])@m['cycle_basis'].T
    old_raw=predict(m,c,nonnegative=False)
    x=(features(c,float(m['current_scale']))-m['feature_mean'])/m['feature_scale']
    for i in range(int(m['layer_count'])-1):x=np.tanh(x@m[f'w{i}']+m[f'b{i}'])
    design=np.column_stack([x,np.ones(60)])
    # All fitting targets enter only the linear NN layer's supervised objective.
    head,residual,rank,singular=np.linalg.lstsq(design,target/m['output_scale'],rcond=1e-12)
    if rank!=60:raise ValueError(f'Hidden response matrix rank {rank}; no full60 readout calibration claim')
    last=int(m['layer_count'])-1;m[f'w{last}']=head[:-1];m[f'b{last}']=head[-1]
    metadata=json.loads(str(m['metadata_json']))
    metadata.update(version='new60-full60-mlp-head-calibration-v3',branch='fitting_full60',
        method='Previously gradient-trained tanh MLP; supervised least-squares optimization of the linear output layer on all60 conditions',
        base_model_sha256=digest(source),prediction_freeze_sha256=freeze_sha,
        all60_training=True,heldout_accuracy_claim=False,source_sha256=digest(__file__),
        output_layer_training_rank=int(rank),readout_rcond=1e-12,units='T')
    m['metadata_json']=np.array(json.dumps(metadata,ensure_ascii=False))
    raw=predict(m,c,nonnegative=False);deployed=np.maximum(raw,0) if str(m['quantity'])=='|B|' else raw
    if not np.isfinite(raw).all():raise AssertionError('Nonfinite calibrated field')
    oracle=m['cycle_mean']+target@m['cycle_basis']
    result=dict(branch='fitting_full60',region=region,quantity=str(m['quantity']),training_case_ids=ids.tolist(),
        previous_raw=metrics(old_raw,field),fitting_raw=metrics(raw,field),fitting_deployed=metrics(deployed,field),
        raw_negative_count=int((raw<0).sum()) if str(m['quantity'])=='|B|' else None,
        hidden_design_rank=int(rank),hidden_design_condition_number=float(singular[0]/singular[-1]),
        max_latent_training_error=float(np.abs((design@head)*m['output_scale']-target).max()),
        representation_oracle=metrics(oracle,flat),elapsed_seconds=time.perf_counter()-start,
        pure_MLP_forward=True,gradient_hidden_layers_from_V2=True,output_layer_closed_form_training=True,
        prediction_freeze_sha256=freeze_sha,test_labels_used_as_fitting_only=True,unseen_prediction_claim=False)
    np.savez_compressed(out/'model.npz',**m)
    (out/'model.npz.sha256').write_text(digest(out/'model.npz')+'\n',encoding='ascii')
    result['model_sha256']=digest(out/'model.npz');result['source_sha256']=digest(__file__)
    if digest(freeze)!=freeze_sha:raise AssertionError('Prediction freeze changed')
    dump(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--region',required=True,choices=['rotor','stator','airgap'])
    p.add_argument('--prediction-freeze',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(limits=2):run(a.region,a.prediction_freeze,a.output)
