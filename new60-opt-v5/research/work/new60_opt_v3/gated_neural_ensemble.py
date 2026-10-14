"""25 finite pure-neural output combinations on train-only fixed risk masks.

NumPy inference is self-contained in this file and saved model weights. The
fixed point mask is not learned from validation labels; one global operator
weight is chosen by full validation RMSE. No new neural network training here.
"""
from pathlib import Path
import argparse,hashlib,json,time,zipfile,importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parent;V1=ROOT.parent/'new60_20261014'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def features(c,kind,current_scale):
    r=c[:,0]/current_scale;a=np.deg2rad(c[:,1])
    if kind=='native':return np.column_stack((r,c[:,1]/90))
    if kind.startswith('sat'):
        cap=float(kind[3:]);r=cap/current_scale*np.tanh(c[:,0]/cap)
    x=r*np.cos(a);y=r*np.sin(a)
    if kind=='cartesian':return np.column_stack((x,y))
    if kind=='mixed':return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))
    return np.column_stack((x,y,x*x,x*y,y*y))


def neural_cycle(m,c):
    scale=float(m['current_scale']);raw=features(c,str(m['feature_kind']),scale)
    x=(raw-m['feature_mean'])/m['feature_scale']
    for i in range(int(m['layer_count'])):
        x=x@m[f'w{i}']+m[f'b{i}']
        if i+1<int(m['layer_count']):x=np.tanh(x) if str(m['activation'])=='tanh' else x/(1+np.exp(-np.clip(x,-700,700)))
    base=features(c,'cartesian',scale);degree=int(m['base_degree'])
    poly=np.column_stack([base[:,0]**i*base[:,1]**(total-i) for total in range(degree+1) for i in range(total+1)])
    latent=x*m['output_scale']+poly@m['base_weights']
    coefficient=(m['cycle_mean']+latent@m['cycle_basis']).reshape(len(c),int(m['frame_count']),-1)
    return m['spatial_mean']+coefficient@m['spatial_basis'].T


def independent_mean(m,c):
    x=(features(c,str(m['feature_kind']),float(m['current_scale']))-m['feature_mean'])/m['feature_scale']
    h=np.broadcast_to(x,(int(m['point_count']),*x.shape))
    for i in range(int(m['layer_count'])):
        h=np.einsum('pcf,pfh->pch',h,m[f'w{i}'])+m[f'b{i}'][:,None,:]
        if i+1<int(m['layer_count']):h=np.tanh(h)
    if bool(m['has_skip']):h+=np.einsum('cf,pfh->pch',x,m['skip_weight'])+m['skip_bias'][:,None,:]
    return (h*m['output_scale'][:,None,:]+m['output_mean'][:,None,:]).transpose(1,0,2)


def operator_mean(m,c):
    x=((features(c,'mixed',float(m['current_scale']))-m['condition_center'])/m['condition_scale']).astype(np.float32)
    attrs=np.concatenate((m['point_embedding'],m['point_attributes']),axis=1)
    h=np.concatenate((np.broadcast_to(x[:,None,:],(len(c),len(attrs),x.shape[-1])),np.broadcast_to(attrs[None,:,:],(len(c),*attrs.shape))),axis=-1).reshape(-1,x.shape[-1]+attrs.shape[-1])
    for i in range(int(m['layer_count'])):
        h=h@m[f'w{i}']+m[f'b{i}']
        if i+1<int(m['layer_count']):h=np.tanh(h)
    return (m['zero_current_baseline'][None,:,:]+h.reshape(len(c),len(attrs),2)*m['point_response_scale'][None,:,:]).astype(np.float64)


def branches(m,c):
    a={k[2:]:v for k,v in m.items() if k.startswith('a_')};b={k[2:]:v for k,v in m.items() if k.startswith('b_')}
    old=neural_cycle({k[9:]:v for k,v in a.items() if k.startswith('baseline_')},c)
    oldvec=old.reshape(len(c),int(a['frame_count']),int(a['point_count']),2)
    fluctuation=oldvec-oldvec.mean(1,keepdims=True)
    newvec=independent_mean(a,c)[:,None,:,:]+float(a['dynamic_scale'])*fluctuation
    aa=float(a['blend_alpha'])
    if str(a['blend_route'])=='scalar_magnitude':first=(1-aa)*np.hypot(oldvec[...,0],oldvec[...,1])+aa*np.hypot(newvec[...,0],newvec[...,1])
    else:
        mixed=(1-aa)*oldvec+aa*newvec;first=np.hypot(mixed[...,0],mixed[...,1])
    opvec=operator_mean(b,c)[:,None,:,:]+fluctuation
    second=np.hypot(opvec[...,0],opvec[...,1])
    return first,second


def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if digest(path)!=path.with_suffix('.npz.sha256').read_text().strip():raise ValueError('Gated neural model hash mismatch')
    with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    if str(m['format'])!='v3_trainrisk_gated_neural_combination':raise ValueError('Model format mismatch')
    return m


def predict(model,conditions,frame_index=None):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    if c.ndim!=2 or c.shape[-1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Finite native RMS-current>=0 and Thet pairs required')
    one,two=branches(m,c);weights=m['operator_weight_by_point'][None,None,:]
    if not np.isfinite(weights).all() or np.any(weights<0) or np.any(weights>1):raise ValueError('Convex point weights invalid')
    output=(1-weights)*one+weights*two
    if not np.isfinite(output).all():raise ValueError('Nonfinite prediction')
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<60:raise ValueError('Original supplied frame index0..59 required')
        output=output[:,frame_index]
    return output[0] if single else output


def metrics(pred,truth,ids):
    e=pred-truth;sse=np.sum(e*e,axis=(1,2));truthenergy=np.sum(truth*truth,axis=(1,2));rms=np.sqrt(sse/np.prod(truth.shape[1:]));w=int(np.argmax(rms));loc=tuple(map(int,np.unravel_index(np.argmax(np.abs(e)),e.shape)))
    return dict(relative_L2_pct=float(100*np.sqrt(sse.sum()/truthenergy.sum())),rmse_T=float(np.sqrt(sse.sum()/e.size)),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        worst_case_id=int(ids[w]),worst_case_rmse_T=float(rms[w]),negative_count=int(np.count_nonzero(pred<0)),
        max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=loc[1],point_index=loc[2],prediction_T=float(pred[loc]),truth_T=float(truth[loc])),
        per_case={str(cid):dict(rmse_T=float(rms[i]),relative_L2_pct=float(100*np.sqrt(sse[i]/truthenergy[i]))) for i,cid in enumerate(ids)})


def permitted_rows(path,indices):
    mapping={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as z,z.open('field.npy') as f:
        v=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,v)
        if shape!=(60,60,3600) or fortran:raise ValueError('Unexpected case-major rotor field')
        out=np.empty((len(indices),*shape[1:]),dtype=dtype);size=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(60):
            if i in mapping:out[mapping[i]]=np.frombuffer(f.read(size),dtype=dtype).reshape(shape[1:])
            else:f.seek(size,1)
    return out.astype(np.float64)


def run():
    out=ROOT/'gated_neural_ensemble';out.mkdir(exist_ok=False);t0=time.perf_counter()
    a_path=ROOT/'joint_neural/pure_neural_blend/portable_model.npz';b_path=ROOT/'point_operator_nn/pilot_02/model.npz';risk_path=ROOT/'static_local_control/training_loo_diagnostics.npz'
    sourcepaths=[a_path,b_path,risk_path,ROOT/'joint_neural/pure_neural_blend_portable_v3.py',ROOT/'joint_neural/point_mean_lowcap_nn.py',ROOT/'joint_neural/joint_spatial_nn.py',ROOT/'point_operator_nn.py',ROOT.parent/'new60_opt_v2/neural_route/rotor/selected_model.npz']
    dependencies={str(p.relative_to(ROOT.parent)):digest(p) for p in sourcepaths}
    with np.load(a_path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(b_path,allow_pickle=False) as z:b={k:z[k] for k in z.files}
    with np.load(risk_path,allow_pickle=False) as z:rp=z['shared_best_point_loo_predictions'];rt=z['train_mean_fields'];pid=z['point_id'];riskids=z['train_case_ids']
    if rp.shape!=(42,1800) or rt.shape!=rp.shape:raise ValueError('Train42 LOO risk shape mismatch')
    split=json.loads((V1/'cache_v1/split.json').read_text('utf8'));va=split['validation']
    if riskids.tolist()!=split['train'] or not np.array_equal(a['point_id'],pid) or not np.array_equal(b['point_id'],pid):raise ValueError('Risk/model point order and train IDs mismatch')
    scores=np.mean((rp-rt)**2,axis=0);order=np.argsort(-scores,kind='stable')
    configs=[dict(mask_point_count=k,operator_alpha=alpha) for k in (18,90,180,600,1800) for alpha in (0.,.25,.5,.75,1.)]
    protocol=dict(candidate_count=len(configs),configs=configs,training_case_ids=split['train'],validation_case_ids=va,excluded_test_case_ids=split['test'],
        risk_definition='mean((train42 shared_best_point_LOO_prediction - train_mean_field)^2,axis=condition); stable descending ordering; all data from training LOO artifact',
        risk_selection_limit='The old training LOO selected one shared static kernel from360 finite candidates, reused here for risk construction; not unbiased model performance',
        selection='Minimum complete raw original-T validation RMSE over all9 conditions60frames1800points; one common alpha on fixed train-risk mask, no per-val-point tuning',
        composition='Two pureNN output branches, convex scalar magnitude mix; no new NN training',new_neural_training_runs=0,
        test_labels_materialized=False,full60_fitting_used=False,source_sha256=digest(__file__),dependencies=dependencies)
    dump(out/'candidate_protocol.json',protocol)
    mask_payload={str(k):dict(point_indices=order[:k].tolist(),point_ids=pid[order[:k]].tolist()) for k in (18,90,180,600,1800)}
    dump(out/'train_risk_masks_frozen.json',dict(risk_source_sha256=digest(risk_path),train_case_ids=riskids.tolist(),score=scores.tolist(),masks=mask_payload,test_labels_used=False))
    with np.load(V1/'cache_v1/rotor.npz',allow_pickle=False) as z:c=z['conditions'][va];ids=z['case_ids'][va]
    signed=permitted_rows(V1/'cache_v1/rotor.npz',va);truth=np.hypot(signed[...,0::2],signed[...,1::2]);del signed
    m={**{'a_'+k:v for k,v in a.items()},**{'b_'+k:v for k,v in b.items()},'format':np.array('v3_trainrisk_gated_neural_combination'),'point_id':pid,'risk_training_case_ids':riskids,
       'training_risk_score':scores,'dependencies_json':np.array(json.dumps(dependencies))}
    first,second=branches(m,c)
    # Self-contained equations must reproduce both immutable NN branch fields.
    with np.load(ROOT/'joint_neural/pure_neural_blend/validation_predictions.npz',allow_pickle=False) as z:expecteda=z['prediction']
    with np.load(ROOT/'point_operator_nn/pilot_02/validation_predictions.npz',allow_pickle=False) as z:expectedb=z['prediction']
    paritya=float(np.max(np.abs(first-expecteda)));parityb=float(np.max(np.abs(second-expectedb)))
    if max(paritya,parityb)>1e-9:raise AssertionError('Selfcontained NN replay mismatch '+str((paritya,parityb)))
    rows=[];best=np.inf;bestrow=None;bestweights=None
    for config in configs:
        weights=np.zeros(1800);weights[order[:config['mask_point_count']]]=config['operator_alpha'];pred=(1-weights[None,None,:])*first+weights[None,None,:]*second
        row=dict(configuration=config,validation=metrics(pred,truth,ids));rows.append(row)
        if row['validation']['rmse_T']<best:best=row['validation']['rmse_T'];bestrow=row;bestweights=weights
    baseline=metrics(first,truth,ids);improvement=100*(1-best/baseline['rmse_T']);m['operator_weight_by_point']=bestweights;m['selection_json']=np.array(json.dumps(bestrow))
    np.savez_compressed(out/'selected_model.npz',**m);(out/'selected_model.npz.sha256').write_text(digest(out/'selected_model.npz')+'\n',encoding='ascii')
    replay=predict(out/'selected_model.npz',c);met=metrics(replay,truth,ids)
    if not np.isclose(met['rmse_T'],best,atol=1e-12):raise AssertionError('Selected model replay mismatch')
    np.savez_compressed(out/'validation_predictions.npz',prediction=replay,case_ids=ids,point_id=pid)
    pareto=[]
    for row in rows:
        v=row['validation']
        if not any(all(other['validation'][key]<=v[key] for key in ('rmse_T','max_abs_T','worst_case_rmse_T')) and any(other['validation'][key]<v[key] for key in ('rmse_T','max_abs_T','worst_case_rmse_T')) for other in rows):pareto.append(row)
    dump(out/'selection_validation_only.json',dict(selected=bestrow,baseline=baseline,operator=metrics(second,truth,ids),relative_RMSE_improvement_pct=improvement,
        rows=rows,pareto=pareto,new_neural_training_runs=0,completed_output_combinations=len(rows),test_labels_materialized=False,full60_fitting_used=False,
        selfcontained_branch_max_difference_T=dict(joint_pureNN=paritya,operator=parityb),model_sha256=digest(out/'selected_model.npz'),
        source_sha256=digest(__file__),elapsed_seconds=time.perf_counter()-t0))
    print(json.dumps(dict(selected=bestrow,RMSE_improvement_pct=improvement,branch_parity_T=(paritya,parityb))),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run()
