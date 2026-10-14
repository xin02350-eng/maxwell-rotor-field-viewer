"""Finite train-only LOO point-local mean-field experiment; test rows skipped."""
from pathlib import Path
import csv,hashlib,json,os,sys,time
from datetime import datetime,timezone
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.spatial import cKDTree
from static_local_control import weights,predict,digest

HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
V2=WORK/'new60_opt_v2';V1=WORK/'new60_20261014'
sys.path.insert(0,str(V2/'neural_route'))
from cycle_neural import selected_rows

def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def metrics(p,t):
    e=p-t;sse=np.sum(e*e,axis=tuple(range(1,e.ndim)));den=np.sum(t*t,axis=tuple(range(1,t.ndim)))
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(t)),
        mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        per_case_rmse_T=np.sqrt(sse/np.prod(e.shape[1:])).tolist(),per_case_L2_pct=(100*np.sqrt(sse/den)).tolist())

def main():
    threadpool_limits(2)
    split=json.loads((V1/'cache_v1/split.json').read_text('utf-8'))
    cache=V1/'cache_v1/rotor.npz'
    with np.load(cache,allow_pickle=False) as z:conditions=z['conditions'];ids=z['point_id'];xy=z['xy'];geo=z['dataset_hotspot']
    train=selected_rows(cache,'field',split['train']).astype(np.float64);train=np.hypot(train[...,0::2],train[...,1::2]);target=train.mean(1)
    trc=conditions[split['train']];vac=conditions[split['validation']]
    truth=selected_rows(cache,'field',split['validation']).astype(np.float64);truth=np.hypot(truth[...,0::2],truth[...,1::2]);truthmean=truth.mean(1)
    specs=[dict(coordinate=coord,kernel=kind,length=length,ridge=ridge,degree=degree)
        for coord in ['cartesian','sat05','sat1','sat2','native']
        for kind in ['gaussian','imq','matern32'] for length in [.4,.8,1.6,3.2]
        for ridge in [.0001,.01,.1] for degree in [0,1]]
    # 360 fixed common candidates; fewer effective spatial selectors, no val per-point fitting.
    protocol=dict(version='static-local-control-v3.1',frozen_utc=datetime.now(timezone.utc).isoformat(),
        train_case_ids=split['train'],validation_case_ids=split['validation'],test_labels_loaded=False,
        candidate_count=len(specs),candidate_specs=specs,cache_sha256=digest(cache),
        objective='Improve spatial-local time-mean response. Analytic static kernel plus frozen pureNN zero-mean temporal dynamics.',
        loo='Each one-case-out fold refits native current scale, embedding center and input scale on its remaining41 conditions. Output target is raw point mean, not a globally fitted target basis.',
        selections=['global_shared','geometry2_shared','roughness4_shared','loo_residual4_shared','point_shrink005','point_shrink020','point_shrink100'],
        grouping='geometry2 from supplied fixed geometry; roughness quartiles from train42 nearest5 mean-square neighbor jump; loo quartiles from train-only selected shared kernel LOO point MSE.',
        point_shrink='Per point use 42 training OOF static MSE plus eta times (candidate global MSE / minimum global MSE - 1)*globalmeanpointMSE; eta is fixed at .05/.20/1.0. No per-point validation parameter selection.',
        caution='Training LOO is reused for hyperparameter/group construction so is a tuning criterion, not unbiased external model performance. Validation results remain exploratory after prior V2 tuning.',
        dynamic='Frozen V2 pureNN rotor vector -> hypot -> subtract its own per-condition60-frame mean; never full60 fit.',
        test='No test prediction is computed by this runner; parent freezes choices before reuse.',budget_bytes=150_000_000)
    write(HERE/'protocol.json',protocol)
    n=len(trc);loo_pred=np.empty((len(specs),n,1800));val_mean=np.empty((len(specs),len(vac),1800));full_train=np.empty((len(specs),n,1800));scores=[]
    start=time.time()
    for si,spec in enumerate(specs):
        loo=np.zeros((n,n))
        for held in range(n):
            mask=np.arange(n)!=held
            loo[held,mask]=weights(trc[mask],trc[held:held+1],spec)[0]
        loo_pred[si]=loo@target
        val_mean[si]=weights(trc,vac,spec)@target
        full_train[si]=weights(trc,trc,spec)@target
        mse=np.mean((loo_pred[si]-target)**2,axis=0)
        scores.append(dict(candidate_index=si,**spec,train_loo_static_rmse_T=float(np.sqrt(mse.mean())),
            validation_static_rmse_T=float(np.sqrt(np.mean((val_mean[si]-truthmean)**2)))))
        if si%60==0:print(f'Completed {si+1}/{len(specs)} common static candidates; {time.time()-start:.1f}s',flush=True)
    loo_mse=np.mean((loo_pred-target[None])**2,axis=1);global_mse=loo_mse.mean(1);global_best=int(np.argmin(global_mse))
    _,near=cKDTree(xy).query(xy,k=6)
    roughness=np.mean((train[:,:,near[:,1:]]-train[:,:,:,None])**2,axis=(0,1,3))
    point_error=loo_mse[global_best]
    def quantile_group(x):return np.searchsorted(np.quantile(x,[.25,.5,.75]),x,side='right')
    groupings=dict(global_shared=np.zeros(1800,dtype=int),geometry2_shared=geo.astype(int),
        roughness4_shared=quantile_group(roughness),loo_residual4_shared=quantile_group(point_error))
    selectors={}
    for name,groups in groupings.items():
        selection=np.zeros(1800,dtype=int)
        for g in np.unique(groups):selection[groups==g]=int(np.argmin(loo_mse[:,groups==g].sum(1)))
        selectors[name]=selection
    for eta,name in [(.05,'point_shrink005'),(.20,'point_shrink020'),(1.0,'point_shrink100')]:
        penalty=eta*(global_mse/global_mse[global_best]-1)*global_mse[global_best]
        selectors[name]=np.argmin(loo_mse+penalty[:,None],axis=0)
    # Freeze all train-derived selectors before inspecting their final val full-field scores.
    selector_payload={name:indices.tolist() for name,indices in selectors.items()}
    write(HERE/'train_loo_selectors_frozen.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),test_loaded=False,
        global_best_spec_index=global_best,selectors=selector_payload,protocol_sha256=digest(HERE/'protocol.json')))
    nnp=V2/'neural_route/rotor/selected_model.npz';nnmodule=V2/'neural_route/cycle_neural.py'
    from cycle_neural import predict as nn_predict
    dyn_train=nn_predict(nnp,trc);dyn_train=np.hypot(dyn_train[...,0::2],dyn_train[...,1::2]);dyn_train-=dyn_train.mean(1,keepdims=True)
    dyn_val=nn_predict(nnp,vac);dyn_val=np.hypot(dyn_val[...,0::2],dyn_val[...,1::2]);dyn_val-=dyn_val.mean(1,keepdims=True)
    runs=[]
    for name,index in selectors.items():
        meanv=val_mean[index,:,np.arange(1800)].T;meant=full_train[index,:,np.arange(1800)].T
        val=meanv[:,None,:]+dyn_val;predtr=meant[:,None,:]+dyn_train
        raw=metrics(val,truth);deployed=metrics(np.maximum(val,0),truth)
        meta=dict(format='new60-static-local-control-v3',name=name,specifications=specs,
            dynamic_model_relative_path=os.path.relpath(nnp,HERE).replace('\\','/'),dynamic_model_sha256=digest(nnp),
            dynamic_module_relative_path=os.path.relpath(nnmodule,HERE).replace('\\','/'),dynamic_module_sha256=digest(nnmodule),
            train_case_ids=split['train'],validation_case_ids=split['validation'],test_labels_loaded=False,
            selector_freeze_sha256=digest(HERE/'train_loo_selectors_frozen.json'),cache_sha256=digest(cache))
        modelpath=HERE/f'{name}_model.npz'
        if modelpath.exists():raise FileExistsError(modelpath)
        np.savez_compressed(modelpath,metadata_json=np.asarray(json.dumps(meta,ensure_ascii=False)),train_conditions=trc,
            train_mean_fields=target,spec_id_for_point=index,point_id=ids,xy=xy)
        replay=predict(modelpath,vac)
        assert np.max(np.abs(replay-val))<1e-12
        result=dict(name=name,method='Analytic local static kernel plus frozen pureNN dynamic; not pureNN',
            validation_raw=raw,validation_nonnegative=deployed,train_in_sample=metrics(predtr,train),
            train_loo_static_tuning_rmse_T=float(np.sqrt(np.mean(loo_mse[index,np.arange(1800)]))),
            validation_static=metrics(meanv,truthmean),unique_static_specs=int(len(np.unique(index))),
            negative_raw_count=int(np.count_nonzero(val<0)),model_sha256=digest(modelpath),numpy_replay_max_abs_T=float(np.max(np.abs(replay-val))))
        runs.append(result)
        np.savez_compressed(HERE/f'{name}_validation_predictions.npz',prediction=val,truth=truth,case_ids=np.asarray(split['validation']),
            point_id=ids,xy=xy,quantity=np.asarray('planar_magnitude'),metadata_json=np.asarray(json.dumps(result,ensure_ascii=False)))
    concentration={}
    for score_name,values in [('train_neighbor_jump',roughness),('train_LOO_point_MSE',point_error)]:
        order=np.argsort(values)[::-1];parts={}
        for part in ['validation']:
            # Only validation residuals of the existing frozen V2 baseline.
            with np.load(V2/f'evaluation/prediction_rotor_magnitude/{part}_predictions.npz',allow_pickle=False) as z:ps=np.sum((z['prediction']-z['truth'])**2,axis=(0,1))
            parts[part]={str(k):float(100*ps[order[:k]].sum()/ps.sum()) for k in [18,90,180,600]}
        concentration[score_name]=dict(v2_SSE_coverage_pct=parts,top180_point_ids=ids[order[:180]].tolist())
    with (HERE/'candidate_metrics.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    np.savez_compressed(HERE/'training_loo_diagnostics.npz',train_case_ids=np.asarray(split['train']),point_id=ids,
        point_mse_all_candidates=loo_mse,shared_best_point_loo_predictions=loo_pred[global_best],train_mean_fields=target,
        roughness=roughness,global_candidate_mse=global_mse)
    report=dict(status='PASS',elapsed_seconds=time.time()-start,candidate_count=len(specs),test_model_scored=False,
        candidate_selection='train42 LOO only; seven predeclared selectors. Validation does not choose a point hyperparameter.',
        completed=runs,point_concentration=concentration)
    write(HERE/'results.json',report)
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    total=sum(p.stat().st_size for p in files)
    if total>150_000_000:raise ValueError('Storage budget exceeded')
    write(HERE/'manifest.json',dict(bytes_total=total,files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}))
    print(json.dumps(dict(status='PASS',bytes_total=total,runs=[dict(name=r['name'],validation_L2_pct=r['validation_raw']['relative_L2_pct'],validation_max_abs_T=r['validation_raw']['max_abs_T']) for r in runs]),ensure_ascii=False))

if __name__=='__main__':main()
