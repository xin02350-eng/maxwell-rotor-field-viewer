"""38 unsigned time-mean magnitude local interpolation controls, separately scored.

This predicts mean(hypot(Br,Bt)), unlike the original112 signed-mean controls.
Frozen NN magnitude fluctuation is added, raw prediction and max0 projection
are scored separately. Common recipe selection uses raw original-T RMSE.
"""
from pathlib import Path
import json,time,zipfile
import numpy as np
import local_interpolation as li

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'direct_mean_magnitude'


def mean_rows_magnitude(path,indices):
    mapping={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as z,z.open('field.npy') as f:
        ver=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,ver)
        if shape!=(60,60,3600) or fort:raise ValueError('Original rotor field shape invalid')
        out=np.empty((len(indices),1800));block=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(60):
            if i in mapping:
                row=np.frombuffer(f.read(block),dtype=dtype).reshape(shape[1:]).astype(np.float64);out[mapping[i]]=np.hypot(row[:,0::2],row[:,1::2]).mean(0)
            else:f.seek(block,1)
    return out


def predict(model,conditions,frame_index=None,nonnegative=False):
    m=li.load_model(model)
    if li.digest(li.__file__)!=str(m['local_inference_source_sha256']):raise ValueError('Frozen local interpolation source mismatch')
    c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    if c.ndim!=2 or c.shape[-1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Finite native RMS-current>=0/Thet required')
    temporary=dict(m,signed_mean_targets=m['mean_magnitude_targets'])
    mean=li.mean_prediction(temporary,c);signed=li.neural_cycle({k[9:]:v for k,v in m.items() if k.startswith('baseline_')},c)
    raw=np.hypot(signed[...,0::2],signed[...,1::2]);out=mean[:,None,:]+(raw-raw.mean(1,keepdims=True))
    if not np.isfinite(out).all():raise ValueError('Nonfinite raw prediction')
    if nonnegative:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<60:raise ValueError('Original saved frame0..59 required')
        out=out[:,frame_index]
    return out[0] if single else out


def run():
    OUT.mkdir(exist_ok=False);start=time.perf_counter();s=json.loads((li.V1/'cache_v1/split.json').read_text('utf8'));tr=s['train'];va=s['validation'];cache=li.V1/'cache_v1/rotor.npz'
    configs=[dict(method='local_quadratic_mls',coordinate=c,neighbors=k,bandwidth=h,ridge=r) for c in ('native','cartesian','sat300') for k in (12,18,24) for h in (1.,2.) for r in (1e-5,.001)]
    configs += [dict(method='delaunay_linear',coordinate=c) for c in ('native','cartesian')]
    li.dump(OUT/'protocol.json',dict(candidate_count=len(configs),configs=configs,training_case_ids=tr,validation_case_ids=va,excluded_test_case_ids=s['test'],
        quantity='mean(hypot(Br,Bt)) + frozenV2 pureNN magnitude fluctuation with its own60frame mean removed',
        distinction='Not hypot(mean(Br),mean(Bt)) and not the signed-mean112 controls',
        selection='Full raw original-T val9x60x1800 magnitude RMSE across common recipes; no per-val-point routing',
        nonnegative_policy='max(raw,0) is an explicit alternate deployment transform, every configuration records raw/deployed errors and negative count. Selection uses raw RMSE only.',
        train_only='Targets, coordinate mean/std and exterior affine coefficients fit train42 only',
        exterior='Same declared train-only affine/ridge.001 exterior as signed controls; all cases retained',
        test_labels_materialized=False,full60_fitting_used=False,neural_training_runs=0,
        original112_protocol_sha256=li.digest(ROOT/'protocol.json'),local_inference_source_sha256=li.digest(li.__file__),source_sha256=li.digest(__file__),cache_sha256=li.digest(cache)))
    with np.load(cache,allow_pickle=False) as z:c=z['conditions'];pid=z['point_id'];xy=z['xy'];ids=z['case_ids']
    target=mean_rows_magnitude(cache,tr);signed=li.val_rows(cache,va);truth=np.hypot(signed[...,0::2],signed[...,1::2]);vmean=truth.mean(1);signed_mean_mag=np.hypot(signed.mean(1)[:,0::2],signed.mean(1)[:,1::2]);del signed
    # Pull only frozen neural weights from the original112 control winner.
    modelold=li.load_model(ROOT/'selected_model.npz');base={k:v for k,v in modelold.items() if k.startswith('baseline_')}
    neural=li.neural_cycle({k[9:]:v for k,v in base.items()},c[va]);nnmag=np.hypot(neural[...,0::2],neural[...,1::2]);dynamic=nnmag-nnmag.mean(1,keepdims=True)
    discrepancy=np.abs(vmean-signed_mean_mag);li.dump(OUT/'target_semantics_check.json',dict(validation_mean_norm_vs_norm_mean=dict(rmse_T=float(np.sqrt(np.mean(discrepancy**2))),relative_L2_pct=float(100*np.linalg.norm(discrepancy)/np.linalg.norm(vmean)),max_abs_T=float(discrepancy.max())),
        validation_only_diagnostic=True,not_deployed_as_predictor=True,training_target='mean(norm signed BrBt)',full_case_point_coverage=True))
    rows=[];best=np.inf;bm=None;br=None;bp=None
    for index,spec in enumerate(configs):
        raw=li.coordinates(c[tr],spec['coordinate']);center=raw.mean(0);scale=np.maximum(raw.std(0),1e-8);x=(raw-center)/scale;d=li.affine(x);outside=np.linalg.solve(d.T@d+np.diag([1e-6,.001,.001]),d.T@target)
        m=dict(input_center=center,input_scale=scale,centers=x,mean_magnitude_targets=target,exterior_affine_weights=outside,spec_json=np.array(json.dumps(spec)),point_id=pid,xy=xy,**base)
        tmp=dict(m,signed_mean_targets=target);row=dict(index=index,configuration=spec)
        try:
            mean,diag=li.mean_prediction(tmp,c[va],True);pred=mean[:,None,:]+dynamic;rawmet=li.metrics(pred,truth,ids[va]);positive=li.metrics(np.maximum(pred,0),truth,ids[va])
            row.update(validation_raw=rawmet,validation_nonnegative=positive,negative_raw_count=int(np.count_nonzero(pred<0)),raw_min_T=float(np.min(pred)),query_geometry=diag,
                validation_mean_rmse_T=float(np.sqrt(np.mean((mean-vmean)**2))))
            if rawmet['finite'] and rawmet['rmse_T']<best:best=rawmet['rmse_T'];bm=m;br=row;bp=pred.copy();print('BEST',index,spec,rawmet,flush=True)
        except (ValueError,np.linalg.LinAlgError) as err:row['failure']=str(err)
        rows.append(row)
    bm.update(local_inference_source_sha256=np.array(li.digest(li.__file__)),metadata_json=np.array(json.dumps(dict(format='V4-direct-time-mean-magnitude-local-control',quantity='planar_magnitude',target_semantics='mean(hypot(Br,Bt))',selected_recipe=br['configuration'],training_case_ids=tr,validation_case_ids=va,test_labels_materialized=False,nonnegative_default=False,source_sha256=li.digest(__file__)))))
    np.savez_compressed(OUT/'selected_model.npz',**bm);(OUT/'selected_model.npz.sha256').write_text(li.digest(OUT/'selected_model.npz')+'\n',encoding='ascii')
    replay=predict(OUT/'selected_model.npz',c[va]);delta=float(np.max(np.abs(replay-bp)))
    if delta>1e-9:raise AssertionError('Raw deployed NumPy replay differs '+str(delta))
    positive=predict(OUT/'selected_model.npz',c[va],nonnegative=True)
    if not np.array_equal(positive,np.maximum(bp,0)):raise AssertionError('Projection replay mismatch')
    np.savez_compressed(OUT/'validation_best.npz',prediction=bp,prediction_nonnegative=positive,case_ids=ids[va],point_id=pid,xy=xy)
    li.dump(OUT/'candidate_results.json',dict(rows=rows,selected=br,candidate_count=len(rows),successful_count=sum('validation_raw' in r for r in rows),failed_count=sum('failure' in r for r in rows),
        baseline_V2_pureNN=li.metrics(nnmag,truth,ids[va]),neural_training_runs=0,test_labels_materialized=False,full60_fitting_used=False,
        source_sha256=li.digest(__file__),local_inference_source_sha256=li.digest(li.__file__),model_sha256=li.digest(OUT/'selected_model.npz'),replay_max_difference_T=delta,elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(dict(selected=br,completed=len(rows),elapsed_seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run()
