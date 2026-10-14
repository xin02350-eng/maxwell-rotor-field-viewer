"""Separate semantic/raw/projection/portable checks for38 unsigned controls."""
from pathlib import Path
import json,sys,shutil,importlib.util
import numpy as np
from threadpoolctl import threadpool_limits
import direct_mean_magnitude as dm

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'direct_mean_magnitude';checks=[]


def read(path):return json.loads(Path(path).read_text('utf8'))


def check(name,value,detail=None):
    checks.append(dict(name=name,passed=bool(value),detail=detail))
    if not value:raise AssertionError(name)


def run():
    protocol=read(OUT/'protocol.json');result=read(OUT/'candidate_results.json');s=read(dm.li.V1/'cache_v1/split.json');tr=s['train'];va=s['validation'];cache=dm.li.V1/'cache_v1/rotor.npz'
    check('38 completed analytic controls',len(result['rows'])==38 and protocol['candidate_count']==38 and result['neural_training_runs']==0)
    check('112 + 38 total150 original recipes',read(ROOT/'candidate_results.json')['completed_configurations']+len(result['rows'])==150)
    check('No test/full60 labels',protocol['test_labels_materialized'] is False and result['full60_fitting_used'] is False)
    check('Source and common inference binding',protocol['source_sha256']==dm.li.digest(dm.__file__) and protocol['local_inference_source_sha256']==dm.li.digest(dm.li.__file__))
    check('Original112 protocol preserved',protocol['original112_protocol_sha256']==dm.li.digest(ROOT/'protocol.json'))
    check('Only correct split',protocol['training_case_ids']==tr and protocol['validation_case_ids']==va)
    target=dm.mean_rows_magnitude(cache,tr);m=dm.li.load_model(OUT/'selected_model.npz')
    check('Training targets are mean(norm)1800 magnitudes',m['mean_magnitude_targets'].shape==(42,1800) and np.array_equal(target,m['mean_magnitude_targets']))
    check('No falsely named signed mean targets in saved magnitude model','signed_mean_targets' not in m)
    with np.load(cache,allow_pickle=False) as z:c=z['conditions'];pid=z['point_id'];ids=z['case_ids'];xy=z['xy']
    check('Originalpoint identity',np.array_equal(m['point_id'],pid) and np.array_equal(m['xy'],xy))
    spec=json.loads(str(m['spec_json']));coords=dm.li.coordinates(c[tr],spec['coordinate']);mean=coords.mean(0);scale=np.maximum(coords.std(0),1e-8)
    check('Scalers train42 only',np.array_equal(m['input_center'],mean) and np.array_equal(m['input_scale'],scale))
    pred=dm.predict(OUT/'selected_model.npz',c[va]);positive=dm.predict(OUT/'selected_model.npz',c[va],nonnegative=True)
    with np.load(OUT/'validation_best.npz',allow_pickle=False) as z:stored=z['prediction'];storedpositive=z['prediction_nonnegative'];storedids=z['case_ids']
    check('Complete9x60x1800 raw replay',pred.shape==(9,60,1800) and np.array_equal(pred,stored))
    check('Explicit max0 deployed transformation',np.array_equal(positive,np.maximum(pred,0)) and np.array_equal(positive,storedpositive))
    check('Validation IDs original',np.array_equal(storedids,ids[va]))
    fv=dm.li.val_rows(cache,va);truth=np.hypot(fv[...,0::2],fv[...,1::2]);signed_mean=np.hypot(fv.mean(1)[:,0::2],fv.mean(1)[:,1::2]);magmean=truth.mean(1);del fv
    rawmet=dm.li.metrics(pred,truth,ids[va]);projmet=dm.li.metrics(positive,truth,ids[va])
    for key in ('rmse_T','relative_L2_pct','mae_T','max_abs_T','worst_case_rmse_T'):
        check('Raw/deployed original-T '+key,np.isclose(rawmet[key],result['selected']['validation_raw'][key],rtol=1e-12) and np.isclose(projmet[key],result['selected']['validation_nonnegative'][key],rtol=1e-12))
    semantic=read(OUT/'target_semantics_check.json')['validation_mean_norm_vs_norm_mean'];diff=magmean-signed_mean
    check('Two temporal mean semantics differ and correctly reported',np.isclose(semantic['rmse_T'],np.sqrt(np.mean(diff*diff)),rtol=1e-12) and semantic['max_abs_T']>.27)
    check('Raw-only selection',min(result['rows'],key=lambda r:r.get('validation_raw',{}).get('rmse_T',np.inf))==result['selected'])
    # Replay the already tried first negative candidate to verify no censorship.
    row=result['rows'][0];conf=row['configuration'];rawc=dm.li.coordinates(c[tr],conf['coordinate']);cm=rawc.mean(0);cs=np.maximum(rawc.std(0),1e-8);x=(rawc-cm)/cs;d=dm.li.affine(x)
    negative_model=dict(m,input_center=cm,input_scale=cs,centers=x,spec_json=np.array(json.dumps(conf)),exterior_affine_weights=np.linalg.solve(d.T@d+np.diag([1e-6,.001,.001]),d.T@target))
    neg=dm.predict(negative_model,c[va]);clipped=dm.predict(negative_model,c[va],nonnegative=True)
    check('Negative configuration preserved raw',int(np.count_nonzero(neg<0))==row['negative_raw_count'] and row['negative_raw_count']>0)
    check('Projection never deletes points',clipped.shape==neg.shape and np.array_equal(clipped,np.maximum(neg,0)))
    check('Negative candidate raw and deployed metrics true',np.isclose(dm.li.metrics(neg,truth,ids[va])['rmse_T'],row['validation_raw']['rmse_T'],rtol=1e-12) and np.isclose(dm.li.metrics(clipped,truth,ids[va])['rmse_T'],row['validation_nonnegative']['rmse_T'],rtol=1e-12))
    portable=OUT/'qa_portable';portable.mkdir(exist_ok=False)
    modelpath=OUT/'selected_model.npz'
    for source in (Path(dm.__file__),Path(dm.li.__file__),modelpath,modelpath.with_suffix('.npz.sha256')):shutil.copyfile(source,portable/source.name)
    sp=importlib.util.spec_from_file_location('portable_li_unsigned',portable/Path(dm.li.__file__).name);common=importlib.util.module_from_spec(sp);sp.loader.exec_module(common)
    previous=sys.modules.get('local_interpolation');sys.modules['local_interpolation']=common
    try:
        sp2=importlib.util.spec_from_file_location('portable_unsigned38',portable/Path(dm.__file__).name);module=importlib.util.module_from_spec(sp2);sp2.loader.exec_module(module)
    finally:
        if previous is not None:sys.modules['local_interpolation']=previous
        else:sys.modules.pop('local_interpolation',None)
    original=np.load;loads=[]
    def only_model(path,*args,**kwargs):
        actual=Path(path).resolve();loads.append(str(actual))
        if actual!=(portable/modelpath.name).resolve():raise AssertionError('External deployment NPZ read')
        return original(path,*args,**kwargs)
    np.load=only_model
    try:portable_raw=module.predict(portable/modelpath.name,c[va]);portable_clip=module.predict(portable/modelpath.name,c[va],nonnegative=True)
    finally:np.load=original
    check('Relocated raw/projection with only one model loaded',len(loads)==2 and np.array_equal(portable_raw,pred) and np.array_equal(portable_clip,positive),{'npz_reads':loads})
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());check('Additional storage under150MB',size<150_000_000,{'bytes':size})
    report=dict(passed=all(v['passed'] for v in checks),check_count=len(checks),checks=checks,test_labels_materialized=False,
        source_sha256=dm.li.digest(__file__),inference_source_sha256=dm.li.digest(dm.__file__),model_sha256=dm.li.digest(modelpath))
    dm.li.dump(OUT/'qa_replay.json',report);print(json.dumps(dict(passed=report['passed'],checks=len(checks),bytes=size)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
