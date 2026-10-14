"""Independent original-T evaluation and relocated inference checks."""
from pathlib import Path
import json,sys,shutil,importlib.util,importlib.abc
import numpy as np
from threadpoolctl import threadpool_limits


class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','sklearn'):raise RuntimeError('Training dependency forbidden during inference '+fullname)
        return None


sys.meta_path.insert(0,NoTraining())
import local_interpolation as li
ROOT=Path(__file__).resolve().parent;checks=[]


def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError(name)


def read(path):return json.loads(Path(path).read_text('utf8'))


def run():
    protocol=read(ROOT/'protocol.json');results=read(ROOT/'candidate_results.json');regimes=read(ROOT/'regimes_train_only.json');split=read(li.V1/'cache_v1/split.json');va=split['validation'];tr=split['train']
    check('112 finite configurations no NN training',len(protocol['configs'])==112 and len(results['rows'])==112 and results['neural_training_runs']==0)
    check('All success or failures explicitly recorded',results['successful_candidate_count']+results['failed_candidate_count']==112)
    check('All112 method recipes recorded',{x['configuration']['method'] for x in results['rows']}=={'delaunay_linear','clough_tocher','local_quadratic_mls','local_rbf','regime_affine'})
    check('No test/full60fitting labels',protocol['test_labels_materialized'] is False and protocol['full60_fitting_used'] is False)
    check('Training-validation-test boundaries',protocol['training_case_ids']==tr and protocol['validation_case_ids']==va and len(set(tr+va+protocol['excluded_test_case_ids']))==60)
    check('Source SHA binding',protocol['source_sha256']==li.digest(li.__file__) and results['inference_source_sha256']==li.digest(li.__file__))
    check('Input cache source SHA',protocol['cache_sha256']==li.digest(li.V1/'cache_v1/rotor.npz'))
    for rel,sha in protocol['dependencies'].items():check('Frozen dynamic source '+rel,li.digest(li.WORK/rel)==sha)
    check('Train-only PCA regimes',regimes['pca_training_only'] is True and regimes['train_case_ids']==tr and regimes['pca_rank']==4)
    for k in (2,3,4):check('42 train regime IDs for'+str(k),len(regimes['cluster_ids'][str(k)])==42 and set(regimes['cluster_ids'][str(k)])==set(range(k)))
    modelpath=ROOT/'selected_model.npz';m=li.load_model(modelpath);check('Model SHA',li.digest(modelpath)==results['model_sha256'])
    with np.load(li.V1/'cache_v1/rotor.npz',allow_pickle=False) as z:c=z['conditions'];pid=z['point_id'];ids=z['case_ids'];xy=z['xy']
    mean=li.mean_rows(li.V1/'cache_v1/rotor.npz',tr)
    check('Targets exactly train signed temporal mean',np.array_equal(m['signed_mean_targets'],mean))
    spec=json.loads(str(m['spec_json']));raw=li.coordinates(c[tr],spec['coordinate']);center=raw.mean(0);scale=np.maximum(raw.std(0),1e-8)
    check('Input normalizers train only',np.array_equal(m['input_center'],center) and np.array_equal(m['input_scale'],scale))
    check('Original point IDs and sample positions',np.array_equal(m['point_id'],pid) and np.array_equal(m['xy'],xy) and len(pid)==1800)
    pred=li.predict(modelpath,c[va]);tv=li.val_rows(li.V1/'cache_v1/rotor.npz',va);truth=np.hypot(tv[...,0::2],tv[...,1::2]);del tv
    with np.load(ROOT/'validation_best.npz',allow_pickle=False) as z:saved=z['prediction'];savedids=z['case_ids'];savedpoint=z['point_id']
    error=float(np.max(np.abs(pred-saved)));check('Fullval9x60x1800 replay',pred.shape==(9,60,1800) and error<1e-10,dict(max_diff_T=error))
    check('Scientific prediction IDs',np.array_equal(savedids,ids[va]) and np.array_equal(savedpoint,pid))
    met=li.metrics(pred,truth,ids[va])
    for key in ('rmse_T','relative_L2_pct','mae_T','max_abs_T','worst_case_rmse_T'):check('Raw original-T '+key,np.isclose(met[key],results['selected']['validation'][key],rtol=1e-12,atol=1e-12))
    check('Selection minimum complete validation RMSE',min(results['rows'],key=lambda row:row.get('validation',{}).get('rmse_T',float('inf')))==results['selected'])
    check('No points deleted or magnitude clipping',np.min(pred)>=0 and met['negative_magnitude_count']==0)
    signed=li.predict(modelpath,c[va],return_signed=True);check('Signed mean + signed dynamic before hypot',np.array_equal(pred,np.hypot(signed[...,0::2],signed[...,1::2])))
    current=np.array([[2000.,180.]])
    mm,diagnostic=li.mean_prediction(m,current,True);x=(li.coordinates(current,spec['coordinate'])-m['input_center'])/m['input_scale'];expected=li.affine(x)@m['exterior_affine_weights']
    check('Explicit exterior signed affine response',diagnostic['outside_train_convex_hull_count']==1 and np.array_equal(mm,expected))
    check('Exterior kept finite with original full point count',li.predict(modelpath,current).shape==(1,60,1800) and np.isfinite(li.predict(modelpath,current)).all())
    p=ROOT/'qa_portable';p.mkdir(exist_ok=False)
    for source in (Path(li.__file__),modelpath,modelpath.with_suffix('.npz.sha256')):shutil.copyfile(source,p/source.name)
    specmod=importlib.util.spec_from_file_location('portable_local_v4',p/Path(li.__file__).name);module=importlib.util.module_from_spec(specmod);specmod.loader.exec_module(module)
    oldload=np.load;loads=[]
    def only_model(path,*args,**kwargs):
        actual=Path(path).resolve();loads.append(str(actual))
        if actual!=(p/modelpath.name).resolve():raise AssertionError('Inference read external file '+str(actual))
        return oldload(path,*args,**kwargs)
    np.load=only_model
    try:actual=module.predict(p/modelpath.name,c[va]);last=module.predict(p/modelpath.name,c[va[0]],frame_index=59)
    finally:np.load=oldload
    check('Relocated inference only saved model loaded',len(loads)==2 and np.max(np.abs(actual-pred))<1e-10,{'loaded_paths':loads})
    check('Actual supplied final frame59',np.max(np.abs(last-pred[0,59]))<1e-10)
    check('No Torch or sklearn imported',not any(key.split('.')[0] in ('torch','sklearn') for key in sys.modules))
    total=sum(q.stat().st_size for q in ROOT.rglob('*') if q.is_file());check('Small output storage',total<150_000_000,{'bytes':total})
    report=dict(passed=all(z['passed'] for z in checks),check_count=len(checks),checks=checks,source_sha256=li.digest(__file__),inference_sha256=li.digest(li.__file__),model_sha256=li.digest(modelpath),test_labels_materialized=False)
    li.dump(ROOT/'qa_replay.json',report);print(json.dumps(dict(passed=report['passed'],checks=len(checks),bytes=total)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
