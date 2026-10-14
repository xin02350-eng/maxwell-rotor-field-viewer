"""Independent gate provenance, original-T metrics and portable NumPy replay."""
from pathlib import Path
import shutil,sys,importlib.util,importlib.abc,json
import numpy as np
from threadpoolctl import threadpool_limits


class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','sklearn','scipy'):raise RuntimeError('Forbidden inference import '+fullname)
        return None


sys.meta_path.insert(0,NoTraining())
import gated_neural_ensemble as ge
ROOT=Path(__file__).resolve().parent;folder=ROOT/'gated_neural_ensemble';checks=[]


def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError(name)


def read(path):return json.loads(Path(path).read_text('utf8'))


def run():
    protocol=read(folder/'candidate_protocol.json');result=read(folder/'selection_validation_only.json');frozen=read(folder/'train_risk_masks_frozen.json');split=read(ge.V1/'cache_v1/split.json')
    check('25 combinations and zero new NN training',len(result['rows'])==25 and protocol['candidate_count']==25 and result['new_neural_training_runs']==0)
    check('No test/full60fit label use declared',protocol['test_labels_materialized'] is False and result['full60_fitting_used'] is False and frozen['test_labels_used'] is False)
    check('Source binding',protocol['source_sha256']==ge.digest(ge.__file__) and result['source_sha256']==ge.digest(ge.__file__))
    for rel,sha in protocol['dependencies'].items():check('DependencySHA '+rel,ge.digest(ROOT.parent/rel)==sha)
    riskpath=ROOT/'static_local_control/training_loo_diagnostics.npz'
    check('Frozen train risk input',frozen['risk_source_sha256']==ge.digest(riskpath) and frozen['train_case_ids']==split['train'])
    with np.load(riskpath,allow_pickle=False) as z:risk=np.mean((z['shared_best_point_loo_predictions']-z['train_mean_fields'])**2,axis=0);pid=z['point_id']
    order=np.argsort(-risk,kind='stable');check('Recompute train-only risk scores',np.allclose(risk,frozen['score'],rtol=1e-14))
    for n in (18,90,180,600,1800):check('GeometrypointIDs preserved top'+str(n),frozen['masks'][str(n)]['point_indices']==order[:n].tolist() and frozen['masks'][str(n)]['point_ids']==pid[order[:n]].tolist())
    minimum=min(result['rows'],key=lambda r:r['validation']['rmse_T']);check('Select fullval RMSE only',minimum==result['selected'])
    modelpath=folder/'selected_model.npz';m=ge.load_model(modelpath);conf=result['selected']['configuration'];weights=np.zeros(1800);weights[order[:conf['mask_point_count']]]=conf['operator_alpha']
    check('Saved point mask exactly matches predeclared train-risk selection',np.array_equal(weights,m['operator_weight_by_point']))
    check('Convex fixed weights',np.all(weights>=0) and np.all(weights<=1) and weights.shape==(1800,))
    check('All models preserve original point identities',np.array_equal(m['point_id'],m['a_point_id']) and np.array_equal(m['point_id'],m['b_point_id']))
    with np.load(ge.V1/'cache_v1/rotor.npz',allow_pickle=False) as z:c=z['conditions'][split['validation']];ids=z['case_ids'][split['validation']]
    signed=ge.permitted_rows(ge.V1/'cache_v1/rotor.npz',split['validation']);truth=np.hypot(signed[...,0::2],signed[...,1::2]);del signed
    pred=ge.predict(modelpath,c)
    with np.load(folder/'validation_predictions.npz',allow_pickle=False) as z:saved=z['prediction'];savedids=z['case_ids'];savedpid=z['point_id']
    err=float(np.max(np.abs(pred-saved)));check('Full precision completeval replay',err<1e-10,{'max_diff_T':err})
    check('All9x60x1800 coverage and IDs',pred.shape==(9,60,1800) and np.array_equal(savedids,ids) and np.array_equal(savedpid,pid))
    mets=ge.metrics(pred,truth,ids)
    for name in ('rmse_T','relative_L2_pct','mae_T','max_abs_T','worst_case_rmse_T'):check('Recompute original-T metric '+name,np.isclose(mets[name],result['selected']['validation'][name],rtol=1e-12,atol=1e-12))
    check('No negative projection needed',np.min(pred)>=0 and mets['negative_count']==0)
    one,two=ge.branches(m,c);check('No change outside train-risk mask',np.array_equal(pred[:,:,weights==0],one[:,:,weights==0]))
    check('TwoNN formula at selected points',np.allclose(pred[:,:,weights>0],(1-weights[weights>0])[None,None,:]*one[:,:,weights>0]+weights[weights>0][None,None,:]*two[:,:,weights>0],rtol=0,atol=0))
    # Relocate only new source/model/sidecar. Reject every external NPZ load.
    portable=folder/'qa_portable'
    if not portable.exists():portable.mkdir()
    for p in (Path(ge.__file__),modelpath,modelpath.with_suffix('.npz.sha256')):
        target=portable/p.name
        if target.exists():check('Reused portable file source parity '+p.name,ge.digest(target)==ge.digest(p))
        else:shutil.copyfile(p,target)
    spec=importlib.util.spec_from_file_location('portable_gated_nn_only',portable/Path(ge.__file__).name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=np.load;loads=[]
    def checked_load(path,*args,**kwargs):
        resolved=Path(path).resolve();loads.append(str(resolved))
        if resolved!= (portable/modelpath.name).resolve():raise AssertionError('Inference touched external NPZ '+str(resolved))
        return original(path,*args,**kwargs)
    np.load=checked_load
    try:
        actual=module.predict(portable/modelpath.name,c);frame=module.predict(portable/modelpath.name,c[0],frame_index=59)
        for invalid in (-1,60):
            try:module.predict(portable/modelpath.name,c[0],frame_index=invalid)
            except ValueError:check('Invalid saved frame rejected '+str(invalid),True)
            else:check('Invalid saved frame rejected '+str(invalid),False)
    finally:np.load=original
    check('Relocated inference no workspace model reads',len(loads)==4 and np.max(np.abs(actual-pred))<1e-10,{'loaded_npz_paths':loads})
    single_error=float(np.max(np.abs(frame-pred[0,59])))
    check('Final supplied frame59 preserved within float32 operator roundoff',single_error<5e-5,{'max_diff_T':single_error,'tolerance_T':5e-5,'reason':'Float32 shared operator GEMM accumulates with different batch dimensions; scientific multi-case validation output is float64 after assembly'})
    check('No training imports',not any(k.split('.')[0] in ('torch','sklearn','scipy') for k in sys.modules))
    check('Output array genuine NumPy and finite',isinstance(actual,np.ndarray) and np.isfinite(actual).all())
    dump=dict(passed=all(z['passed'] for z in checks),check_count=len(checks),checks=checks,test_labels_materialized=False,source_sha256=ge.digest(__file__),
        model_sha256=ge.digest(modelpath),inference_source_sha256=ge.digest(ge.__file__))
    ge.dump(folder/'qa_replay.json',dump);print(json.dumps(dict(passed=dump['passed'],checks=len(checks),model_sha256=dump['model_sha256'])),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
