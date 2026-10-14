"""NumPy-only deployment replay, embedding/training boundary verification."""
from pathlib import Path
import json,sys,importlib.abc
import numpy as np
from threadpoolctl import threadpool_limits


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','sklearn'):raise RuntimeError('Forbidden training-only inference import '+fullname)
        return None


sys.meta_path.insert(0,BlockTorch())
import point_operator_nn as op
ROOT=Path(__file__).resolve().parent;checks=[]


def check(name,value,detail=None):
    checks.append(dict(name=name,passed=bool(value),detail=detail))
    if not value:raise AssertionError(name)


def run():
    folder=ROOT/'point_operator_nn';protocol=op.r.safe.read_json(folder/'protocol.json');info=op.r.safe.read_json(folder/'preparation_identity.json');results=op.r.safe.read_json(folder/'results.json')
    split=op.r.safe.read_json(op.V1/'cache_v1/split.json');va=split['validation']
    with np.load(op.V1/'cache_v1/rotor.npz',allow_pickle=False) as z:c=z['conditions'][va];pid=z['point_id'];ids=z['case_ids'][va]
    truthsigned=op.r.safe.selected_rows(op.V1/'cache_v1/rotor.npz','field',va).astype(np.float64);truth=np.hypot(truthsigned[...,0::2],truthsigned[...,1::2]);del truthsigned
    check('Only two trueNN runs',results['completed_training_runs']==2 and len(protocol['configs'])==2)
    check('Source unchanged',protocol['source_sha256']==op.r.safe.digest(op.__file__))
    check('Embeddings train-only',info['train_case_ids']==split['train'] and info['validation_case_ids']==ids.tolist() and info['test_labels_materialized'] is False and info['full60_fitting_used'] is False)
    check('Training validation test disjoint',len(set(info['train_case_ids']+info['validation_case_ids']+info['excluded_test_case_ids']))==60)
    check('No training libraries imported',not any(k.split('.')[0] in ('torch','sklearn') for k in sys.modules))
    for i in (1,2):
        p=folder/f'pilot_{i:02d}';m=op.load_model(p/'model.npz');summ=op.r.safe.read_json(p/'summary.json');hist=op.r.safe.read_json(p/'history.json')
        check(str(i)+' PCA8 original point shape',m['point_embedding'].shape==(1800,8) and m['point_attributes'].shape==(1800,6) and np.array_equal(m['point_id'],pid))
        check(str(i)+' Four linear layers',int(m['layer_count'])==4 and m['w0'].shape==(21,128) and m['w3'].shape==(128,2))
        check(str(i)+' Real trained float32 weights',all(m[f'w{k}'].dtype==np.float32 for k in range(4)) and summ['executed_steps']>=1200 and summ['trainable_parameters']>30000)
        check(str(i)+' Best validation checkpoint',summ['selected_step']==min(hist,key=lambda h:h['validation_rmse_T'])['step'])
        check(str(i)+' Float32 Torch parity',summ['float32_torch_numpy_max_signed_mean_difference_T']<5e-5)
        pred=op.predict(p/'model.npz',c)
        with np.load(p/'validation_predictions.npz',allow_pickle=False) as z:saved=z['prediction'];storedids=z['case_ids'];storedpoint=z['point_id']
        err=float(np.max(np.abs(pred-saved)));check(str(i)+' NumPy-only full replay',err<1e-10,{'max_diff_T':err})
        check(str(i)+' Coverage and IDs',saved.shape==(9,60,1800) and np.array_equal(storedids,ids) and np.array_equal(storedpoint,pid))
        met=op.r.metrics(pred,truth)
        for key in ('relative_L2_pct','rmse_T','mae_T','max_abs_T'):
            check(str(i)+' raw-T metric '+key,np.isclose(met[key],summ['validation'][key],rtol=1e-11,atol=1e-11))
        signed=op.predict(p/'model.npz',c[0],frame_index=2,return_signed=True)
        check(str(i)+' Signed to magnitude proper',np.allclose(np.hypot(signed[0::2],signed[1::2]),pred[0,2],atol=1e-8))
    size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file());check('150MB extra storage budget',size<150_000_000,{'bytes':size})
    check('Distinct fixed vs trainable embedding configs',results['results'][0]['config']['trainable_embedding'] is False and results['results'][1]['config']['trainable_embedding'] is True)
    report=dict(passed=all(v['passed'] for v in checks),check_count=len(checks),checks=checks,completed_neural_runs=2,
        test_labels_materialized=False,source_sha256=op.r.safe.digest(__file__),file_hashes={str(p.relative_to(ROOT)):op.r.safe.digest(p) for p in folder.rglob('*') if p.is_file()})
    op.r.dump(folder/'qa_replay.json',report);print(json.dumps({'passed':report['passed'],'checks':len(checks),'bytes':size}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
