"""Independent saved-prediction replay and experiment-boundary checks."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import rational_control as rc
import rational_neighbor_control as nc
import local_mean_rbf_control as lc

ROOT=Path(__file__).resolve().parent; checks=[]


def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError(name)


def run():
    split=rc.safe.read_json(rc.V1/'cache_v1/split.json');va=split['validation']
    with np.load(rc.V1/'cache_v1/rotor.npz',allow_pickle=False) as z:
        conditions=z['conditions'][va];ids=z['case_ids'][va];xy=z['xy'];pid=z['point_id']
    nnpath=rc.V2/'neural_route/rotor/selected_model.npz'
    signed=rc.safe.predict(nnpath,conditions);nn=np.hypot(signed[...,0::2],signed[...,1::2]);dynamic=nn-nn.mean(1,keepdims=True)
    expectedcounts={'rational_control':94,'rational_neighbor_control':6,'local_mean_rbf_control':48}
    sizes=0
    for name,count in expectedcounts.items():
        folder=ROOT/name;protocol=rc.safe.read_json(folder/'protocol.json');ranking=rc.safe.read_json(folder/'ranking.json');summary=rc.safe.read_json(folder/'summary.json')
        check(name+' candidate count',len(ranking)==count)
        check(name+' exclusion',protocol.get('test_labels_materialized') is False and protocol.get('full60_fit_used',protocol.get('full60_fitting_used',False)) is False)
        check(name+' train and validation disjoint',not set(protocol.get('training_case_ids',protocol.get('train_case_ids')))&set(protocol.get('validation_case_ids')))
        check(name+' correct split',protocol.get('validation_case_ids')==ids.tolist())
        check(name+' model sidecar',rc.safe.digest(folder/'model.npz')==(folder/'model.npz.sha256').read_text().strip())
        with np.load(folder/'validation_predictions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        check(name+' originalpoint coverage',a['prediction'].shape==(9,60,1800) and np.array_equal(a['point_id'],pid) and np.array_equal(a['case_ids'],ids))
        if name=='rational_control':
            mean,den=rc.predict_mean(folder/'model.npz',conditions);pred=mean[:,None,:]+dynamic
            check(name+' denominator replay',np.array_equal(den,a['denominator']))
            check(name+' raw negative consistency',int(np.count_nonzero(pred<0))==summary['best']['validation']['negative_count'])
            check(name+' failures explicitly preserved',any(row.get('denominator_near_pole_validation',0)>0 for row in ranking) and any(not row.get('validation',{}).get('finite',True) for row in ranking))
        elif name=='rational_neighbor_control':
            pred=nc.predict(folder/'model.npz',conditions)
            with np.load(folder/'model.npz',allow_pickle=False) as z:
                nb=z['neighbor_indices'];given=z['neighbor_point_ids'];k=nb.shape[-1]
            d=np.sum((xy[:,None,:]-xy[None,:,:])**2,axis=-1);np.fill_diagonal(d,np.inf)
            check(name+' geometry-only neighbour identities',np.array_equal(nb,np.argsort(d,axis=1)[:,:k]) and np.array_equal(given,pid[nb]))
        else:
            pred=lc.predict(folder/'model.npz',conditions)
            check(name+' scipy numpy parity',summary['numpy_scipy_max_difference_T']<1e-9)
            check(name+' signed-to-magnitude nonnegative',np.min(pred)>=0)
        err=float(np.max(np.abs(pred-a['prediction'])))
        check(name+' saved full-field replay',err<1e-10,{'max_difference_T':err})
        sizes+=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    check('Total completed finite recipes',sum(expectedcounts.values())==148)
    check('Storage budget',sizes<150_000_000,{'bytes':sizes})
    output=dict(passed=all(x['passed'] for x in checks),check_count=len(checks),checks=checks,completed_recipe_count=148,
        aborted_pilot_count=1,aborted_pilot_excluded_from_completed_recipes=True,test_labels_materialized=False,
        source_sha256=rc.safe.digest(__file__),file_hashes={str(p.relative_to(ROOT)):rc.safe.digest(p)
            for n in expectedcounts for p in (ROOT/n).rglob('*') if p.is_file()})
    rc.dump(ROOT/'mean_controls_check.json',output)
    print(json.dumps({'passed':output['passed'],'checks':len(checks),'bytes':sizes}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
