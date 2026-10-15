"""One final retrospective reused-test replay after immutable RBF choices.
No test metric is used by this script to change a model or recipe.
"""
import json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import rbf_pointwise as r
import inference_v6 as runtime

def run():
    root=r.ROOT;start=time.perf_counter();split=json.loads((r.WORK/'new60_20261014/cache_v1/split.json').read_text());va=split['validation'];te=split['test']
    choice=json.loads((root/'rbf_choices_pretest.json').read_text());nn=json.loads((root/'rbf_neural/results.json').read_text());assert nn['actual_new_gradient_NN_runs']==4
    selected=[]
    for region,s in choice['regions'].items():
        for scope,folder in [('original42',root/s['study']),('final51',root/'final51'/region)]:
            for name in (['shared','pointLOO'] if s['recipe']['route']=='pointLOO_shrunk50' else [s['recipe']['route']]):
                p=folder/(name+'.npz');selected.append(dict(path=p.relative_to(root).as_posix(),sha256=r.sha(p),scope=scope,region=region))
    sources=[dict(path=p.name,sha256=r.sha(p)) for p in [root/'rbf_pointwise.py',root/'rbf_neural.py',root/'inference_v6.py',root/'refit51.py',Path(__file__)]]
    frozen=dict(version='V6_RBF',choices_sha256=r.sha(root/'rbf_choices_pretest.json'),selected_models=selected,sources=sources,rbf_candidate_count=420,actual_RBF_neural_gradient_runs=4,
        neural_result_sha256=r.sha(root/'rbf_neural/results.json'),neural_best_validation=nn['best'],test_case_ids=te,test_status='Already seen in V1-V5, retrospective reused-test, not fresh blind',
        selection='RBF recipes fixed by full original validation9 before test loading; new Gaussian NN diagnostic comparison does not replace these RBF recipes',
        fitting_models='V3/V5 full60 fitting unchanged; no full60 fitting labels used in V6',test_labels_used_for_training=False,test_labels_used_for_selection=False)
    p=root/'iteration_frozen.json';r.dump(p,frozen);p.with_suffix('.json.sha256').write_text(r.sha(p)+'\n')
    out=root/'evaluation';out.mkdir(exist_ok=False);results=[];checks=[]
    for region in ['rotor','stator','airgap']:
        source=r.WORK/'new60_20261014/cache_v1'/f'{region}.npz'
        with np.load(source,allow_pickle=False) as z:c=z['conditions'];geometry={k:z[k] for k in ['point_id','xy','theta']}
        # Recheck old validation predictions before loading numerical test rows.
        study=root/choice['regions'][region]['study'];val=runtime.predict(region,c[va]);expected=choice['regions'][region]['recipe']['metrics']
        with np.load(study/'validation_predictions.npz',allow_pickle=False) as z:vy=z['truth']
        metric=r.metrics(val,vy);assert abs(metric['rmse_T']-expected['rmse_T'])<1e-10
        checks.append(dict(region=region,validation_replay_rmse_delta_T=abs(metric['rmse_T']-expected['rmse_T'])))
        signed=r.rows(source,te);truth=signed if region=='airgap' else np.hypot(signed[...,0::2],signed[...,1::2]);predictions={}
        for scope in ['original42','final51']:
            pred=runtime.predict(region,c[te],scope);base=runtime.old_baseline(region,c[te],scope);met=r.metrics(pred,truth);before=r.metrics(base,truth)
            percase=[dict(case_id=int(cid),**r.metrics(pred[j:j+1],truth[j:j+1])) for j,cid in enumerate(te)]
            i,f,pid=met['max_abs_indices'];met['maximum_error_location']=dict(case_id=te[i],frame_index=f,point_id=int(geometry['point_id'][pid]),reference_xy_mm=geometry['xy'][pid].tolist(),prediction_T=float(pred[i,f,pid]),truth_T=float(truth[i,f,pid]))
            result=dict(region=region,training_scope=scope,quantity='signed_Br' if region=='airgap' else 'planar_magnitude',metrics=met,baseline=before,per_case=percase,relative_L2_change_pct=met['relative_L2_pct']-before['relative_L2_pct'],
                training_note='RBF51 branch + V5 baseline51' if scope=='final51' and region=='rotor' else ('RBF51 branch + original42 baseline branch' if scope=='final51' else 'Both branches use only original42 training labels'),
                negative_magnitude_count=int(np.sum(pred<0)) if region!='airgap' else None)
            results.append(result);predictions[scope]=pred;predictions['baseline_'+scope]=base
            print(region,scope,'RBF',met['relative_L2_pct'],'baseline',before['relative_L2_pct'],'max',met['max_abs_T'],flush=True)
        np.savez_compressed(out/(region+'_reused_test.npz'),truth=truth,case_ids=np.array(te),**predictions,**geometry)
    report=dict(version='V6_RBF',frozen_sha256=r.sha(root/'iteration_frozen.json'),results=results,validation_replay=checks,previous_test_already_seen=True,recipe_unchanged_after_test=True,elapsed_seconds=time.perf_counter()-start)
    r.dump(out/'final_results.json',report)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
