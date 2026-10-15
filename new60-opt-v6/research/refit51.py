"""Freeze RBF recipes using old validation only; refit RBF branches on51.
The rotor baseline is its existing51 NN+kernel blend. Stator and airgap retain
their42 baseline branches, so those are explicitly mixed51/42-branch models.
"""
from pathlib import Path
import json,time,numpy as np
from threadpoolctl import threadpool_limits
import rbf_pointwise as r

def run():
    start=time.perf_counter();choices={}
    for region,studies in [('rotor',['rotor_signed','rotor_magnitude']),('stator',['stator_signed','stator_magnitude']),('airgap',['airgap_Br'])]:
        opts=[(s,json.loads((r.ROOT/s/'results.json').read_text())) for s in studies];name,result=min(opts,key=lambda x:x[1]['selected']['metrics']['rmse_T'])
        choices[region]=dict(study=name,recipe=result['selected'],shared_index=result['shared_index'],protocol_sha256=r.sha(r.ROOT/name/'protocol.json'),results_sha256=r.sha(r.ROOT/name/'results.json'))
    frozen=dict(regions=choices,selection='Full original9 validation RMSE, reused validation; no test read',original42_train_case_ids=json.loads((r.WORK/'new60_20261014/cache_v1/split.json').read_text())['train'],
        source_sha256=r.sha(__file__),rbf_source_sha256=r.sha(r.__file__),runtime_source_sha256=r.sha(r.ROOT/'inference_v6.py'),final51_policy='same chosen shared config, perpoint conditional trainLOO routing recomputed on51; same global blend weights; no independent validation',
        stator_airgap_final51_baseline='original42 baseline retained, only new RBF branch trained on51; rotor baseline is existing51',test_labels_read=False)
    p=r.ROOT/'rbf_choices_pretest.json';r.dump(p,frozen);p.with_suffix('.json.sha256').write_text(r.sha(p)+'\n')
    split=json.loads((r.WORK/'new60_20261014/cache_v1/split.json').read_text());ids=split['train']+split['validation'];dest=r.ROOT/'final51';dest.mkdir(exist_ok=False)
    for region,choice in choices.items():
        out=dest/region;out.mkdir();study=r.ROOT/choice['study'];protocol=json.loads((study/'protocol.json').read_text());specs=protocol['specs'];cache=r.WORK/'new60_20261014/cache_v1'/f'{region}.npz'
        with np.load(cache,allow_pickle=False) as z:c=z['conditions'][ids];geometry={k:z[k] for k in ['point_id','xy','theta']}
        data=r.rows(cache,ids).reshape(51,60,-1,1 if region=='airgap' else 2);route=protocol['route']
        if route=='magnitude':data=np.hypot(data[...,0],data[...,1])[...,None]
        m,target=r.prepare_geometry(c,data,0.);sat,_=r.prepare_geometry(c,data,.7)
        errors=np.full(target.shape[0],np.inf);selected=np.zeros(target.shape[0],dtype=int)
        # If only the globally selected shared model is needed, no rerouting.
        needed=range(len(specs)) if choice['recipe']['route']!='shared' else [choice['shared_index']]
        for i in needed:
            spec=specs[i];_,loo,_=r.solve_candidate(sat if spec['cap'] else m,c,target,c[:1],spec);mask=loo<errors;errors[mask]=loo[mask];selected[mask]=i
            if (i+1)%12==0:print('refit51',region,i+1,len(specs),flush=True)
        hashes={}
        for name,selection in [('shared',np.full(target.shape[0],choice['shared_index'])),('pointLOO',selected)]:
            if name=='pointLOO' and choice['recipe']['route']=='shared':continue
            arrays={k:geometry[k] for k in ['point_id','xy','theta']};groups=[]
            for g,index in enumerate(sorted(set(selection.tolist()))):
                spec=specs[index];points=np.flatnonzero(selection==index);geom=dict(sat if spec['cap'] else m);geom['rotation']=geom['rotation'][points]
                _,_,sol=r.solve_candidate(geom,c,target[points],c[:1],spec);pre=f'g{g}_';arrays.update({pre+k:v for k,v in geom.items()});arrays.update({pre+'points':points,pre+'centers':r.mapped(geom,c,spec['anisotropy']),pre+'weights':sol[:,:51],pre+'trend':sol[:,51:]});groups.append(spec)
            arrays['metadata_json']=np.array(json.dumps(dict(format='v6-pointwise-rbf-grouped',groups=groups,region=region,route=route,train_case_ids=ids,source_sha256=r.sha(__file__),rbf_source_sha256=r.sha(r.__file__),gradient_neural_training=False,recipe_choices_sha256=r.sha(p),independent_validation=False)))
            model=out/(name+'.npz');np.savez_compressed(model,**arrays);model.with_suffix('.npz.sha256').write_text(r.sha(model)+'\n');hashes[name]=r.sha(model)
        r.dump(out/'refit.json',dict(training_case_ids=ids,model_hashes=hashes,original42_recipe=choice,test_labels_read=False,independent_validation=False,actual_new_NN_training=False))
    r.dump(dest/'summary.json',dict(status='PASS',regions=list(choices),recipe_choices_sha256=r.sha(p),test_labels_read=False,elapsed_seconds=time.perf_counter()-start))

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
