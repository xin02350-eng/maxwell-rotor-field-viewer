"""Independent saved-weight replay of all eight V4 NN candidates; no test labels.

Only the existing train42/validation9 mean cache and validation-only full-field
truth are opened. Inference is checked with Torch/sklearn imports prohibited and
np.load restricted to each candidate's weight file. No optimization takes place.
"""
from pathlib import Path
import builtins
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREP=ROOT/'work/new60_opt_v3/joint_neural/point_prepared'
TRUTH=ROOT/'work/new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared/validation_truth.npz'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()

checks=[]
def check(name,ok,detail=None):
    checks.append(dict(name=name,passed=bool(ok),detail=detail))
    if not ok:raise AssertionError(name+': '+str(detail))

def close(name,a,b,atol=1e-12,rtol=1e-12):
    a=np.asarray(a);b=np.asarray(b)
    check(name,a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol),
          dict(max_difference=float(np.max(np.abs(a-b))) if a.shape==b.shape else None,atol=atol,rtol=rtol))

def independent_features(c,scale,quadratic):
    x=c[:,0]/scale*np.cos(c[:,1]*math.pi/180)
    y=c[:,0]/scale*np.sin(c[:,1]*math.pi/180)
    return np.stack((x,y,x*x,x*y,y*y),axis=1) if quadratic else np.stack((x,y),axis=1)

def independent_point_forward(m,c,point,quadratic):
    f=independent_features(c,float(m['current_scale']),quadratic)
    x=(f-m['feature_mean'])/m['feature_scale']
    out=[]
    for row in x:
        response=m['linear_bias'][point].copy()
        for j,v in enumerate(row):response+=v*m['linear_weight'][point,j]
        for k in range(m['gate_bias'].shape[1]):
            z=float(m['gate_bias'][point,k])
            for j in range(2):z+=float(row[j]*m['gate_weight'][point,j,k])
            kind=str(m['expert_kind'])
            if kind=='sigmoid_affine':
                g=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
                e=m['expert_weight'][point,k,0].copy()
                for j in range(2):e+=row[j]*m['expert_weight'][point,k,j+1]
            else:
                if kind=='tanh_constant':g=math.tanh(z)
                else:g=(z if z>20 else max(z,0)+math.log1p(math.exp(-abs(z))))/float(m['initial_sharpness'])
                e=m['expert_weight'][point,k]
            response+=g*e
        out.append(response*m['output_scale'][point]+m['output_mean'][point])
    return np.asarray(out)

def independent_metrics(p,y,ids,point_ids):
    e=p-y
    squared=np.square(e)
    energy=float(np.square(y).sum())
    vals=dict(relative_L2_pct=100*math.sqrt(float(squared.sum())/energy),
              rmse_T=math.sqrt(float(squared.mean())),mae_T=float(np.abs(e).mean()),
              max_abs_T=float(np.abs(e).max()))
    se=squared.sum(axis=(1,2));rm=np.sqrt(se/(60*1800))
    w=int(np.argmax(rm));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape)
    vals.update(worst_case_id=int(ids[w]),worst_case_rmse_T=float(rm[w]),
        max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),
            point_id=int(point_ids[loc[2]]),prediction_T=float(p[loc]),truth_T=float(y[loc]),signed_error_T=float(e[loc])),
        per_case=[dict(case_id=int(cid),rmse_T=float(rr),relative_L2_pct=100*math.sqrt(float(ss)/float(np.square(y[i]).sum())))
                  for i,(cid,rr,ss) in enumerate(zip(ids,rm,se))])
    return vals

def import_numpy_source(path):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise ImportError('Disallowed training library during replay: '+name)
        return original(name,*args,**kwargs)
    builtins.__import__=guarded
    try:
        spec=importlib.util.spec_from_file_location('v4_numpy_'+path.stem,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:builtins.__import__=original
    return module

def guarded_infer(mod,weight,*args,**kwargs):
    original_load=np.load;original_import=builtins.__import__;opened=[]
    def load(path,*a,**k):
        actual=Path(path).resolve();opened.append(str(actual))
        if actual!=weight.resolve():raise AssertionError('Inference opened non-weight NPZ: '+str(actual))
        return original_load(path,*a,**k)
    def imp(name,*a,**k):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise ImportError(name)
        return original_import(name,*a,**k)
    np.load=load;builtins.__import__=imp
    try:result=mod.predict(weight,*args,**kwargs)
    finally:np.load=original_load;builtins.__import__=original_import
    check(weight.parent.name+'/inference_weights_only',bool(opened) and set(opened)=={str(weight.resolve())})
    return result

def main():
    identity=json.loads((PREP/'identity.json').read_text(encoding='utf-8'))
    a=read_npz(PREP/'mean_data.npz');geo=read_npz(PREP/'geometry.npz');base=read_npz(PREP/'baseline_model.npz')
    truth_data=read_npz(TRUTH);truth=truth_data['truth'];ids=truth_data['case_ids'];conditions=a['conditions'][42:]
    check('label_access_exact_train42_plus_val9',identity['label_access']==identity['train_case_ids']+identity['validation_case_ids'])
    check('train42_val9_excludedtest9_disjoint',len(identity['train_case_ids'])==42 and len(ids)==9 and not(set(identity['label_access'])&set(identity['excluded_case_ids'])))
    check('prepared_no_test_no_full60fit',not identity['test_labels_loaded'] and not identity['test_metrics_computed'] and not identity['full60_fit_labels_loaded'])
    check('validation_truth_case_order',ids.tolist()==identity['validation_case_ids']==a['case_ids'][42:].tolist())
    check('full_validation_shape',truth.shape==(9,60,1800))
    check('all1800_point_ids_unique',len(set(geo['point_id'].tolist()))==1800)
    check('baseline_weight_identity',sha(PREP/'baseline_model.npz')==identity['baseline_model_sha256'])
    source_paths=[HERE/'regime_point_nn.py',HERE/'regime_point_nn_softplus_consistent.py',HERE/'regime_quadratic_skip_nn.py']
    sources={sha(p):p for p in source_paths};modules={h:import_numpy_source(p) for h,p in sources.items()}
    result_paths=sorted((HERE/'runs').glob('*/result.json'))
    failed_paths=sorted((HERE/'runs').glob('*/failed.json'))
    check('completed_candidate_count_8',len(result_paths)==8)
    check('failed_attempt_count_2',len(failed_paths)==2)
    candidates=[];saved={}
    frozen_parameters=sum(v.size for k,v in base.items() if k.startswith(('w','b')) and k[1:].isdigit())
    for result_path in result_paths:
        run=result_path.parent;name=run.name;r=json.loads(result_path.read_text(encoding='utf-8'));weight=run/'model.npz';m=read_npz(weight);meta=json.loads(str(m['metadata_json']))
        prefix=name+'/'
        check(prefix+'weight_sha',sha(weight)==(run/'model.npz.sha256').read_text().strip()==r['model_sha256'])
        check(prefix+'source_bound_actual',r['source_sha256']==meta['source_sha256'] and r['source_sha256'] in sources)
        mod=modules[r['source_sha256']];quadratic='linear_feature_kind' in m
        for key in ('train_case_ids','validation_case_ids','excluded_case_ids'):check(prefix+key,meta[key]==identity[key])
        for key in ('test_labels_loaded','test_metrics_computed','full60_fit_labels_loaded'):check(prefix+key,not r[key] and not meta[key])
        check(prefix+'prepared_identity_sha',meta['prepared_identity_sha256']==sha(PREP/'identity.json'))
        check(prefix+'prepared_mean_sha',meta['prepared_means_sha256']==sha(PREP/'mean_data.npz'))
        check(prefix+'uniform_full1800_loss',meta['point_weighting']=='uniform all1800')
        check(prefix+'full_saved_frame_scope',int(m['frame_count'])==60 and int(m['point_count'])==1800)
        for key in geo:check(prefix+'geometry_'+key,np.array_equal(m[key],geo[key]))
        embedded={k[9:]:v for k,v in m.items() if k.startswith('baseline_')}
        check(prefix+'baseline_keys',set(embedded)==set(base))
        for key in base:check(prefix+'baseline_'+key,np.array_equal(embedded[key],base[key]))
        check(prefix+'dynamic_trainonly_scale',float(m['dynamic_scale'])==float(a['signed_dynamic_scale'])==1.)
        f=independent_features(a['conditions'],float(a['conditions'][:42,0].max()),quadratic)
        close(prefix+'current_scale_trainonly',m['current_scale'],np.asarray(a['conditions'][:42,0].max()),atol=0,rtol=0)
        close(prefix+'feature_mean_trainonly',m['feature_mean'],f[:42].mean(0))
        close(prefix+'feature_scale_trainonly',m['feature_scale'],np.maximum(f[:42].std(0),1e-8))
        target=a['signed_mean'][:42];cm=target.mean(0);sc=target.std(0);sc=np.maximum(sc,sc.max()*.001)
        close(prefix+'output_mean_trainonly',m['output_mean'],cm,atol=0,rtol=0)
        close(prefix+'output_scale_trainonly',m['output_scale'],sc,atol=0,rtol=0)
        param_count=sum(m[k].size for k in ('linear_weight','linear_bias','gate_weight','gate_bias','expert_weight'))
        check(prefix+'actual_parameter_count',param_count==r['new_trainable_parameters']==1800*r['new_parameters_per_point'])
        check(prefix+'gate_count',m['gate_bias'].shape[1]==r['configuration']['gates'])
        p=guarded_infer(mod,weight,conditions)
        signed=guarded_infer(mod,weight,conditions,return_signed=True).reshape(9,60,1800,2)
        mean=guarded_infer(mod,weight,conditions,return_mean=True)
        check(prefix+'finite_raw_magnitude',p.shape==truth.shape and np.isfinite(p).all() and np.all(p>=0))
        check(prefix+'signed_and_mean_shapes',signed.shape==(9,60,1800,2) and mean.shape==(9,1800,2))
        close(prefix+'magnitude_from_signed',p,np.sqrt(signed[...,0]**2+signed[...,1]**2))
        close(prefix+'signed_time_mean',signed.mean(1),mean)
        close(prefix+'frozen_dynamic_replay',signed-mean[:,None],a['validation_signed_dynamic'])
        for frame in (0,17,59):close(prefix+'single_frame_'+str(frame),guarded_infer(mod,weight,conditions,frame_index=frame),p[:,frame])
        close(prefix+'single_condition',guarded_infer(mod,weight,conditions[0]),p[0])
        for point in (0,900,1359,1751,1799):close(prefix+'independent_scalar_point_'+str(point),independent_point_forward(m,conditions,point,quadratic),mean[:,point])
        for bad in (-1,60,1.5):
            try:guarded_infer(mod,weight,conditions,frame_index=bad);ok=False
            except ValueError:ok=True
            check(prefix+'reject_invalid_frame_'+str(bad),ok)
        values=independent_metrics(p,truth,ids,geo['point_id'])
        for key in ('relative_L2_pct','rmse_T','mae_T','max_abs_T','worst_case_rmse_T'):close(prefix+'metric_'+key,np.asarray(values[key]),np.asarray(r['validation'][key]))
        check(prefix+'worst_case',values['worst_case_id']==r['validation']['worst_case_id'])
        check(prefix+'max_abs_location',values['max_abs_location']==r['validation']['max_abs_location'])
        for j,case in enumerate(values['per_case']):
            for key in ('rmse_T','relative_L2_pct'):close(prefix+'percase_'+str(case['case_id'])+'_'+key,np.asarray(case[key]),np.asarray(r['validation']['per_case'][j][key]))
        with (run/'history.csv').open(encoding='utf-8-sig',newline='') as stream:history=list(csv.DictReader(stream))
        check(prefix+'all121_checkpoint_rows',len(history)==121 and [int(h['step']) for h in history]==[0,*range(50,5001,50),*range(5010,5201,10)])
        winner=min(history,key=lambda h:float(h['validation_rmse_T']))
        check(prefix+'checkpoint_val_rmse_winner',int(winner['step'])==r['best_step']==meta['best_step'])
        close(prefix+'checkpoint64_replay',np.asarray(values['rmse_T']),np.asarray(float(winner['validation_rmse_T'])),atol=1e-7,rtol=2e-4)
        check(prefix+'full_5000_Adam_200_LBFGS',r['configuration']['adam_steps']==5000 and r['configuration']['lbfgs_blocks']==20 and int(r['final_adam']['step'])==5000 and int(r['final_optimizer']['step'])==5200)
        check(prefix+'torch64_replay_proof',r['numpy_torch_float64_replay_max_abs_difference_T']<1e-10)
        actual_train_mean=mod.predict_mean(m,a['conditions'][:42]);train_error=100*math.sqrt(float(60*np.square(actual_train_mean-target).sum())/float(a['truth_energy'][:42].sum()))
        close(prefix+'train_mean_metric',np.asarray(train_error),np.asarray(r['train_signed_mean_relative_L2_pct']))
        candidates.append(dict(name=name,model=rel(weight),model_sha256=sha(weight),source=rel(sources[r['source_sha256']]),source_sha256=r['source_sha256'],
            history=rel(run/'history.csv'),history_sha256=sha(run/'history.csv'),history_checkpoint_count=len(history),result=rel(result_path),result_sha256=sha(result_path),
            stdout=rel(HERE/(name+'_stdout.log')),stdout_sha256=sha(HERE/(name+'_stdout.log')),configuration=r['configuration'],
            structure=dict(condition_features=5 if quadratic else 2,skip_degree=2 if quadratic else 1,gates=r['configuration']['gates'],expert_kind=r['configuration']['expert_kind'],
                new_trainable_parameters=param_count,new_parameters_per_point=r['new_parameters_per_point'],frozen_dynamic_neural_parameters=frozen_parameters,
                gating_layer_count=1,output_count_per_point=2,decoder='independent signed Br/Bt mean at all1800 points'),
            validation=values,train_signed_mean_relative_L2_pct=train_error,best_step=r['best_step'],elapsed_seconds=r['elapsed_seconds'],lbfgs_closure_evaluations=r['lbfgs_closure_evaluations'],
            numpy_torch_float64_replay_max_abs_difference_T=r['numpy_torch_float64_replay_max_abs_difference_T'],independent_numpy_replay='passed',
            flags=dict(test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)))
        saved[name]=p.copy()
    best=min(candidates,key=lambda c:c['validation']['rmse_T']);pareto=min(candidates,key=lambda c:c['validation']['max_abs_T'])
    selected=[]
    for role,c in (('best_validation_average',best),('best_validation_maximum',pareto)):
        path=HERE/(role+'_validation_replay.npz')
        np.savez_compressed(path,prediction=saved[c['name']],case_ids=ids,conditions=conditions,point_id=geo['point_id'],theta=geo['theta'],time_s=geo['time_s'],model_sha256=np.asarray(c['model_sha256']))
        selected.append(dict(role=role,name=c['name'],model=c['model'],source=c['source'],validation=c['validation'],validation_prediction=rel(path),validation_prediction_sha256=sha(path)))
    failures=[dict(name=p.parent.name,evidence=rel(p),evidence_sha256=sha(p),data=json.loads(p.read_text(encoding='utf-8'))) for p in failed_paths]
    controls=dict(v2_pure_neural_magnitude=dict(validation_relative_L2_pct=2.643129332697101,validation_max_abs_T=.7859681108862597),
                  v3_pure_neural_global_blend=dict(validation_relative_L2_pct=2.490260003339934,validation_rmse_T=.029588531204457157,validation_max_abs_T=.9511031519122706),
                  qualification='Comparison numbers are fixed earlier validation controls, not re-trained or test-evaluated in this route. V4 single NN mean error does not beat V3 global NN blend; V4 3-gate substantially reduces validation peak.')
    output=dict(version='v4-point-regime-final8-neural-candidate-inventory',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed_verified_gradient_nn_candidates=8,failed_gradient_attempts=2,total_gradient_attempts=10,local_combination_evaluations=0,
        protocol=rel(HERE/'candidate_protocol.json'),protocol_sha256=sha(HERE/'candidate_protocol.json'),last2_revision=rel(HERE/'candidate_protocol_last2_revision.json'),last2_revision_sha256=sha(HERE/'candidate_protocol_last2_revision.json'),
        runtime_repairs=[dict(path=rel(HERE/f),sha256=sha(HERE/f)) for f in ('runtime_repair_binding.json','runtime_softplus_repair_binding.json')],
        train_case_ids=identity['train_case_ids'],validation_case_ids=identity['validation_case_ids'],excluded_case_ids=identity['excluded_case_ids'],
        prepared=dict(path=rel(PREP),identity_sha256=sha(PREP/'identity.json'),mean_data_sha256=sha(PREP/'mean_data.npz'),geometry_sha256=sha(PREP/'geometry.npz'),baseline_sha256=sha(PREP/'baseline_model.npz')),
        truth=dict(path=rel(TRUTH),sha256=sha(TRUTH),shape=list(truth.shape),scalar_count=int(truth.size)),
        sources=[dict(path=rel(p),sha256=h,inference_imports='Python standard library + NumPy only; Torch training import not executed',self_contained=True) for h,p in sources.items()],
        inference=dict(api='predict(load_model(path), native_conditions, frame_index=None, return_signed=False, return_mean=False)',
            condition_format='[I_RMS_A, Thet_deg], native Maxwell angle; IcosThet/IsinThet mathematical features, unverified physical dq mapping',
            returns=dict(default='N×60×1800 planar magnitude T',return_signed='N×60×3600 interleaved signed Br/Bt T',return_mean='N×1800×2 static signed Br/Bt mean T'),
            dependencies=['numpy','threadpoolctl (recommended CPU thread limit)'],single_condition_drops_case_axis=True,
            scope='Only all60 supplied first-cycle saved frames; no arbitrary-time/second-cycle inference claim',unit='T',xy_unit='mm',quantity='sqrt(Br^2+Bt^2), no Bz'),
        candidates=candidates,selected_validation_controls=selected,previous_validation_controls=controls,failed_attempt_details=failures,
        verification=dict(checker=rel(Path(__file__)),checker_sha256=sha(__file__),passed_checks=sum(c['passed'] for c in checks),failed_checks=sum(not c['passed'] for c in checks),report=rel(HERE/'independent_numpy_replay_report.json')),
        flags=dict(test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,val_point_weighting=False,hotspot_cropping=False,posthoc_magnitude_projection=False),
        scope_note='This route never loaded original9 test labels. The overall project has already reused the old test set in earlier rounds, so it is not a new untouched external test. 51-condition deployment refit is a separate agent/add-on and does not change this42/9 model selection.')
    report=dict(version='independent-numpy-original-T-fullfield-replay-v4',checks=checks,passed_checks=sum(c['passed'] for c in checks),failed_checks=sum(not c['passed'] for c in checks),loaded_label_files=[rel(PREP/'mean_data.npz'),rel(TRUTH)],test_labels_loaded=False,full60_fit_labels_loaded=False)
    (HERE/'independent_numpy_replay_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=HERE/'candidate_inventory.json';path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');path.with_suffix('.sha256').write_text(sha(path)+'\n')
    print(json.dumps(dict(completed=8,failed_attempts=2,checks=output['verification']['passed_checks'],best_average=best['name'],best_average_validation=best['validation'],best_max=pareto['name'],best_max_validation=pareto['validation'],inventory_sha256=sha(path)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
