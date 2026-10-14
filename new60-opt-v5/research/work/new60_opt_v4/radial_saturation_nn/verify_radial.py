"""Independent saved-file NumPy QA and deployable two-model radial manifest."""
from pathlib import Path
import builtins
import csv
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

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

checks=[]
def check(name,ok,detail=None):
    checks.append(dict(name=name,passed=bool(ok),detail=detail))
    if not ok:raise AssertionError(name+': '+str(detail))

def close(name,a,b,atol=1e-12,rtol=1e-12):
    a=np.asarray(a);b=np.asarray(b)
    check(name,a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol),float(np.max(np.abs(a-b))) if a.shape==b.shape else 'shape')

def scalar_forward(m,c,p):
    result=[]
    for current,theta in c:
        angle=float(theta)*math.pi/180
        x=np.asarray([current/float(m['current_scale'])*math.cos(angle),current/float(m['current_scale'])*math.sin(angle)])
        x=(x-m['feature_mean'])/m['feature_scale'];h=m['drive_bias'][p].copy()
        for j in range(2):h+=x[j]*m['drive_weight'][p,j]
        if str(m['radial_kind'])=='elliptic':
            stretch=float(m['max_log_stretch'])*math.tanh(float(m['raw_stretch'][p]));shear=float(m['max_shear'])*math.tanh(float(m['raw_shear'][p]));q=[math.exp(stretch)*h[0]+shear*h[1],math.exp(-stretch)*h[1]]
        else:q=h
        rho=math.sqrt(sum(float(v*v) for v in q)+1e-20);raw=float(m['raw_saturation'][p]);bs=(raw if raw>20 else max(raw,0)+math.log1p(math.exp(-abs(raw))))+float(m['minimum_saturation']);z=rho/bs
        result.append(h*math.tanh(z)/z*float(m['point_output_scale'][p]))
    return np.asarray(result)

def metric(pred,truth):
    e=pred-truth
    return dict(relative_L2_pct=float(100*np.sqrt(np.square(e).sum()/np.square(truth).sum())),rmse_T=float(np.sqrt(np.square(e).mean())),mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()))

def mean_metrics(pred,target):
    e=pred-target;normpred=np.sqrt(np.sum(pred*pred,axis=-1));normtarget=np.sqrt(np.sum(target*target,axis=-1));me=normpred-normtarget
    return dict(signed_static_mean_relative_L2_pct=float(100*np.sqrt(np.square(e).sum()/np.square(target).sum())),signed_static_mean_component_rmse_T=float(np.sqrt(np.square(e).mean())),signed_static_mean_component_max_abs_T=float(np.abs(e).max()),norm_of_signed_mean_relative_L2_pct=float(100*np.sqrt(np.square(me).sum()/np.square(normtarget).sum())),norm_of_signed_mean_rmse_T=float(np.sqrt(np.square(me).mean())),norm_of_signed_mean_max_abs_T=float(np.abs(me).max()))

def infer(module,path,c,**kwargs):
    old_import=builtins.__import__;old_load=np.load;opened=[]
    def imp(name,*a,**k):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise ImportError('Training dependency forbidden: '+name)
        return old_import(name,*a,**k)
    def load(file,*a,**k):
        actual=Path(file).resolve();opened.append(str(actual))
        if actual!=path.resolve():raise AssertionError('Non-weight read in inference: '+str(actual))
        return old_load(file,*a,**k)
    builtins.__import__=imp;np.load=load
    try:result=module.predict(path,c,**kwargs)
    finally:builtins.__import__=old_import;np.load=old_load
    check(path.parent.name+'/inference_weights_only',opened==[str(path.resolve())])
    return result

def main():
    source=HERE/'radial_point_nn.py';protocol_path=HERE/'candidate_protocol.json';protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    identity=json.loads((PREP/'identity.json').read_text(encoding='utf-8'));a=read_npz(PREP/'mean_data.npz');geometry=read_npz(PREP/'geometry.npz');base=read_npz(PREP/'baseline_model.npz');truthdata=read_npz(TRUTH);truth=truthdata['truth'];ids=truthdata['case_ids'];conditions=a['conditions'][42:]
    check('source_predeclared_hash',sha(source)==protocol['source_sha256'])
    check('protocol_hash_sidecar',sha(protocol_path)==(HERE/'candidate_protocol.sha256').read_text().strip())
    check('train42_val9_only',len(identity['train_case_ids'])==42 and len(identity['validation_case_ids'])==9 and identity['label_access']==identity['train_case_ids']+identity['validation_case_ids'] and not(set(identity['label_access'])&set(identity['excluded_case_ids'])))
    check('no_test_no_fullfit_provenance',not identity['test_labels_loaded'] and not identity['full60_fit_labels_loaded'])
    for key in ('train_case_ids','validation_case_ids','excluded_case_ids'):check('protocol_'+key,protocol[key]==identity[key])
    check('truth_fullfield',truth.shape==(9,60,1800) and ids.tolist()==identity['validation_case_ids'])
    check('prepared_identity_binding',sha(PREP/'identity.json')==protocol['prepared_identity_sha256'])
    check('prepared_mean_binding',sha(PREP/'mean_data.npz')==protocol['prepared_means_sha256'])
    check('prepared_geometry_binding',sha(PREP/'geometry.npz')==protocol['geometry_sha256'])
    check('old_train42_baseline_binding',sha(PREP/'baseline_model.npz')==protocol['baseline_model_sha256'])
    check('validation_truth_binding',sha(TRUTH)==protocol['validation_truth_sha256'])
    frozen=ROOT/'work/new60_opt_v4/point_regime_nn/frozen_directory_manifest.json';snapshot=json.loads(frozen.read_text(encoding='utf-8'))
    check('original8_manifest_immutable',sha(frozen)==protocol['frozen_directory_manifest_sha256'])
    for entry in snapshot['files']:check('original8_file_immutable/'+entry['path'],sha(frozen.parent/entry['path'])==entry['sha256'])
    old_import=builtins.__import__
    def imp(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise ImportError(name)
        return old_import(name,*args,**kwargs)
    builtins.__import__=imp
    try:
        spec=importlib.util.spec_from_file_location('radial_numpy_checked',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:builtins.__import__=old_import
    results=sorted((HERE/'runs').glob('*/result.json'));failures=sorted((HERE/'runs').glob('*/failed.json'))
    check('two_completed_candidates',len(results)==2);check('zero_failed_attempts',len(failures)==0)
    models=[];predictions={}
    for result_path in results:
        run=result_path.parent;name=run.name;prefix=name+'/';r=json.loads(result_path.read_text(encoding='utf-8'));path=run/'model.npz';m=read_npz(path);metadata=json.loads(str(m['metadata_json']))
        check(prefix+'model_sha',sha(path)==r['model_sha256']==(run/'model.npz.sha256').read_text().strip())
        check(prefix+'source_sha',sha(source)==r['source_sha256']==metadata['source_sha256'])
        check(prefix+'protocol_sha',sha(protocol_path)==r['protocol_sha256']==metadata['protocol_sha256'])
        check(prefix+'configuration_exact_predeclared',r['configuration']==metadata['configuration'] and r['configuration'] in protocol['candidates'])
        for key in ('train_case_ids','validation_case_ids','excluded_case_ids'):check(prefix+key,metadata[key]==identity[key])
        for key in ('test_labels_loaded','test_metrics_computed','full60_fit_labels_loaded'):check(prefix+key,not r[key] and not metadata[key])
        check(prefix+'uniform1800_points',metadata['point_weighting']=='uniform all1800')
        for key,value in geometry.items():check(prefix+'geometry_'+key,np.array_equal(m[key],value))
        embedded={k[9:]:v for k,v in m.items() if k.startswith('baseline_')};check(prefix+'baseline_keys',set(embedded)==set(base))
        for key,value in base.items():check(prefix+'baseline_'+key,np.array_equal(embedded[key],value))
        check(prefix+'frozen_dynamic_scale',float(m['dynamic_scale'])==float(a['signed_dynamic_scale'])==1.)
        current_scale=float(a['conditions'][:42,0].max());angles=a['conditions'][:,1]*np.pi/180;rr=a['conditions'][:,0]/current_scale;f=np.column_stack((rr*np.cos(angles),rr*np.sin(angles)))
        close(prefix+'current_scale_trainonly',m['current_scale'],np.asarray(current_scale),atol=0,rtol=0)
        close(prefix+'feature_mean_trainonly',m['feature_mean'],f[:42].mean(0));close(prefix+'feature_std_trainonly',m['feature_scale'],np.maximum(f[:42].std(0),1e-8))
        scale=np.sqrt(np.mean(np.sum(a['signed_mean'][:42]**2,axis=-1),axis=0));scale=np.maximum(scale,scale.max()*.001);close(prefix+'output_energy_scale_trainonly',m['point_output_scale'],scale,atol=0,rtol=0)
        keys=['drive_weight','drive_bias','raw_saturation']+(['raw_stretch','raw_shear'] if str(m['radial_kind'])=='elliptic' else [])
        count=sum(m[k].size for k in keys);expected=9 if str(m['radial_kind'])=='elliptic' else 7
        check(prefix+'actual_parameter_count',count==expected*1800==r['new_trainable_parameters'] and expected==r['new_parameters_per_point'])
        p=infer(module,path,conditions);signed=infer(module,path,conditions,return_signed=True).reshape(9,60,1800,2);mean=infer(module,path,conditions,return_mean=True)
        check(prefix+'finite_allframes_unprojected',p.shape==(9,60,1800) and mean.shape==(9,1800,2) and np.isfinite(p).all() and np.all(p>=0))
        close(prefix+'hypot_from_signed',p,np.sqrt(np.sum(signed*signed,axis=-1)));close(prefix+'time_mean',signed.mean(1),mean);close(prefix+'dynamic_replay',signed-mean[:,None],a['validation_signed_dynamic'])
        for frame in (0,17,59):close(prefix+'single_frame'+str(frame),infer(module,path,conditions,frame_index=frame),p[:,frame])
        close(prefix+'single_condition',infer(module,path,conditions[0]),p[0])
        for point in (0,900,1359,1751,1799):close(prefix+'scalar_forward_point'+str(point),scalar_forward(m,conditions,point),mean[:,point])
        for bad in (-1,60,1.5):
            try:infer(module,path,conditions,frame_index=bad);ok=False
            except ValueError:ok=True
            check(prefix+'reject_frame'+str(bad),ok)
        bs=np.where(m['raw_saturation']>20,m['raw_saturation'],np.logaddexp(0,m['raw_saturation']))+float(m['minimum_saturation'])
        check(prefix+'positive_saturation',np.all(bs>0) and np.isfinite(bs).all())
        normalized_mean=mean/m['point_output_scale'][None,:,None]
        if str(m['radial_kind'])=='elliptic':
            s=float(m['max_log_stretch'])*np.tanh(m['raw_stretch']);aa=float(m['max_shear'])*np.tanh(m['raw_shear']);q0=np.exp(s)[None,:]*normalized_mean[...,0]+aa[None,:]*normalized_mean[...,1];q1=np.exp(-s)[None,:]*normalized_mean[...,1];rho=np.sqrt(q0*q0+q1*q1)
            close(prefix+'ellipse_determinant_one',np.exp(s)*np.exp(-s),np.ones(1800));check(prefix+'bounded_ellipse_parameters',np.all(np.abs(s)<float(m['max_log_stretch'])) and np.all(np.abs(aa)<float(m['max_shear'])))
        else:rho=np.linalg.norm(normalized_mean,axis=-1)
        check(prefix+'radial_output_saturation_bound',np.all(rho<=bs[None,:]+1e-12))
        empty=dict(m);empty['drive_weight']=np.zeros_like(m['drive_weight']);empty['drive_bias']=np.zeros_like(m['drive_bias']);zero=module.predict_mean(empty,conditions);close(prefix+'zero_drive_finite_limit',zero,np.zeros((9,1800,2)),atol=0,rtol=0)
        values=metric(p,truth)
        for key,value in values.items():close(prefix+'fullval_'+key,np.asarray(value),np.asarray(r['validation'][key]))
        e=p-truth;rm=np.sqrt(np.square(e).mean(axis=(1,2)));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape)
        check(prefix+'worstcase',int(ids[np.argmax(rm)])==r['validation']['worst_case_id']);check(prefix+'maxlocation',dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),point_id=int(geometry['point_id'][loc[2]]),prediction_T=float(p[loc]),truth_T=float(truth[loc]),signed_error_T=float(e[loc]))==r['validation']['max_abs_location'])
        for i,record in enumerate(r['validation']['per_case']):
            check(prefix+'caseid'+str(i),record['case_id']==int(ids[i]));close(prefix+'percase_rmse'+str(i),np.asarray(rm[i]),np.asarray(record['rmse_T']));close(prefix+'percase_L2'+str(i),np.asarray(metric(p[i],truth[i])['relative_L2_pct']),np.asarray(record['relative_L2_pct']))
        train_mean=module.predict_mean(m,a['conditions'][:42])
        for key,value in mean_metrics(train_mean,a['signed_mean'][:42]).items():close(prefix+'train_'+key,np.asarray(value),np.asarray(r['train_static_mean'][key]))
        for key,value in mean_metrics(mean,a['signed_mean'][42:]).items():close(prefix+'validation_'+key,np.asarray(value),np.asarray(r['validation_static_mean'][key]))
        with (run/'history.csv').open(encoding='utf-8-sig',newline='') as stream:history=list(csv.DictReader(stream))
        check(prefix+'121_complete_checkpoints',[int(row['step']) for row in history]==[0,*range(50,5001,50),*range(5010,5201,10)])
        selected=min(history,key=lambda row:float(row['validation_rmse_T']));check(prefix+'minimum_validation_RMSE_step',int(selected['step'])==r['best_step']==metadata['best_step']);close(prefix+'checkpoint_float64',np.asarray(values['rmse_T']),np.asarray(float(selected['validation_rmse_T'])),atol=1e-7,rtol=2e-4)
        check(prefix+'full_training_budget',r['final_adam']['step']==5000 and r['final_optimizer']['step']==5200 and len(history)==r['history_checkpoint_count']==121)
        check(prefix+'torch64_numpy_proof',r['numpy_torch_float64_replay_max_abs_difference_T']<1e-10)
        models.append(dict(id=name,model_relative_path=path.relative_to(HERE).as_posix(),inference_module_relative_path=source.relative_to(HERE).as_posix(),model_sha256=sha(path),inference_module_sha256=sha(source),predict_kwargs={},signed_output_kwargs=dict(return_signed=True),mean_output_kwargs=dict(return_mean=True),validation=r['validation'],train_static_mean=r['train_static_mean'],validation_static_mean=r['validation_static_mean'],configuration=r['configuration'],new_parameters_per_point=expected,new_trainable_parameters=count,best_step=r['best_step'],history_relative_path=(run/'history.csv').relative_to(HERE).as_posix(),history_sha256=sha(run/'history.csv'),history_checkpoint_count=len(history),result_relative_path=result_path.relative_to(HERE).as_posix(),result_sha256=sha(result_path),stdout_relative_path=name+'_stdout.log',stdout_sha256=sha(HERE/(name+'_stdout.log')),numpy_torch_float64_replay_max_abs_difference_T=r['numpy_torch_float64_replay_max_abs_difference_T']))
        predictions[name]=p.copy()
    best=min(models,key=lambda m:m['validation']['rmse_T']);cache=HERE/'best_validation_replay.npz';np.savez_compressed(cache,prediction=predictions[best['id']],case_ids=ids,conditions=conditions,point_id=geometry['point_id'],theta=geometry['theta'],time_s=geometry['time_s'],model_sha256=np.asarray(best['model_sha256']))
    report=dict(passed_checks=sum(c['passed'] for c in checks),failed_checks=sum(not c['passed'] for c in checks),checks=checks,label_paths_opened=[str(PREP/'mean_data.npz'),str(TRUTH)],checker_sha256=sha(__file__),test_labels_loaded=False,full60_fit_labels_loaded=False)
    qa=HERE/'independent_numpy_replay_report.json';qa.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest=dict(version='v4-extra2-radial-neural-completed-deployment-manifest',models=models,completed_new_neural_trainings=2,failed_attempts_excluded=[],cumulative_V4_completed_NN42_including_frozen_regime8=10,previous_regime_failed_gradient_attempts=2,local_blend_evaluations=0,
        protocol_relative_path='candidate_protocol.json',protocol_sha256=sha(protocol_path),train_case_ids=identity['train_case_ids'],validation_case_ids=identity['validation_case_ids'],excluded_case_ids=identity['excluded_case_ids'],full_validation_shape=[9,60,1800],
        inference=dict(api='predict(model_or_path, native_conditions, frame_index=None, return_signed=False, return_mean=False)',default_shape='C60x1800 magnitude T',signed_shape='C60x3600 Br/Bt interleaved T',mean_shape='C1800x2 static signed mean T',dependencies=['numpy','threadpoolctl recommended for CPU2threads'],self_contained_numpy_module=True,saved_frames_only=True,first_cycle_only=True,xy_units='mm',quantity='sqrt(Br^2+Bt^2), no Bz'),
        independent_numpy_QA=dict(report_relative_path=qa.relative_to(HERE).as_posix(),report_sha256=sha(qa),checker_relative_path='verify_radial.py',checker_sha256=sha(__file__),passed_checks=report['passed_checks'],failed_checks=report['failed_checks'],prior_regime8_unchanged=True),
        best_validation_cache=dict(model_id=best['id'],relative_path=cache.relative_to(HERE).as_posix(),sha256=sha(cache)),
        interpretation='Two completed physically coupled low-capacity radial NN controls; comparison and primary/global blends are handled by root. No claim these results meet very-small-error goal. Static signed-mean norm metrics are distinct from fullframe magnetic magnitude metrics.',
        test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,validation_point_weighting=False,hotspot_cropping=False,posthoc_magnitude_projection=False)
    path=HERE/'final_manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(HERE/'final_manifest.sha256').write_text(sha(path)+'\n')
    print(json.dumps(dict(models=[dict(id=m['id'],val_L2=m['validation']['relative_L2_pct'],max_T=m['validation']['max_abs_T']) for m in models],checks=report['passed_checks'],failed_checks=report['failed_checks'],manifest_sha256=sha(path)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
