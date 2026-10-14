"""Independent NumPy train51 replay; does not decode any originaltest9 label."""
from pathlib import Path
import builtins
import csv
import importlib.util
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
import qa_primitives as q

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREP=ROOT/'work/new60_opt_v3/joint_neural/point_prepared'
CACHE=ROOT/'work/new60_20261014/cache_v1/rotor.npz'

def read51(ids):
    mapping={cid:i for i,cid in enumerate(ids)};out=np.empty((51,60,3600),dtype=np.float64);interpreted=[]
    with zipfile.ZipFile(CACHE) as archive,archive.open('field.npy') as stream:
        version=np.lib.format.read_magic(stream);shape,fortran,dtype=np.lib.format._read_array_header(stream,version);q.check('selected51_source_header',shape==(60,60,3600) and not fortran and dtype.kind=='f');block=60*3600*dtype.itemsize
        for cid in range(60):
            if cid in mapping:
                raw=stream.read(block);q.check('selected51_row_bytes/'+str(cid),len(raw)==block);out[mapping[cid]]=np.frombuffer(raw,dtype=dtype).reshape(60,3600);interpreted.append(cid)
            else:stream.seek(block,1)
    q.check('allandonly51_numericlabels',interpreted==sorted(ids) and len(interpreted)==51)
    return out.reshape(51,60,1800,2)

def main():
    sha=q.sha;pp=HERE/'candidate_protocol.json';protocol=json.loads(pp.read_text(encoding='utf-8'));identity=json.loads((PREP/'identity.json').read_text(encoding='utf-8'));a=q.read_npz(PREP/'mean_data.npz');geo=q.read_npz(PREP/'geometry.npz');base=q.read_npz(PREP/'baseline_model.npz');ids=identity['train_case_ids']+identity['validation_case_ids'];c=a['conditions'];source=HERE/'magnitude_final51.py';trainer=HERE/'fine_tune_train51.py'
    q.check('one_predeclared_candidate',protocol['candidate']['gates']==2 and protocol['candidate']['lambda_signed']==.02 and protocol['candidate']['adam_steps']==2000 and protocol['candidate']['lbfgs_blocks']==10)
    q.check('protocol_sidecar',sha(pp)==(HERE/'candidate_protocol.sha256').read_text().strip());q.check('inference_source_predeclared',sha(source)==protocol['inference_source_sha256']);q.check('trainer_source_predeclared',sha(trainer)==protocol['training_source_sha256']);q.check('cache_sha_identity',sha(CACHE)==protocol['raw_cache_sha256']==identity['cache_sha256']);q.check('prepared_identity',sha(PREP/'identity.json')==protocol['prepared_identity_sha256']);q.check('prepared_means',sha(PREP/'mean_data.npz')==protocol['prepared_means_sha256'])
    q.check('51_training_disjoint_test9',len(ids)==51 and len(set(ids))==51 and not(set(ids)&set(identity['excluded_case_ids'])) and protocol['training_case_ids']==ids==a['case_ids'].tolist());q.check('no_independentval',protocol['validation_case_ids']==[]);q.check('fixed25_75',protocol['fixed_final51_recipe']['old_V4_pure51_scalar_norm_weight']==.25 and protocol['fixed_final51_recipe']['new_this51_scalar_norm_weight']==.75)
    frozen=ROOT/'work/new60_opt_v5/magnitude_objective_nn/frozen_directory_manifest.json';old=json.loads(frozen.read_text(encoding='utf-8'));q.check('old42_four_manifest_immutable',sha(frozen)==protocol['original_V5_four_frozen_manifest_sha256'])
    for item in old['files']:q.check('old42_four_file_immutable/'+item['path'],sha(frozen.parent/item['path'])==item['sha256'])
    old51=ROOT/'work/new60_opt_v4/final51_refit/final51_models_frozen.json';old51data=json.loads(old51.read_text(encoding='utf-8'));q.check('V4_final51_manifest_immutable',sha(old51)==protocol['warmstart_manifest_sha256'])
    for item in old51data['models']:
        q.check('V4_final51_model_immutable/'+item['id'],sha(old51.parent/item['model_relative_path'])==item['model_sha256']);q.check('V4_final51_source_immutable/'+item['id'],sha(old51.parent/item['inference_module_relative_path'])==item['inference_module_sha256'])
    q.check('V4_pure51_recipeweight_immutable',sha(old51.parent/'fixed_global_blends/pure_nn_model.npz')==protocol['fixed_final51_recipe']['old_V4_pure51_model_sha256'])
    true_signed=read51(ids);truth=np.hypot(true_signed[...,0],true_signed[...,1]);target=true_signed.mean(1);q.close('true51_staticmean',target,a['signed_mean']);q.close('true51_energy',np.square(truth).sum(axis=(1,2)),a['truth_energy'],atol=1e-8)
    orig_import=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise ImportError(name)
        return orig_import(name,*args,**kwargs)
    builtins.__import__=guard
    try:spec=importlib.util.spec_from_file_location('v5_final51_numpy_checked',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    finally:builtins.__import__=orig_import
    results=list((HERE/'runs').glob('*/result.json'));failures=list((HERE/'runs').glob('*/failed.json'));q.check('exactlyone_completed_gradientfinetune',len(results)==1);q.check('zero_failed_attempts',len(failures)==0)
    rp=results[0];run=rp.parent;path=run/'model.npz';r=json.loads(rp.read_text(encoding='utf-8'));m=q.read_npz(path);meta=json.loads(str(m['metadata_json']));config=r['configuration'];warm_path=Path(config['warmstart_model']);warm=q.read_npz(warm_path);warmmeta=json.loads(str(warm['metadata_json']))
    q.check('actual_model_sha',sha(path)==r['model_sha256']==(run/'model.npz.sha256').read_text().strip());q.check('inference_SHA_binding',sha(source)==r['source_sha256']==meta['source_sha256']);q.check('training_SHA_binding',sha(trainer)==r['training_source_sha256']==meta['training_source_sha256']);q.check('protocol_SHA_binding',sha(pp)==r['protocol_sha256']==meta['protocol_sha256']);q.check('actual_exact_config',config==protocol['candidate']==meta['configuration']);q.check('warmstart_model_SHA',sha(warm_path)==config['warmstart_model_sha256']==meta['warmstart_model_sha256']);q.check('warmstart_source_SHA',warmmeta['source_sha256']==config['warmstart_source_sha256']==meta['warmstart_source_sha256']);q.check('warmstart_actual51',warmmeta['train_case_ids']==ids and warmmeta['validation_case_ids']==[] and not warmmeta['test_labels_loaded'] and not warmmeta['full60_fit_labels_loaded'])
    q.check('actual51_training_ids',meta['train_case_ids']==meta['raw_label_rows_loaded']==r['raw_label_rows_loaded']==ids);q.check('former9nowtrain',meta['former_validation_now_training_case_ids']==identity['validation_case_ids']);q.check('no_val_model',meta['validation_case_ids']==[] and r['validation'] is None);q.check('test9_exclusion',meta['excluded_case_ids']==identity['excluded_case_ids']);q.check('real_gradient_notfromscratch',r['new_gradient_fine_tuning_run'] and not r['from_scratch_training'])
    for key in ('test_labels_loaded','test_metrics_computed','full60_fit_labels_loaded'):q.check(key,not r[key] and not meta[key])
    q.check('train51_labels_declared',r['train51_labels_loaded'] and meta['train51_labels_loaded']);q.check('allpointsuniform',meta['point_weighting']=='uniform all1800')
    for key,value in geo.items():q.check('geometry_'+key,np.array_equal(m[key],value))
    for key in ('current_scale','feature_mean','feature_scale','output_mean','output_scale','dynamic_scale'):q.check('warm51_scaler_immutable_'+key,np.array_equal(m[key],warm[key]))
    f=np.column_stack((c[:,0]/c[:,0].max()*np.cos(np.deg2rad(c[:,1])),c[:,0]/c[:,0].max()*np.sin(np.deg2rad(c[:,1]))));q.close('current_scale_allandonlytrain51',m['current_scale'],np.asarray(c[:,0].max()),atol=0,rtol=0);q.close('feature_mean_allandonlytrain51',m['feature_mean'],f.mean(0));q.close('feature_scale_allandonlytrain51',m['feature_scale'],np.maximum(f.std(0),1e-8));q.close('output_mean_allandonlytrain51',m['output_mean'],target.mean(0));sc=target.std(0);sc=np.maximum(sc,sc.max()*.001);q.close('output_scale_allandonlytrain51',m['output_scale'],sc)
    embedded={k[9:]:v for k,v in m.items() if k.startswith('baseline_')};q.check('oldbaseline_keys',set(embedded)==set(base))
    for key,value in base.items():q.check('frozen_originaltrain42_neural_'+key,np.array_equal(embedded[key],value))
    keys=('linear_weight','linear_bias','gate_weight','gate_bias','expert_weight');count=sum(m[k].size for k in keys);q.check('real_parameter_count',count==43200==r['new_trainable_parameters'] and r['new_parameters_per_point']==24 and m['gate_bias'].shape[1]==2)
    p=q.infer(mod,path,c);signed=q.infer(mod,path,c,return_signed=True).reshape(51,60,1800,2);mean=q.infer(mod,path,c,return_mean=True);q.check('finite_unprojected_full51shape',p.shape==(51,60,1800) and np.isfinite(p).all() and np.all(p>=0));q.close('signed_hypot',p,np.sqrt(np.sum(signed*signed,axis=-1)));q.close('temporal_mean',signed.mean(1),mean);q.close('signed_frozen_dynamic',signed-mean[:,None],mod.baseline_predict(base,c).reshape(51,60,1800,2)-mod.baseline_predict(base,c).reshape(51,60,1800,2).mean(1,keepdims=True))
    for frame in (0,17,59):q.close('single_frame_'+str(frame),q.infer(mod,path,c,frame_index=frame),p[:,frame])
    q.close('single_case',q.infer(mod,path,c[0]),p[0])
    for point in (0,900,1359,1751,1799):q.close('scalar_forward_point_'+str(point),q.static_forward(m,c,point),mean[:,point])
    for bad in (-1,60,1.5):
        try:q.infer(mod,path,c,frame_index=bad);ok=False
        except ValueError:ok=True
        q.check('invalid_frame_'+str(bad),ok)
    values=q.field_metrics(p,truth,np.asarray(ids),geo['point_id']);reference=r['train51_fitting_magnitude']
    for key in ('relative_L2_pct','rmse_T','mae_T','max_abs_T','worst_case_rmse_T'):q.close('train51_fullframe_'+key,np.asarray(values[key]),np.asarray(reference[key]))
    q.check('train51_worstcase',values['worst_case_id']==reference['worst_case_id']);q.check('train51_maxlocation',values['max_abs_location']==reference['max_abs_location'])
    for i,record in enumerate(values['per_case']):
        q.check('train51_case_id_'+str(i),record['case_id']==reference['per_case'][i]['case_id'])
        for key in ('rmse_T','relative_L2_pct'):q.close('train51_case_'+str(i)+'_'+key,np.asarray(record[key]),np.asarray(reference['per_case'][i][key]))
    for key,value in q.mean_metrics(mean,target).items():q.close('train51_static_'+key,np.asarray(value),np.asarray(r['train51_static_mean'][key]))
    se=signed-true_signed;sv=dict(relative_L2_pct=float(100*np.sqrt(np.square(se).sum()/np.square(true_signed).sum())),component_rmse_T=float(np.sqrt(np.square(se).mean())),component_max_abs_T=float(np.abs(se).max()))
    for key,value in sv.items():q.close('train51_fullsigned_'+key,np.asarray(value),np.asarray(r['train51_full60_signed_components'][key]))
    mag_mse=float(np.mean((p-truth)**2));regularizer=float(np.mean((mean-target)**2));objective=mag_mse+.02*regularizer
    q.close('selected_objective',np.asarray(objective),np.asarray(r['train51_selected_objective_T2']));q.close('selected_mag_mse',np.asarray(mag_mse),np.asarray(r['train51_selected_magnitude_mse_T2']));q.close('selected_signed_reg',np.asarray(regularizer),np.asarray(r['train51_selected_signed_regularizer_T2']))
    with (run/'history.csv').open(encoding='utf-8-sig',newline='') as stream:history=list(csv.DictReader(stream))
    q.check('all51_complete_checkpoints',[int(h['step']) for h in history]==[0,*range(50,2001,50),*range(2010,2101,10)])
    for h in history:
        q.close('checkpoint_originalT_objective_'+h['step'],np.asarray(float(h['train51_full_objective_T2'])),np.asarray(float(h['train51_full60_magnitude_mse_T2'])+.02*float(h['train51_signed_mean_regularizer_T2'])))
        if h['stage']=='Adam_fixed6frame_cycle':q.check('fixed_frames_'+h['step'],h['frame_batch']==','.join(map(str,range((int(h['step'])-1)%10,60,10))))
    winner=min(history,key=lambda h:float(h['train51_full_objective_T2']));q.check('trainobjective_only_winner',int(winner['step'])==r['best_step']==meta['best_step']);q.close('selectedcheckpoint_float64',np.asarray(objective),np.asarray(float(winner['train51_full_objective_T2'])),atol=1e-9,rtol=2e-4);q.check('actual2000Adam100LBFGS',r['final_adam']['step']==2000 and r['final_optimizer']['step']==2100 and r['history_checkpoint_count']==51);q.check('Torch64_NumPy_proof',r['numpy_torch_float64_replay_max_abs_difference_T']<1e-10)
    report=dict(passed_checks=sum(x['passed'] for x in q.checks),failed_checks=sum(not x['passed'] for x in q.checks),checks=q.checks,raw_numeric_label_rows_loaded=ids,test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,checker_sha256=sha(__file__),primitives_sha256=sha(HERE/'qa_primitives.py'));qa=HERE/'independent_numpy_replay_report.json';qa.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    model=dict(id=r['name'],model_relative_path=path.relative_to(HERE).as_posix(),inference_module_relative_path=source.relative_to(HERE).as_posix(),model_sha256=sha(path),inference_module_sha256=sha(source),training_source_relative_path=trainer.relative_to(HERE).as_posix(),training_source_sha256=sha(trainer),predict_kwargs={},signed_output_kwargs=dict(return_signed=True),mean_output_kwargs=dict(return_mean=True),validation=None,train51_fitting_magnitude=reference,train51_static_mean=r['train51_static_mean'],train51_full60_signed_components=r['train51_full60_signed_components'],training_case_ids=ids,validation_case_ids=[],excluded_test_case_ids=identity['excluded_case_ids'],configuration=config,new_trainable_parameters=count,new_parameters_per_point=24,new_gradient_fine_tuning_run=True,from_scratch_training=False,best_step=r['best_step'],selected_warmstart_step0=r['best_step']==0,trainable_arrays_changed=sum(not np.array_equal(m[k],warm[k]) for k in keys),train51_selected_objective_T2=objective,history_relative_path=(run/'history.csv').relative_to(HERE).as_posix(),history_sha256=sha(run/'history.csv'),history_checkpoint_count=len(history),result_relative_path=rp.relative_to(HERE).as_posix(),result_sha256=sha(rp),stdout_relative_path=r['name']+'_stdout.log',stdout_sha256=sha(HERE/(r['name']+'_stdout.log')),warmstart_model_sha256=sha(warm_path),warmstart_source_sha256=config['warmstart_source_sha256'],numpy_torch_float64_replay_max_abs_difference_T=r['numpy_torch_float64_replay_max_abs_difference_T'])
    manifest=dict(version='v5-prespecified-one-final51-magnitude-finetune-frozen',models=[model],completed_new_neural_trainings=1,completed_gradient_fine_tuning_runs=1,completed_from_scratch_training_runs=0,failed_attempts_excluded=[],protocol_relative_path='candidate_protocol.json',protocol_sha256=sha(pp),training_case_ids=ids,validation_case_ids=[],excluded_test_case_ids=identity['excluded_case_ids'],fulltrain_shape=[51,60,1800],validation=None,train51_metrics_are_fitting=True,fixed_recipe=protocol['fixed_final51_recipe'],local_blend_evaluations=0,
        inference=dict(api='predict(model_or_path,nativeconditions,frame_index=None,return_signed=False,return_mean=False)',default_shape='C60x1800 rawplanar magnitude T',signed_shape='C60x3600 interleavedBrBt T',mean_shape='C1800x2 staticmeanBrBt T',self_contained_numpy_module=True,dependencies=['numpy','threadpoolctl recommended CPU2threads'],xy_units='mm',saved_firstcycle60frames_only=True),independent_numpy_QA=dict(report_relative_path=qa.relative_to(HERE).as_posix(),report_sha256=sha(qa),checker_relative_path='verify_final51.py',checker_sha256=sha(__file__),primitives_relative_path='qa_primitives.py',primitives_sha256=sha(HERE/'qa_primitives.py'),passed_checks=report['passed_checks'],failed_checks=report['failed_checks'],original42_four_andV4_final51_immutable=True),
        scope='Actual51 fitting addon: originalval9 nowtraining; originaltest9notdecoded. No independentvalidation, no newblindtest; fixedrecipe previously selectedon42/9. Notfull60 fitting, not42baseline replacement.',test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,train51_labels_loaded=True,validation_point_weighting=False,hotspot_cropping=False,posthoc_magnitude_projection=False)
    fm=HERE/'final_manifest.json';fm.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(HERE/'final_manifest.sha256').write_text(sha(fm)+'\n');print(json.dumps(dict(name=r['name'],train51_L2=reference['relative_L2_pct'],train51_max_T=reference['max_abs_T'],best_step=r['best_step'],passed=report['passed_checks'],failed=report['failed_checks'],manifest_sha256=sha(fm)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
