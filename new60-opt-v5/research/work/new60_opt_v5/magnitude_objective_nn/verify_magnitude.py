"""Saved-weight independent V5 fulltrain42/fullval9 original-T NumPy QA."""
from pathlib import Path
import builtins
import csv
import hashlib
import importlib.util
import json
import math
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREP=ROOT/'work/new60_opt_v3/joint_neural/point_prepared'
TRUTH=ROOT/'work/new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared/validation_truth.npz'
CACHE=ROOT/'work/new60_20261014/cache_v1/rotor.npz'

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

def selected_train_rows(path,ids):
    with zipfile.ZipFile(path) as archive,archive.open('field.npy') as stream:
        version=np.lib.format.read_magic(stream);shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
        check('independent_raw_field_header',shape==(60,60,3600) and not fortran and dtype.kind=='f')
        block=60*3600*dtype.itemsize;out=np.empty((42,60,3600),dtype=np.float64);mapping={int(cid):i for i,cid in enumerate(ids)};access=[]
        for cid in range(60):
            if cid in mapping:
                raw=stream.read(block);check('selected_row_byte_count/'+str(cid),len(raw)==block);out[mapping[cid]]=np.frombuffer(raw,dtype=dtype).reshape(60,3600);access.append(cid)
            else:stream.seek(block,1)
    check('numeric_labels_exact_train42',access==ids)
    return out.reshape(42,60,1800,2)

def static_forward(m,c,point):
    result=[]
    for current,theta in c:
        angle=float(theta)*math.pi/180;f=np.asarray([current/float(m['current_scale'])*math.cos(angle),current/float(m['current_scale'])*math.sin(angle)]);x=(f-m['feature_mean'])/m['feature_scale'];v=m['linear_bias'][point].copy()
        for j in range(2):v+=x[j]*m['linear_weight'][point,j]
        for gate in range(m['gate_bias'].shape[1]):
            z=float(m['gate_bias'][point,gate])
            for j in range(2):z+=float(x[j]*m['gate_weight'][point,j,gate])
            g=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z));e=m['expert_weight'][point,gate,0].copy()
            for j in range(2):e+=x[j]*m['expert_weight'][point,gate,j+1]
            v+=g*e
        result.append(v*m['output_scale'][point]+m['output_mean'][point])
    return np.asarray(result)

def field_metrics(p,y,ids,point_ids):
    e=p-y;se=np.square(e).sum(axis=(1,2));energy=np.square(y).sum(axis=(1,2));rm=np.sqrt(se/(60*1800));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape);worst=int(np.argmax(rm))
    return dict(relative_L2_pct=float(100*np.sqrt(se.sum()/energy.sum())),rmse_T=float(np.sqrt(np.square(e).mean())),mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),worst_case_id=int(ids[worst]),worst_case_rmse_T=float(rm[worst]),
        max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),point_id=int(point_ids[loc[2]]),prediction_T=float(p[loc]),truth_T=float(y[loc]),signed_error_T=float(e[loc])),
        per_case=[dict(case_id=int(cid),rmse_T=float(rr),relative_L2_pct=float(100*np.sqrt(ss/ee))) for cid,rr,ss,ee in zip(ids,rm,se,energy)])

def mean_metrics(p,y):
    e=p-y;pn=np.sqrt(np.sum(p*p,axis=-1));yn=np.sqrt(np.sum(y*y,axis=-1));me=pn-yn
    return dict(signed_static_mean_relative_L2_pct=float(100*np.sqrt(np.square(e).sum()/np.square(y).sum())),signed_static_mean_component_rmse_T=float(np.sqrt(np.square(e).mean())),signed_static_mean_component_max_abs_T=float(np.abs(e).max()),norm_of_signed_mean_relative_L2_pct=float(100*np.sqrt(np.square(me).sum()/np.square(yn).sum())),norm_of_signed_mean_rmse_T=float(np.sqrt(np.square(me).mean())),norm_of_signed_mean_max_abs_T=float(np.abs(me).max()))

def infer(mod,path,c,**kwargs):
    old_import=builtins.__import__;old_load=np.load;opened=[]
    def imp(name,*a,**k):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise ImportError('Training dependency forbidden: '+name)
        return old_import(name,*a,**k)
    def load(file,*a,**k):
        actual=Path(file).resolve();opened.append(str(actual))
        if actual!=path.resolve():raise AssertionError('Non-weight read during inference: '+str(actual))
        return old_load(file,*a,**k)
    builtins.__import__=imp;np.load=load
    try:result=mod.predict(path,c,**kwargs)
    finally:builtins.__import__=old_import;np.load=old_load
    check(path.parent.name+'/inference_only_own_weights',opened==[str(path.resolve())]);return result

def main():
    source=HERE/'magnitude_point_nn.py';pp=HERE/'candidate_protocol.json';protocol=json.loads(pp.read_text(encoding='utf-8'));identity=json.loads((PREP/'identity.json').read_text(encoding='utf-8'));a=read_npz(PREP/'mean_data.npz');geo=read_npz(PREP/'geometry.npz');base=read_npz(PREP/'baseline_model.npz');truthdata=read_npz(TRUTH);truth=truthdata['truth'];vids=truthdata['case_ids'];tids=a['case_ids'][:42];c=a['conditions']
    check('source_predeclared_actual',sha(source)==protocol['source_sha256']);check('protocol_sidecar',sha(pp)==(HERE/'candidate_protocol.sha256').read_text().strip())
    check('raw_cache_source_identity',sha(CACHE)==protocol['raw_cache_sha256']==identity['cache_sha256']);check('prepared_identity_binding',sha(PREP/'identity.json')==protocol['prepared_identity_sha256']);check('prepared_means_binding',sha(PREP/'mean_data.npz')==protocol['prepared_means_sha256']);check('validation_truth_binding',sha(TRUTH)==protocol['validation_truth_sha256'])
    check('42_9_9_disjoint',len(tids)==42 and len(vids)==9 and not(set(identity['label_access'])&set(identity['excluded_case_ids'])) and tids.tolist()==identity['train_case_ids'] and vids.tolist()==identity['validation_case_ids'])
    check('fullvalshape',truth.shape==(9,60,1800));check('six_frame_cycles_exact_all60',sorted([offset+10*k for offset in range(10) for k in range(6)])==list(range(60)))
    for relative,key in [('work/new60_opt_v4/point_regime_nn','original_regime8_frozen_manifest_sha256'),('work/new60_opt_v4/radial_saturation_nn','radial2_frozen_manifest_sha256')]:
        frozen=ROOT/relative/'frozen_directory_manifest.json';snapshot=json.loads(frozen.read_text(encoding='utf-8'));check(relative+'/manifest_immutable',sha(frozen)==protocol[key])
        for entry in snapshot['files']:check(relative+'/immutable/'+entry['path'],sha(frozen.parent/entry['path'])==entry['sha256'])
    true_signed=selected_train_rows(CACHE,identity['train_case_ids']);traintruth=np.hypot(true_signed[...,0],true_signed[...,1]);close('trainmean_trueframes',true_signed.mean(1),a['signed_mean'][:42]);close('trainfulltrueenergy',np.square(traintruth).sum(axis=(1,2)),a['truth_energy'][:42],atol=1e-8)
    old_import=builtins.__import__
    def imp(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise ImportError(name)
        return old_import(name,*args,**kwargs)
    builtins.__import__=imp
    try:spec=importlib.util.spec_from_file_location('magnitude_numpy_checked',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    finally:builtins.__import__=old_import
    results=sorted((HERE/'runs').glob('*/result.json'));failures=sorted((HERE/'runs').glob('*/failed.json'));check('four_actual_completed_finetunes',len(results)==4);check('zero_failed_attempts',len(failures)==0)
    models=[];saved={}
    for rp in results:
        run=rp.parent;name=run.name;pre=name+'/';r=json.loads(rp.read_text(encoding='utf-8'));path=run/'model.npz';m=read_npz(path);meta=json.loads(str(m['metadata_json']));warm=read_npz(Path(r['configuration']['warmstart_model']));warmmeta=json.loads(str(warm['metadata_json']))
        check(pre+'weight_sha',sha(path)==r['model_sha256']==(run/'model.npz.sha256').read_text().strip());check(pre+'source_sha',sha(source)==r['source_sha256']==meta['source_sha256']);check(pre+'protocol_sha',sha(pp)==r['protocol_sha256']==meta['protocol_sha256']);check(pre+'exact_predeclaredconfig',r['configuration'] in protocol['candidates'] and r['configuration']==meta['configuration'])
        check(pre+'warmstart_sha',sha(Path(r['configuration']['warmstart_model']))==meta['warmstart_model_sha256']==r['configuration']['warmstart_model_sha256']);check(pre+'warmstart_source_sha',warmmeta['source_sha256']==meta['warmstart_source_sha256']==r['configuration']['warmstart_source_sha256']);check(pre+'warmstart_actuallytrain42',warmmeta['train_case_ids']==identity['train_case_ids'] and not warmmeta['full60_fit_labels_loaded'] and not warmmeta['test_labels_loaded'])
        check(pre+'notfromscratch',r['new_gradient_fine_tuning_run'] and not r['from_scratch_training'])
        for key in ('train_case_ids','validation_case_ids','excluded_case_ids'):check(pre+key,meta[key]==identity[key])
        for key in ('test_labels_loaded','test_metrics_computed','full60_fit_labels_loaded','train51_labels_loaded'):check(pre+key,not r[key] and not meta[key])
        check(pre+'labels_onlytrain42',meta['raw_label_rows_loaded']==r['raw_label_rows_loaded']==identity['train_case_ids']);check(pre+'uniformallpoints',meta['point_weighting']=='uniform all1800')
        for key,value in geo.items():check(pre+'geometry_'+key,np.array_equal(m[key],value))
        for key in ('current_scale','feature_mean','feature_scale','output_mean','output_scale','dynamic_scale'):check(pre+'immutablewarmscaler_'+key,np.array_equal(m[key],warm[key]))
        embedded={k[9:]:v for k,v in m.items() if k.startswith('baseline_')};check(pre+'oldbaselinenkeys',set(embedded)==set(base))
        for key,value in base.items():check(pre+'immutableoldneural_'+key,np.array_equal(embedded[key],value))
        keys=('linear_weight','linear_bias','gate_weight','gate_bias','expert_weight');count=sum(m[k].size for k in keys);check(pre+'realcapacity',count==r['new_trainable_parameters']==1800*r['new_parameters_per_point'] and m['gate_bias'].shape[1]==r['configuration']['gates'])
        pred=infer(mod,path,c[42:]);signed=infer(mod,path,c[42:],return_signed=True).reshape(9,60,1800,2);mean=infer(mod,path,c[42:],return_mean=True);tpred=infer(mod,path,c[:42]);tsigned=infer(mod,path,c[:42],return_signed=True).reshape(42,60,1800,2);tmean=infer(mod,path,c[:42],return_mean=True)
        check(pre+'fullfiniteunprojected',pred.shape==(9,60,1800) and tpred.shape==(42,60,1800) and np.isfinite(pred).all() and np.isfinite(tpred).all() and np.all(pred>=0) and np.all(tpred>=0));close(pre+'signedhypot',pred,np.sqrt(np.sum(signed*signed,axis=-1)));close(pre+'time_mean',signed.mean(1),mean);close(pre+'frozenvaldynamic',signed-mean[:,None],a['validation_signed_dynamic'])
        for frame in (0,17,59):close(pre+'singleframe'+str(frame),infer(mod,path,c[42:],frame_index=frame),pred[:,frame])
        close(pre+'singlecondition',infer(mod,path,c[42]),pred[0])
        for point in (0,900,1359,1751,1799):close(pre+'independent_scalar_point'+str(point),static_forward(m,c[42:],point),mean[:,point])
        for bad in (-1,60,1.5):
            try:infer(mod,path,c[42:],frame_index=bad);ok=False
            except ValueError:ok=True
            check(pre+'invalidframe'+str(bad),ok)
        for label,p,y,ids in [('val',pred,truth,vids),('train_full60',tpred,traintruth,tids)]:
            value=field_metrics(p,y,ids,geo['point_id']);reference=r['validation'] if label=='val' else r['train_full60_magnitude']
            for key in ('relative_L2_pct','rmse_T','mae_T','max_abs_T','worst_case_rmse_T'):close(pre+label+'_'+key,np.asarray(value[key]),np.asarray(reference[key]))
            check(pre+label+'_worstcase',value['worst_case_id']==reference['worst_case_id']);check(pre+label+'_maxlocation',value['max_abs_location']==reference['max_abs_location'])
            for i,record in enumerate(value['per_case']):
                check(pre+label+'_caseid'+str(i),record['case_id']==reference['per_case'][i]['case_id'])
                for key in ('rmse_T','relative_L2_pct'):close(pre+label+'_case'+str(i)+'_'+key,np.asarray(record[key]),np.asarray(reference['per_case'][i][key]))
        for label,ppred,target in [('train_static_mean',tmean,a['signed_mean'][:42]),('validation_static_mean',mean,a['signed_mean'][42:])]:
            for key,value in mean_metrics(ppred,target).items():close(pre+label+'_'+key,np.asarray(value),np.asarray(r[label][key]))
        se=tsigned-true_signed;tsmetrics=dict(relative_L2_pct=float(100*np.sqrt(np.square(se).sum()/np.square(true_signed).sum())),component_rmse_T=float(np.sqrt(np.square(se).mean())),component_max_abs_T=float(np.abs(se).max()))
        for key,value in tsmetrics.items():close(pre+'trainfullsigned_'+key,np.asarray(value),np.asarray(r['train_full60_signed_components'][key]))
        with (run/'history.csv').open(encoding='utf-8-sig',newline='') as stream:history=list(csv.DictReader(stream))
        check(pre+'51completecheckpoints',[int(h['step']) for h in history]==[0,*range(50,2001,50),*range(2010,2101,10)])
        for h in history:
            close(pre+'objective_formula_step'+h['step'],np.asarray(float(h['train_full_objective_T2'])),np.asarray(float(h['train_full60_magnitude_mse_T2'])+r['configuration']['lambda_signed']*float(h['train_signed_mean_regularizer_T2'])))
            if h['stage']=='Adam_fixed6frame_cycle':check(pre+'predeterminedframebatch_step'+h['step'],h['frame_batch']==','.join(map(str,range((int(h['step'])-1)%10,60,10))))
        winner=min(history,key=lambda h:float(h['validation_rmse_T']));check(pre+'minimumvalRMSEcheckpoint',int(winner['step'])==r['best_step']==meta['best_step']);close(pre+'selectedval64replay',np.asarray(r['validation']['rmse_T']),np.asarray(float(winner['validation_rmse_T'])),atol=1e-7,rtol=2e-4);close(pre+'selectedfulltrainreplay',np.asarray(r['train_full60_magnitude']['rmse_T']),np.asarray(float(winner['train_full60_magnitude_rmse_T'])),atol=1e-7,rtol=2e-4)
        check(pre+'actual2000Adam100LBFGS',r['final_adam']['step']==2000 and r['final_optimizer']['step']==2100 and r['history_checkpoint_count']==51);check(pre+'Torch64_numpyproof',r['numpy_torch_float64_replay_max_abs_difference_T']<1e-10)
        models.append(dict(id=name,model_relative_path=path.relative_to(HERE).as_posix(),inference_module_relative_path=source.relative_to(HERE).as_posix(),model_sha256=sha(path),inference_module_sha256=sha(source),predict_kwargs={},signed_output_kwargs=dict(return_signed=True),mean_output_kwargs=dict(return_mean=True),validation=r['validation'],train_full60_magnitude=r['train_full60_magnitude'],train_static_mean=r['train_static_mean'],validation_static_mean=r['validation_static_mean'],train_full60_signed_components=r['train_full60_signed_components'],configuration=r['configuration'],new_gradient_fine_tuning_run=True,from_scratch_training=False,new_trainable_parameters=count,new_parameters_per_point=r['new_parameters_per_point'],best_step=r['best_step'],selected_warmstart_step0=r['best_step']==0,trainable_arrays_changed=sum(not np.array_equal(m[k],warm[k]) for k in keys),warmstart_model_sha256=meta['warmstart_model_sha256'],warmstart_source_sha256=meta['warmstart_source_sha256'],history_relative_path=(run/'history.csv').relative_to(HERE).as_posix(),history_sha256=sha(run/'history.csv'),history_checkpoint_count=len(history),result_relative_path=rp.relative_to(HERE).as_posix(),result_sha256=sha(rp),stdout_relative_path=name+'_stdout.log',stdout_sha256=sha(HERE/(name+'_stdout.log')),numpy_torch_float64_replay_max_abs_difference_T=r['numpy_torch_float64_replay_max_abs_difference_T']))
        saved[name]=pred.copy()
    selected=[]
    for role,m in [('best_average',min(models,key=lambda m:m['validation']['rmse_T'])),('best_maximum',min(models,key=lambda m:m['validation']['max_abs_T']))]:
        cache=HERE/(role+'_validation_replay.npz');np.savez_compressed(cache,prediction=saved[m['id']],case_ids=vids,conditions=c[42:],point_id=geo['point_id'],theta=geo['theta'],time_s=geo['time_s'],model_sha256=np.asarray(m['model_sha256']));selected.append(dict(role=role,id=m['id'],relative_path=cache.relative_to(HERE).as_posix(),sha256=sha(cache)))
    report=dict(passed_checks=sum(item['passed'] for item in checks),failed_checks=sum(not item['passed'] for item in checks),checks=checks,raw_label_rows_loaded=identity['train_case_ids'],validation_truth_path=str(TRUTH),checker_sha256=sha(__file__),test_labels_loaded=False,full60_fit_labels_loaded=False,train51_labels_loaded=False)
    qa=HERE/'independent_numpy_replay_report.json';qa.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest=dict(version='v5-four-magnitude-objective-finetuned-NN-manifest',models=models,completed_new_neural_trainings=4,completed_gradient_fine_tuning_runs=4,completed_from_scratch_training_runs=0,failed_attempts_excluded=[],local_blend_evaluations=0,
        protocol_relative_path='candidate_protocol.json',protocol_sha256=sha(pp),train_case_ids=identity['train_case_ids'],validation_case_ids=identity['validation_case_ids'],excluded_case_ids=identity['excluded_case_ids'],full_train_shape=[42,60,1800],full_validation_shape=[9,60,1800],
        inference=dict(api='predict(model_or_path,native_conditions,frame_index=None,return_signed=False,return_mean=False)',default_shape='C60x1800 raw planar magnitude T',signed_shape='C60x3600 interleaved Br/Bt T',mean_shape='C1800x2 static signed Br/Bt mean T',self_contained_numpy_module=True,dependencies=['numpy','threadpoolctl recommended for CPU2threads'],native_input='[I_RMS_A,Thet_deg]; IcosThet/IsinThet mathematical features, notverifiedphysicaldq',xy_units='mm',saved_first_cycle60frames_only=True),
        independent_numpy_QA=dict(report_relative_path=qa.relative_to(HERE).as_posix(),report_sha256=sha(qa),checker_relative_path='verify_magnitude.py',checker_sha256=sha(__file__),passed_checks=report['passed_checks'],failed_checks=report['failed_checks'],old_V4_regime8_and_radial2_immutable=True),selected_validation_caches=selected,
        scope='Allnumericfullframe traininglabels onlytrain42, validationnormfromexistingval9-onlytruth. Oldval already repeatedly used. Oldtest9 seeninprojectV1-V4; thisroute no testlabels, anyfuturetest reusednotblind. No train51/full60 model.',
        test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,train51_labels_loaded=False,validation_point_weighting=False,hotspot_cropping=False,posthoc_magnitude_projection=False)
    path=HERE/'final_manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(HERE/'final_manifest.sha256').write_text(sha(path)+'\n');print(json.dumps(dict(models=[dict(id=m['id'],val_L2=m['validation']['relative_L2_pct'],max_T=m['validation']['max_abs_T'],train_L2=m['train_full60_magnitude']['relative_L2_pct'],best_step=m['best_step']) for m in models],passed=report['passed_checks'],failed=report['failed_checks'],manifest_sha256=sha(path)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
