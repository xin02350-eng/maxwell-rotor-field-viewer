"""Independent final frozen V3 scientific replay; never selects or trains.

Reads test/full60 numeric targets only after both registries and all declared
SHA bindings validate. Original merged float32 components are converted to
float64 before magnitude/metric calculations. Website q16 is never opened.
"""
from __future__ import annotations
from pathlib import Path
import builtins,csv,copy,hashlib,importlib.util,json,sys,time,zipfile
from datetime import datetime,timezone
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;V1=WORK/'new60_20261014';V2=WORK/'new60_opt_v2';OUT=ROOT/'evaluation'
MERGED=V1/'source/combined/flux60_s60_merged.npz'
SOURCE_SHA='e456656a2aefb718de541ae3169c79e3950a6d7ada0ac980e4a22fdc097b1e96'
checks=[];results=[];accesses=[];case_rows=[];frame_rows=[];point_rows=[]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def check(name,ok,detail=None):
    checks.append(dict(check=name,pass_=bool(ok),detail=detail))
    if not ok:raise AssertionError(name)
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def permitted_rows(path,key,indices):
    rows={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as z,z.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or len(shape)!=3 or shape[0]!=60:raise ValueError('Source case-major shape changed')
        size=int(np.prod(shape[1:]))*dtype.itemsize;out=np.empty((len(indices),*shape[1:]),dtype=dtype)
        for i in range(60):
            if i in rows:
                raw=f.read(size)
                if len(raw)!=size:raise ValueError('Short source row')
                out[rows[i]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(size,1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite source labels')
    accesses.append(dict(path=str(Path(path).relative_to(WORK)),key=key,numeric_case_rows=list(map(int,indices)),context='independent scientific truth replay'))
    return out.astype(np.float64)
def source_truth(region,indices):
    br=permitted_rows(MERGED,region+'_Br',indices)
    if region=='airgap':return br
    bt=permitted_rows(MERGED,region+'_Bt',indices);return np.hypot(br,bt)
def metric(p,y,point_count=None,frame_count=None):
    e=p-y;sse=float(np.sum(e*e));energy=float(np.sum(y*y))
    return dict(rmse_T=float(np.sqrt(sse/e.size)),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),relative_L2_pct=float(100*np.sqrt(sse/energy)),
        point_sample_count=int(y.shape[-1]) if point_count is None else point_count,frame_count=(int(y.shape[-2]) if y.ndim>1 else None) if frame_count is None else frame_count,scalar_count=int(y.size))
def guarded_predict(api,choice,c,frame_index=None):
    old_import=builtins.__import__;old_get=np.lib.npyio.NpzFile.__getitem__;records=[]
    sourcepaths={MERGED.resolve(),*(p.resolve() for p in (V1/'cache_v1').glob('*.npz'))}
    def restricted(name,*a,**kw):
        if name.split('.')[0] in ('torch','sklearn','scipy'):raise ImportError('Deployment must not import '+name)
        return old_import(name,*a,**kw)
    def watched(self,key):
        path=Path(self.zip.filename).resolve();records.append(dict(path=str(path),key=str(key)))
        if path in sourcepaths and str(key) in ('field','coeff','truth','target','stator_Br','stator_Bt','rotor_Br','rotor_Bt','airgap_Br'):
            raise AssertionError('Inference attempted source label access')
        return old_get(self,key)
    builtins.__import__=restricted;np.lib.npyio.NpzFile.__getitem__=watched
    try:p=api.predict(ROOT,choice,c,frame_index=frame_index)
    finally:builtins.__import__=old_import;np.lib.npyio.NpzFile.__getitem__=old_get
    check(choice['id']+' inference no source labels / no torch sklearn scipy',True,dict(npz_keys_accessed=len(records),declared_dependencies=['numpy','threadpoolctl']))
    return p
def write_csv(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    threadpool_limits(2);start=time.time()
    api=module(ROOT/'code/inference_numpy_v3.py','independent_final_v3_api')
    registry=api.read_freeze(ROOT);choices_sha=sha(ROOT/'both_versions_choices_frozen.json');prediction_sha=sha(ROOT/'prediction_choices_frozen.json')
    check('registry test not used for training or selection',registry['test_labels_used_for_training'] is False and registry['test_labels_used_for_selection'] is False)
    check('registry acknowledges reused test',registry['previous_test_already_seen'] is True)
    check('final8 model branches',len(registry['models'])==8 and sum(c['branch']=='prediction' for c in registry['models'])==5)
    check('raw merged SHA exact',sha(MERGED)==SOURCE_SHA)
    split_path=(ROOT/registry['split_relative_path']).resolve();split=read(split_path)
    check('split SHA exact',sha(split_path)==registry['split_sha256'])
    check('original42/9/9 disjoint',len(split['train'])==42 and len(split['validation'])==9 and len(split['test'])==9 and len(set(split['train']+split['validation']+split['test']))==60)
    for choice in registry['models']:
        check(choice['id']+' final model SHA',sha(ROOT/choice['model_relative_path'])==choice['model_sha256'])
        check(choice['id']+' final source SHA',sha(ROOT/choice['inference_module_relative_path'])==choice['inference_module_sha256'])
        for item in choice.get('provenance_files',[]):check(choice['id']+' provenance '+item['relative_path'],sha(ROOT/item['relative_path'])==item['sha256'])
    # The preceding gates run before any test/full60 target is materialized.
    with np.load(MERGED,allow_pickle=False) as z:
        names=['case_id','I_A','Thet_deg','speed_rpm','solver_index','time_s','theta_e_deg','theta_m_relative_deg','theta_m_absolute_deg','initial_mechanical_deg']
        shared={k:z[k] for k in names};geometry={r:dict(point_id=z[r+'_point_id'],xy=np.column_stack((z[r+'_x_ref_mm'],z[r+'_y_ref_mm'])),hotspot=z[r+'_hotspot']) for r in ('rotor','stator','airgap')}
    conditions=np.column_stack((shared['I_A'],shared['Thet_deg']))
    check('merged case order0..59',shared['case_id'].tolist()==list(range(60)))
    check('all60 supplied frames exact',shared['solver_index'].tolist()==list(range(1,61)))
    check('theta absolute uses initial plus relative',np.array_equal(shared['theta_m_absolute_deg'],shared['theta_m_relative_deg']+shared['initial_mechanical_deg']))
    cache_meta={}
    for region in ('rotor','stator','airgap'):
        with np.load(V1/'cache_v1'/f'{region}.npz',allow_pickle=False) as z:
            m=json.loads(str(z['metadata_json']));cache_meta[region]=m
            check(region+' units T/mm',m['units']=='T' and m['xy_units']=='mm')
            for key,value in [('point_id',geometry[region]['point_id']),('xy',geometry[region]['xy']),('conditions',conditions),('theta',shared['theta_m_absolute_deg']),('time_s',shared['time_s'])]:check(region+' original geometry/input '+key,np.array_equal(z[key],value))
    OUT.mkdir(exist_ok=True)
    # Predictions, then fitting. No step below can modify or select a model.
    for choice in registry['models']:
        cid=choice['id'];region=choice['region'];geo=geometry[region];npoints=len(geo['point_id']);cache=V1/'cache_v1'/f'{region}.npz'
        check(cid+' cache SHA exact',sha(cache)==choice['cache_sha256'])
        if choice['branch']=='prediction':
            check(cid+' training42 identities',choice['train_case_ids']==split['train'])
            partitions=[('train',split['train']),('validation',split['validation']),('reused_test',split['test'])]
        else:partitions=[('training_fit',list(range(60)))]
        for part,index in partitions:
            y=source_truth(region,index);sourcecache=permitted_rows(cache,'field',index)
            check(cid+'/'+part+' cache/merged truth bit-exact',np.array_equal(y,sourcecache if region=='airgap' else np.hypot(sourcecache[...,0::2],sourcecache[...,1::2])))
            del sourcecache
            c=conditions[index];p=guarded_predict(api,choice,c)
            check(cid+'/'+part+' output scalar shape',p.shape==(len(index),60,npoints))
            check(cid+'/'+part+' finite float64',p.dtype==np.float64 and np.isfinite(p).all() and y.dtype==np.float64)
            raw=None
            if choice['branch']=='fitting_full60':
                rawchoice=copy.deepcopy(choice);rawchoice['predict_kwargs']=dict(choice.get('predict_kwargs',{}),nonnegative=False)
                raw=guarded_predict(api,rawchoice,c)
                check(cid+' deployed sign policy',np.array_equal(p,np.maximum(raw,0) if region!='airgap' else raw))
                fit=load(ROOT/choice['model_relative_path']);meta=json.loads(str(fit['metadata_json']));old=load(V2/f'neural_route/fitting_full60/{region}/model.npz')
                check(cid+' post-prediction-freeze training binding',meta['prediction_freeze_sha256']==prediction_sha)
                check(cid+' no heldout fitting claim',meta['heldout_accuracy_claim'] is False and meta['all60_training'] is True)
                check(cid+' all60 fitting case IDs',meta['training_case_ids']==list(range(60)))
                for layer in range(int(fit['layer_count'])-1):
                    check(cid+f' trained hidden layer{layer} unchanged',np.array_equal(fit[f'w{layer}'],old[f'w{layer}']) and np.array_equal(fit[f'b{layer}'],old[f'b{layer}']))
                for key in ('cycle_basis','cycle_mean','output_scale','feature_mean','feature_scale'):
                    check(cid+' original trained representation '+key,np.array_equal(fit[key],old[key]))
                r=c[:,0]/float(fit['current_scale']);a=np.deg2rad(c[:,1]);xx=r*np.cos(a);yy=r*np.sin(a)
                x=(np.column_stack((r,c[:,1]/90,xx,yy,xx*xx,xx*yy,yy*yy))-fit['feature_mean'])/fit['feature_scale']
                for layer in range(int(fit['layer_count'])-1):x=np.tanh(x@old[f'w{layer}']+old[f'b{layer}'])
                design=np.column_stack((x,np.ones(60)));target=(y.reshape(60,-1)-fit['cycle_mean'])@fit['cycle_basis'].T
                head,_,rank,singular=np.linalg.lstsq(design,target/fit['output_scale'],rcond=1e-12)
                last=int(fit['layer_count'])-1
                head_error=float(max(np.max(np.abs(head[:-1]-fit[f'w{last}'])),np.max(np.abs(head[-1]-fit[f'b{last}']))))
                check(cid+' supervised last-layer LS independently verified',rank==60 and head_error<1e-9,dict(rank=int(rank),head_max_abs_difference=head_error,design_condition_number=float(singular[0]/singular[-1])))
                fit_report=read((ROOT/choice['model_relative_path']).parent/'result.json')
                rawmetrics=metric(raw,y)
                check(cid+' original fitting raw score replay',np.isclose(rawmetrics['rmse_T'],fit_report['fitting_raw']['rmse_T'],rtol=1e-6,atol=1e-21),rawmetrics)
            scores=metric(p,y);e=p-y;abs_e=np.abs(e)
            case_sse=np.sum(e*e,axis=(1,2));point_sse=np.sum(e*e,axis=(0,1));frame_sse=np.sum(e*e,axis=(0,2))
            loc=np.unravel_index(np.argmax(abs_e),e.shape);worstcase=int(np.argmax(case_sse));worstpoint=int(np.argmax(point_sse));worstframe=int(np.argmax(frame_sse))
            extreme=dict(case_id=int(index[loc[0]]),frame_index=int(loc[1]),solver_index=int(shared['solver_index'][loc[1]]),point_index=int(loc[2]),point_id=int(geo['point_id'][loc[2]]),
                theta_m_absolute_deg=float(shared['theta_m_absolute_deg'][loc[1]]),time_s=float(shared['time_s'][loc[1]]),x_ref_mm=float(geo['xy'][loc[2],0]),y_ref_mm=float(geo['xy'][loc[2],1]),prediction_T=float(p[loc]),truth_T=float(y[loc]),absolute_error_T=float(abs_e[loc]))
            for j,i in enumerate(index):
                case_rows.append(dict(id=cid,branch=choice['branch'],region=region,quantity=choice['quantity'],partition=part,case_id=int(i),**metric(p[j],y[j])))
                for frame in range(60):frame_rows.append(dict(id=cid,partition=part,case_id=int(i),frame_index=frame,solver_index=frame+1,theta_m_absolute_deg=float(shared['theta_m_absolute_deg'][frame]),**metric(p[j,frame],y[j,frame],frame_count=1)))
            for point in range(npoints):point_rows.append(dict(id=cid,partition=part,point_index=point,point_id=int(geo['point_id'][point]),x_ref_mm=float(geo['xy'][point,0]),y_ref_mm=float(geo['xy'][point,1]),**metric(p[:,:,point],y[:,:,point],point_count=1,frame_count=60)))
            metainfo=dict(version='NEW60_OPT_V3_SCIENTIFIC_REPLAY',model_id=cid,branch=choice['branch'],region=region,quantity=choice['quantity'],partition=part,
                choices_sha256=choices_sha,prediction_choices_sha256=prediction_sha,model_sha256=choice['model_sha256'],inference_module_sha256=choice['inference_module_sha256'],source_sha256=SOURCE_SHA,
                source_storage='float32 components; converted to float64 before hypot and metrics',prediction_storage='float64 output; neural operator contains trained float32 forward arrays',
                metric_weight='original spatial samples equally weighted; no FEM area weights',source_xy_units='mm',units='T',previous_test_already_seen=True,
                test_scope='reused9 retrospective evaluation, not fresh blind test',space_scope='actual sampled45-degree sector only, no hole filling or mirroring',time_scope='original60 frames of first saved electrical period; no steady-state guarantee',
                minimum_inference_dependencies=['numpy','threadpoolctl'],no_training_or_selection_in_this_replay=True,metrics=scores)
            folder=OUT/cid;folder.mkdir(exist_ok=True);npz=folder/f'{part}_predictions.npz'
            if npz.exists():
                # Resume only byte-preserved files proven to bind this same final registry.
                with np.load(npz,allow_pickle=False) as prior:
                    prior_meta=json.loads(str(prior['metadata_json']))
                    check(cid+'/'+part+' resumed identical frozen arrays',prior_meta['choices_sha256']==choices_sha and np.array_equal(prior['prediction'],p) and np.array_equal(prior['truth'],y))
            else:
                np.savez_compressed(npz,prediction=p,truth=y,case_ids=np.asarray(index,dtype=np.int32),original_row_indices=np.asarray(index,dtype=np.int64),
                    partition=np.asarray(part),conditions=c,point_id=geo['point_id'],xy=geo['xy'],region=np.asarray(region),quantity=np.asarray(choice['quantity']),metadata_json=np.asarray(json.dumps(metainfo,ensure_ascii=False)),
                    theta=shared['theta_m_absolute_deg'],solver_index=shared['solver_index'],time_s=shared['time_s'],theta_m_relative_deg=shared['theta_m_relative_deg'],theta_e_deg=shared['theta_e_deg'],initial_mechanical_deg=shared['initial_mechanical_deg'],speed_rpm=shared['speed_rpm'][index])
            with np.load(npz,allow_pickle=False) as z:
                check(cid+'/'+part+' compressed scientific arrays exact',np.array_equal(z['prediction'],p) and np.array_equal(z['truth'],y))
                roundtrip=metric(z['prediction'],z['truth'])
                for key in ('rmse_T','mae_T','max_abs_T','relative_L2_pct'):check(cid+'/'+part+' scientific metric roundtrip '+key,roundtrip[key]==scores[key])
            # Single-condition / selected-frame forward checks keep all times unchanged.
            frame_error=float(np.max(np.abs(guarded_predict(api,choice,c[0],frame_index=59)-p[0,59])))
            check(cid+'/'+part+' batch/single frame numerical stability',frame_error<1e-6,frame_error)
            result=dict(id=cid,branch=choice['branch'],region=region,quantity=choice['quantity'],partition=part,pass_=True,metrics=scores,
                worst_case_id=int(index[worstcase]),worst_case_rmse_T=float(np.sqrt(case_sse[worstcase]/(60*npoints))),worst_frame_index=worstframe,worst_point_id=int(geo['point_id'][worstpoint]),max_absolute_location=extreme,
                negative_raw_count=int(np.count_nonzero((raw if raw is not None else p)<0)),raw_fitting_metrics=metric(raw,y) if raw is not None else None,
                scientific_npz_relative_path=str(npz.relative_to(ROOT)).replace('\\','/'),scientific_npz_sha256=sha(npz),scientific_npz_bytes=npz.stat().st_size,
                model_sha256=choice['model_sha256'],choices_sha256=choices_sha)
            result['pass']=result.pop('pass_');results.append(result)
            if part=='validation':
                oldscore=choice['validation_selection_metrics']
                if 'rmse_T' not in oldscore:oldscore=oldscore['selected']['validation']
                check(cid+' frozen validation score replay',np.isclose(scores['rmse_T'],oldscore['rmse_T'],rtol=5e-7,atol=1e-10),dict(actual=scores['rmse_T'],frozen=oldscore['rmse_T']))
            print(json.dumps(dict(id=cid,partition=part,relative_L2_pct=scores['relative_L2_pct'],rmse_T=scores['rmse_T'],max_abs_T=scores['max_abs_T'],scientific_bytes=result['scientific_npz_bytes']),ensure_ascii=False),flush=True)
            del p,y,raw,e,abs_e
    write_csv(OUT/'case_metrics.csv',case_rows);write_csv(OUT/'frame_metrics.csv',frame_rows);write_csv(OUT/'point_metrics.csv',point_rows)
    check('18 required scientific partitions',len(results)==18)
    check('final registry unchanged after replay',sha(ROOT/'both_versions_choices_frozen.json')==choices_sha)
    check('prediction registry unchanged after replay',sha(ROOT/'prediction_choices_frozen.json')==prediction_sha)
    api.read_freeze(ROOT)
    report=dict(version='NEW60_OPT_V3_SCIENTIFIC_QA',all_checks_pass=True,choices_sha256=choices_sha,prediction_choices_sha256=prediction_sha,
        frozen_utc=registry['frozen_utc'],completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=SOURCE_SHA,source_script_sha256=sha(__file__),
        pass_count=len(checks),checks=checks,results=results,elapsed_seconds=time.time()-start,scientific_npz_total_bytes=sum(x['scientific_npz_bytes'] for x in results),
        total_evaluation_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),label_access=accesses,minimum_inference_dependencies=['numpy','threadpoolctl'],
        floating_limit_caution='Fitting scores near floating precision are equality on all60 training labels, not evidence of new-condition accuracy or zero Maxwell error.',
        test_scope='Previously seen nine test conditions; model choices were frozen before this retrospective replay. No test-based reselection.',
        chronology_note='Three all60 readout updates occurred after prediction freeze and before final both-version freeze. The inherited registry count label after_this_freeze refers to the earlier prediction stage.')
    for p in (OUT/'scientific_qa.json',ROOT/'qa/scientific_qa.json'):
        with p.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (ROOT/'qa/scientific_qa.md').open('x',encoding='utf-8') as f:
        f.write('# V3 冻结模型独立科学复评\n\n'+f'{len(checks)}项检查通过，18个 float64 科学NPZ，压缩总大小{report["scientific_npz_total_bytes"]:,}bytes。全部指标从原merged组件计算，不读取网站q16。\n\n')
        f.write('| 模型 | 分区 | L2/% | RMSE/T | 最大绝对误差/T |\n|---|---|---:|---:|---:|\n')
        for r in results:f.write(f'| {r["id"]} | {r["partition"]} | {r["metrics"]["relative_L2_pct"]:.9g} | {r["metrics"]["rmse_T"]:.9g} | {r["metrics"]["max_abs_T"]:.9g} |\n')
        f.write('\n预测训练与选型只用train42/validation9，测试为已见9工况复用复评。全60拟合的浮点极限误差只是训练内标签复现；原trained tanh隐藏层未改，最后线性头以监督LS训练、rank60。气隙保留有符号Br。部署依赖NumPy+threadpoolctl；后者由训练/推理共享源文件顶层导入，不需要Torch/sklearn/SciPy。\n')
    print(json.dumps(dict(all_checks_pass=True,pass_count=len(checks),scientific_npz_total_bytes=report['scientific_npz_total_bytes'],report=str(ROOT/'qa/scientific_qa.json')),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
