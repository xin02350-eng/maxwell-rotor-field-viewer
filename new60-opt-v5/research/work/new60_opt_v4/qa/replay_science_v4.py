"""Independent final frozen V4 scientific replay; never selects or trains.

Copied from V3 with explicit42/51/60 scopes and immutable reused science.
Reads test/full60 numeric targets only after both registries and all declared
SHA bindings validate AND explicit final replay authorization arguments match.
Original merged float32 components become float64 before hypot and metrics.
Website q16 is never opened; training arrays are never duplicated as NPZ.
"""
from __future__ import annotations
from pathlib import Path
import argparse,builtins,csv,copy,hashlib,importlib.util,json,sys,time,zipfile
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
    old_import=builtins.__import__;old_get=np.lib.npyio.NpzFile.__getitem__;old_open=zipfile.ZipFile.open;records=[]
    sourcepaths={MERGED.resolve(),*(p.resolve() for p in (V1/'cache_v1').glob('*.npz'))}
    forbidden={'field','coeff','truth','target','stator_Br','stator_Bt','rotor_Br','rotor_Bt','airgap_Br'}
    def restricted(name,*a,**kw):
        if name.split('.')[0] in ('torch','sklearn','scipy'):raise ImportError('Deployment must not import '+name)
        return old_import(name,*a,**kw)
    def watched(self,key):
        path=Path(self.zip.filename).resolve();records.append(dict(path=str(path),key=str(key)))
        if path in sourcepaths and str(key) in forbidden:
            raise AssertionError('Inference attempted source label access')
        return old_get(self,key)
    def zipwatched(self,name,*a,**kw):
        raw=name.filename if isinstance(name,zipfile.ZipInfo) else str(name)
        key=raw.removesuffix('.npy')
        if isinstance(self.filename,(str,Path)) and Path(self.filename).resolve() in sourcepaths and key in forbidden:
            raise AssertionError('Inference attempted direct ZIP source label access')
        return old_open(self,name,*a,**kw)
    builtins.__import__=restricted;np.lib.npyio.NpzFile.__getitem__=watched;zipfile.ZipFile.open=zipwatched
    try:p=api.predict(ROOT,choice,c,frame_index=frame_index)
    finally:builtins.__import__=old_import;np.lib.npyio.NpzFile.__getitem__=old_get;zipfile.ZipFile.open=old_open
    check(choice['id']+' inference no source labels / no torch sklearn scipy',True,dict(npz_keys_accessed=len(records),declared_dependencies=['numpy','threadpoolctl']))
    return p
def write_csv(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def expected_training(split,scope):
    if scope=='original42':return list(split['train'])
    if scope=='final51':return list(split['train'])+list(split['validation'])
    if scope=='full60':return list(range(60))
    raise ValueError('Unknown training scope')

def resolve_science(choice,current_sha):
    rel=choice.get('scientific_source_relative_root',ROOT.name)
    raw=Path(rel)
    if raw.is_absolute() or '..' in raw.parts or ':' in rel:raise ValueError('Scientific root must remain relative to work')
    folder=(WORK/raw).resolve()
    if not folder.is_relative_to(WORK.resolve()):raise ValueError('Scientific root escapes work')
    expected=choice.get('expected_scientific_choices_sha256',current_sha)
    if expected=='CURRENT':expected=current_sha
    if folder!=ROOT.resolve() and expected==current_sha:raise ValueError('Reused science requires its original freeze')
    return folder,choice.get('scientific_model_id',choice['id']),expected

def inspect_training_identity(path,scope,split,seen=None):
    """Check actual leaf model IDs; global blend metadata contains bound leaves."""
    path=Path(path).resolve();seen=set() if seen is None else seen
    if path in seen:return
    seen.add(path)
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json']))
    components=meta.get('components')
    if components and all(isinstance(c,dict) and 'model_relative_path' in c for c in components):
        if 'training_case_count' in meta:
            check(str(path)+' declared training count',int(meta['training_case_count'])==len(expected_training(split,scope)))
        else:
            check(str(path)+' metadata-only wrapper retains train-excluded test',meta.get('format')=='v4_radial_finite_blend' and meta.get('test_labels_used') is False)
        check(str(path)+' nonnegative global unit-sum blend',all(float(q['weight'])>=0 for q in components) and abs(sum(float(q['weight']) for q in components)-1)<1e-12)
        for comp in components:
            mp=(path.parent/comp['model_relative_path']).resolve();sp=(path.parent/comp['module_relative_path']).resolve()
            check(str(path)+' component model '+comp['name'],sha(mp)==comp['model_sha256'])
            check(str(path)+' component source '+comp['name'],sha(sp)==comp['module_sha256'])
            inspect_training_identity(mp,scope,split,seen)
        return
    actual=next((meta[k] for k in ['train_case_ids','training_case_ids','training_original_ids'] if k in meta),None)
    if actual is None:
        nested=meta.get('identity',meta.get('data_identity',{}))
        actual=next((nested[k] for k in ['train_case_ids','training_case_ids','training_original_ids'] if k in nested),None)
    check(str(path)+' exact training IDs/order',actual==expected_training(split,scope),dict(actual=actual,expected=expected_training(split,scope)))
    if scope=='final51':
        check(str(path)+' former validation is training only',meta.get('validation_case_ids',meta.get('validation_original_ids'))==[] and meta.get('former_validation_now_training_case_ids')==split['validation'])
        excluded=meta.get('excluded_case_ids',meta.get('excluded_test_original_ids'))
        check(str(path)+' test IDs excluded',excluded==split['test'])
        check(str(path)+' test/full60fit labels not trained',meta.get('test_labels_loaded') is False and meta.get('full60_fit_labels_loaded') is False)

def validate_prelabel_bindings(api,registry,choices_sha,prediction_sha,split):
    check('registry test excluded from training/selection',registry['test_labels_used_for_training'] is False and registry['test_labels_used_for_selection'] is False)
    check('previously seen test acknowledged',registry['previous_test_already_seen'] is True)
    check('9 final model identities',len(registry['models'])==9 and len({c['id'] for c in registry['models']})==9)
    counts={s:sum(c['training_scope']==s for c in registry['models']) for s in ['original42','final51','full60']}
    check('four42/two51/three60 model scopes',counts==dict(original42=4,final51=2,full60=3),counts)
    check('raw merged SHA exact',sha(MERGED)==SOURCE_SHA)
    check('registry binds prediction freeze',registry['prediction_freeze_sha256']==prediction_sha)
    check('original42/9/9 disjoint',tuple(len(split[k]) for k in ['train','validation','test'])==(42,9,9) and len(set(split['train']+split['validation']+split['test']))==60)
    for choice in registry['models']:
        cid=choice['id'];scope=choice['training_scope'];expect=expected_training(split,scope)
        check(cid+' model SHA',sha(ROOT/choice['model_relative_path'])==choice['model_sha256'])
        check(cid+' inference source SHA',sha(ROOT/choice['inference_module_relative_path'])==choice['inference_module_sha256'])
        check(cid+' original source cache SHA before label replay',sha(V1/'cache_v1'/f"{choice['region']}.npz")==choice['cache_sha256'])
        for item in choice.get('provenance_files',[]):check(cid+' provenance '+item['relative_path'],sha(ROOT/item['relative_path'])==item['sha256'])
        check(cid+' exact registry training order',choice['train_case_ids']==expect and choice.get('training_case_ids',expect)==expect)
        allowed={'original42':['validation','reused_test'],'final51':['reused_test'],'full60':['training_fit']}[scope]
        check(cid+' saved scientific partitions exact',choice['scientific_partitions']==allowed)
        check(cid+' validation role',choice['validation_case_ids']==(split['validation'] if scope=='original42' else []))
        check(cid+' reused test role',choice['reused_test_case_ids']==(split['test'] if scope!='full60' else []))
        if scope=='final51':check(cid+' no independent validation',choice['no_independent_validation'] is True)
        inspect_training_identity(ROOT/choice['model_relative_path'],scope,split)
        source,sourceid,expected=resolve_science(choice,choices_sha)
        if source!=ROOT.resolve():
            check(cid+' inherited registry SHA',sha(source/'both_versions_choices_frozen.json')==expected)
            old=api.read_freeze(source)
            item=next(q for q in old['models'] if q['id']==sourceid)
            for key in ['model_sha256','inference_module_sha256','cache_sha256','quantity','region']:
                check(cid+' inherited binding '+key,item[key]==choice[key])
            check(cid+' inherited model exact file',sha(source/item['model_relative_path'])==choice['model_sha256'])
            check(cid+' inherited source exact file',sha(source/item['inference_module_relative_path'])==choice['inference_module_sha256'])
    # These checks and both external SHA arguments precede all label replay.

def replay_fitting(choice,c,y,p,api,scientific_root):
    cid=choice['id'];region=choice['region'];rawchoice=copy.deepcopy(choice)
    rawchoice['predict_kwargs']=dict(choice.get('predict_kwargs',{}),nonnegative=False)
    raw=guarded_predict(api,rawchoice,c)
    check(cid+' inherited deployed sign policy',np.array_equal(p,np.maximum(raw,0) if region!='airgap' else raw))
    fit=load(ROOT/choice['model_relative_path']);meta=json.loads(str(fit['metadata_json']));old=load(V2/f'neural_route/fitting_full60/{region}/model.npz')
    # These immutable heads were trained after V3 prediction freeze, not V4.
    prior_prediction=sha(scientific_root/'prediction_choices_frozen.json')
    check(cid+' original V3 prediction freeze binding',meta['prediction_freeze_sha256']==prior_prediction)
    check(cid+' full60 fit carries no heldout claim',meta['all60_training'] is True and meta['heldout_accuracy_claim'] is False and meta['training_case_ids']==list(range(60)))
    for layer in range(int(fit['layer_count'])-1):
        check(cid+f' trained hidden layer{layer} unchanged',np.array_equal(fit[f'w{layer}'],old[f'w{layer}']) and np.array_equal(fit[f'b{layer}'],old[f'b{layer}']))
    for key in ['cycle_basis','cycle_mean','output_scale','feature_mean','feature_scale']:
        check(cid+' inherited representation '+key,np.array_equal(fit[key],old[key]))
    r=c[:,0]/float(fit['current_scale']);a=np.deg2rad(c[:,1]);xx=r*np.cos(a);yy=r*np.sin(a)
    x=(np.column_stack((r,c[:,1]/90,xx,yy,xx*xx,xx*yy,yy*yy))-fit['feature_mean'])/fit['feature_scale']
    for layer in range(int(fit['layer_count'])-1):x=np.tanh(x@old[f'w{layer}']+old[f'b{layer}'])
    design=np.column_stack((x,np.ones(60)));target=(y.reshape(60,-1)-fit['cycle_mean'])@fit['cycle_basis'].T
    head,_,rank,singular=np.linalg.lstsq(design,target/fit['output_scale'],rcond=1e-12);last=int(fit['layer_count'])-1
    delta=float(max(np.max(np.abs(head[:-1]-fit[f'w{last}'])),np.max(np.abs(head[-1]-fit[f'b{last}']))))
    check(cid+' inherited supervised linear head LS/rank60',rank==60 and delta<1e-9,dict(rank=int(rank),head_max_abs_difference=delta,design_condition_number=float(singular[0]/singular[-1])))
    report=read((ROOT/choice['model_relative_path']).parent/'result.json')
    check(cid+' raw fitting metric reproduced',np.isclose(metric(raw,y)['rmse_T'],report['fitting_raw']['rmse_T'],rtol=1e-6,atol=1e-21))
    return raw

def verify_scientific(npz,choice,part,index,c,geo,shared,p,y,expected,sourceid,oldproof=None):
    """Only finalized replay may read an inherited scientific field array."""
    cid=choice['id'];before=sha(npz)
    if oldproof is not None:
        check(cid+'/'+part+' original V3 artifact SHA',oldproof['scientific_npz_sha256']==before)
        check(cid+'/'+part+' original V3 model/choices',oldproof['model_sha256']==choice['model_sha256'] and oldproof['choices_sha256']==expected and oldproof['pass'] is True)
    with np.load(npz,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json']))
        check(cid+'/'+part+' scientific metadata binding',meta['choices_sha256']==expected and meta['model_sha256']==choice['model_sha256'] and meta['model_id']==sourceid and meta['partition']==part)
        check(cid+'/'+part+' scientific source SHA',meta['source_sha256']==SOURCE_SHA)
        check(cid+'/'+part+' scientific inference SHA',meta['inference_module_sha256']==choice['inference_module_sha256'])
        check(cid+'/'+part+' original row indices',np.array_equal(z['original_row_indices'],np.asarray(index,dtype=np.int64)))
        check(cid+'/'+part+' scalar partition/region/quantity',str(z['partition'])==part and str(z['region'])==choice['region'] and str(z['quantity'])==choice['quantity'] and meta['region']==choice['region'])
        for key,value in [('case_ids',np.asarray(index)),('conditions',c),('point_id',geo['point_id']),('xy',geo['xy']),('theta',shared['theta_m_absolute_deg']),('solver_index',shared['solver_index']),('time_s',shared['time_s']),('theta_m_relative_deg',shared['theta_m_relative_deg']),('theta_e_deg',shared['theta_e_deg']),('initial_mechanical_deg',shared['initial_mechanical_deg']),('speed_rpm',shared['speed_rpm'][index])]:
            check(cid+'/'+part+' original science axis '+key,np.array_equal(z[key],value))
        check(cid+'/'+part+' inherited/original float64 truth exact',z['truth'].dtype==np.float64 and np.array_equal(z['truth'],y))
        check(cid+'/'+part+' independent forward matches science exact',z['prediction'].dtype==np.float64 and np.array_equal(z['prediction'],p))
        check(cid+'/'+part+' metadata quantity/units',meta['quantity']==choice['quantity'] and meta['units']=='T' and meta['source_xy_units']=='mm')
        roundtrip=metric(z['prediction'],z['truth']);scores=metric(p,y)
        for key in ['rmse_T','mae_T','max_abs_T','relative_L2_pct']:check(cid+'/'+part+' metric roundtrip '+key,roundtrip[key]==scores[key])
    check(cid+'/'+part+' science immutable during QA',sha(npz)==before)
    return before

def main(args):
    if not args.allow_final_label_replay or not args.expected_choices_sha256 or not args.expected_prediction_sha256:
        raise ValueError('Final test/fullfit replay requires explicit authorization and both root-supplied frozen SHAs')
    threadpool_limits(2);start=time.time()
    check('external final choices SHA',sha(ROOT/'both_versions_choices_frozen.json')==args.expected_choices_sha256)
    check('external prediction SHA',sha(ROOT/'prediction_choices_frozen.json')==args.expected_prediction_sha256)
    api=module(ROOT/'code/inference_numpy_v4.py','independent_final_v4_api')
    registry=api.read_freeze(ROOT);choices_sha=sha(ROOT/'both_versions_choices_frozen.json');prediction_sha=sha(ROOT/'prediction_choices_frozen.json')
    split_path=(ROOT/registry['split_relative_path']).resolve();split=read(split_path)
    check('split SHA exact',sha(split_path)==registry['split_sha256'])
    validate_prelabel_bindings(api,registry,choices_sha,prediction_sha,split)
    with np.load(MERGED,allow_pickle=False) as z:
        names=['case_id','I_A','Thet_deg','speed_rpm','solver_index','time_s','theta_e_deg','theta_m_relative_deg','theta_m_absolute_deg','initial_mechanical_deg']
        shared={k:z[k] for k in names};geometry={r:dict(point_id=z[r+'_point_id'],xy=np.column_stack((z[r+'_x_ref_mm'],z[r+'_y_ref_mm'])),hotspot=z[r+'_hotspot']) for r in ['rotor','stator','airgap']}
    conditions=np.column_stack((shared['I_A'],shared['Thet_deg']))
    check('merged original case order',shared['case_id'].tolist()==list(range(60)))
    check('60 original solver frames',shared['solver_index'].tolist()==list(range(1,61)))
    check('theta initial plus relative',np.array_equal(shared['theta_m_absolute_deg'],shared['theta_m_relative_deg']+shared['initial_mechanical_deg']))
    for region in ['rotor','stator','airgap']:
        with np.load(V1/'cache_v1'/f'{region}.npz',allow_pickle=False) as z:
            m=json.loads(str(z['metadata_json']));check(region+' T/mm source units',m['units']=='T' and m['xy_units']=='mm')
            for key,v in [('point_id',geometry[region]['point_id']),('xy',geometry[region]['xy']),('conditions',conditions),('theta',shared['theta_m_absolute_deg']),('time_s',shared['time_s'])]:check(region+' metadata '+key,np.array_equal(z[key],v))
    OUT.mkdir(exist_ok=True);written=0;reused=0;old_reports={}
    for choice in registry['models']:
        cid=choice['id'];region=choice['region'];scope=choice['training_scope'];geo=geometry[region];npoints=len(geo['point_id']);cache=V1/'cache_v1'/f'{region}.npz'
        check(cid+' source cache SHA',sha(cache)==choice['cache_sha256'])
        source,sourceid,expected=resolve_science(choice,choices_sha)
        partitions=[('training_fit',expected_training(split,scope))] if scope=='full60' else [('train',expected_training(split,scope))]+([('validation',split['validation'])] if scope=='original42' else [])+[('reused_test',split['test'])]
        for part,index in partitions:
            save=part in choice['scientific_partitions'];y=source_truth(region,index);cached=permitted_rows(cache,'field',index)
            check(cid+'/'+part+' raw ZIP/cache label equality',np.array_equal(y,cached if region=='airgap' else np.hypot(cached[...,0::2],cached[...,1::2])));del cached
            c=conditions[index];p=guarded_predict(api,choice,c)
            check(cid+'/'+part+' scalar full-array shape/dtype',p.shape==(len(index),60,npoints) and p.dtype==np.float64 and y.dtype==np.float64 and np.isfinite(p).all())
            check(cid+'/'+part+' magnitude nonnegative',region=='airgap' or np.all(p>=0))
            raw=replay_fitting(choice,c,y,p,api,source) if scope=='full60' else None
            scores=metric(p,y);e=p-y;abs_e=np.abs(e)
            cs=np.sum(e*e,axis=(1,2));ps=np.sum(e*e,axis=(0,1));fs=np.sum(e*e,axis=(0,2))
            loc=np.unravel_index(np.argmax(abs_e),e.shape);wc=int(np.argmax(cs));wp=int(np.argmax(ps));wf=int(np.argmax(fs))
            extreme=dict(case_id=int(index[loc[0]]),frame_index=int(loc[1]),solver_index=int(shared['solver_index'][loc[1]]),point_index=int(loc[2]),point_id=int(geo['point_id'][loc[2]]),theta_m_absolute_deg=float(shared['theta_m_absolute_deg'][loc[1]]),time_s=float(shared['time_s'][loc[1]]),x_ref_mm=float(geo['xy'][loc[2],0]),y_ref_mm=float(geo['xy'][loc[2],1]),prediction_T=float(p[loc]),truth_T=float(y[loc]),absolute_error_T=float(abs_e[loc]))
            context=dict(id=cid,branch=choice['branch'],region=region,quantity=choice['quantity'],training_scope=scope,training_case_count=len(expected_training(split,scope)),partition=part,heldout_for_this_model=part in ['validation','reused_test'],original_validation_now_training=scope in ['final51','full60'])
            for j,i in enumerate(index):
                case_rows.append(dict(**context,case_id=int(i),**metric(p[j],y[j])))
                for frame in range(60):frame_rows.append(dict(**context,case_id=int(i),frame_index=frame,solver_index=frame+1,theta_m_absolute_deg=float(shared['theta_m_absolute_deg'][frame]),**metric(p[j,frame],y[j,frame],frame_count=1)))
            for point in range(npoints):point_rows.append(dict(**context,point_index=point,point_id=int(geo['point_id'][point]),x_ref_mm=float(geo['xy'][point,0]),y_ref_mm=float(geo['xy'][point,1]),**metric(p[:,:,point],y[:,:,point],point_count=1,frame_count=60)))
            metainfo=dict(version='NEW60_OPT_V4_SCIENTIFIC_REPLAY',model_id=cid,**{k:v for k,v in context.items() if k!='id'},choices_sha256=choices_sha,prediction_choices_sha256=prediction_sha,model_sha256=choice['model_sha256'],inference_module_sha256=choice['inference_module_sha256'],source_sha256=SOURCE_SHA,source_storage='original float32 Br/Bt converted to float64 before hypot',prediction_storage='float64 output',source_xy_units='mm',units='T',previous_test_already_seen=True,test_scope='reused9 retrospective evaluation, not fresh blind test',space_scope='actual sampled45-degree sector; no hole filling',time_scope='original60 first-electrical-period frames; no steady-state guarantee',metric_weight='equally weighted original points; not FEM area weights',minimum_inference_dependencies=['numpy','threadpoolctl'],no_training_or_selection_in_this_replay=True,metrics=scores)
            npz=None;artifact_sha=None;artifact_bytes=0;reuse=False
            if save:
                npz=source/'evaluation'/sourceid/f'{part}_predictions.npz';reuse=source!=ROOT.resolve();oldproof=None
                if reuse:
                    if source not in old_reports:
                        old_reports[source]=read(source/'qa/scientific_qa.json')
                        check(cid+' inherited science QA frozen identity',old_reports[source]['all_checks_pass'] is True and old_reports[source]['choices_sha256']==expected and old_reports[source]['source_sha256']==SOURCE_SHA)
                    oldproof=next(z for z in old_reports[source]['results'] if z['id']==sourceid and z['partition']==part)
                    check(cid+'/'+part+' inherited science exists',npz.is_file());reused+=1
                elif not npz.exists():
                    npz.parent.mkdir(exist_ok=True)
                    np.savez_compressed(npz,prediction=p,truth=y,case_ids=np.asarray(index,dtype=np.int32),original_row_indices=np.asarray(index,dtype=np.int64),partition=np.asarray(part),conditions=c,point_id=geo['point_id'],xy=geo['xy'],region=np.asarray(region),quantity=np.asarray(choice['quantity']),metadata_json=np.asarray(json.dumps(metainfo,ensure_ascii=False)),theta=shared['theta_m_absolute_deg'],solver_index=shared['solver_index'],time_s=shared['time_s'],theta_m_relative_deg=shared['theta_m_relative_deg'],theta_e_deg=shared['theta_e_deg'],initial_mechanical_deg=shared['initial_mechanical_deg'],speed_rpm=shared['speed_rpm'][index]);written+=1
                artifact_sha=verify_scientific(npz,choice,part,index,c,geo,shared,p,y,expected,sourceid,oldproof);artifact_bytes=npz.stat().st_size
            frame_error=float(np.max(np.abs(guarded_predict(api,choice,c[0],frame_index=59)-p[0,59])))
            check(cid+'/'+part+' batch/single stability',frame_error<1e-6,frame_error)
            result=dict(**context,pass_=True,metrics=scores,worst_case_id=int(index[wc]),worst_case_rmse_T=float(np.sqrt(cs[wc]/(60*npoints))),worst_frame_index=wf,worst_point_id=int(geo['point_id'][wp]),max_absolute_location=extreme,negative_raw_count=int(np.count_nonzero((raw if raw is not None else p)<0)),raw_fitting_metrics=metric(raw,y) if raw is not None else None,scientific_arrays_saved=save,scientific_arrays_reused=bool(reuse),scientific_npz_relative_path=npz.relative_to(source).as_posix() if npz else None,scientific_npz_sha256=artifact_sha,scientific_npz_bytes=artifact_bytes,scientific_source_relative_root=source.relative_to(WORK).as_posix() if save else ROOT.name,scientific_model_id=sourceid if save else cid,expected_scientific_choices_sha256=expected if save else choices_sha,source_sha256=SOURCE_SHA,model_sha256=choice['model_sha256'],inference_module_sha256=choice['inference_module_sha256'],choices_sha256=choices_sha)
            result['pass']=True;results.append(result)
            if part=='validation':
                frozen=choice['validation_selection_metrics'];frozen=frozen if 'rmse_T' in frozen else frozen['selected']['validation']
                check(cid+' frozen validation metric replay',np.isclose(scores['rmse_T'],frozen['rmse_T'],rtol=5e-7,atol=1e-10),dict(actual=scores['rmse_T'],frozen=frozen['rmse_T']))
            print(json.dumps(dict(id=cid,training_scope=scope,partition=part,relative_L2_pct=scores['relative_L2_pct'],rmse_T=scores['rmse_T'],max_abs_T=scores['max_abs_T'],scientific_arrays_saved=save,scientific_arrays_reused=reuse,scientific_bytes=artifact_bytes),ensure_ascii=False),flush=True)
            del p,y,raw,e,abs_e
    write_csv(OUT/'case_metrics.csv',case_rows);write_csv(OUT/'frame_metrics.csv',frame_rows);write_csv(OUT/'point_metrics.csv',point_rows)
    check('19 full-field metric partitions',len(results)==19)
    check('13 scientific partitions,7 immutable inherited',sum(r['scientific_arrays_saved'] for r in results)==13 and reused==7)
    check('training metrics no duplicate NPZ',all(not r['scientific_arrays_saved'] for r in results if r['partition']=='train'))
    check('final freeze unchanged',sha(ROOT/'both_versions_choices_frozen.json')==choices_sha and sha(ROOT/'prediction_choices_frozen.json')==prediction_sha);api.read_freeze(ROOT)
    report=dict(version='NEW60_OPT_V4_SCIENTIFIC_QA',all_checks_pass=True,choices_sha256=choices_sha,prediction_choices_sha256=prediction_sha,frozen_utc=registry['frozen_utc'],completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=SOURCE_SHA,source_script_sha256=sha(__file__),pass_count=len(checks),checks=checks,results=results,elapsed_seconds=time.time()-start,scientific_npz_count=13,new_scientific_npz_count=sum(r['scientific_arrays_saved'] and not r['scientific_arrays_reused'] for r in results),reused_scientific_npz_count=reused,new_scientific_npz_total_bytes=sum(r['scientific_npz_bytes'] for r in results if not r['scientific_arrays_reused']),reused_scientific_npz_total_bytes=sum(r['scientific_npz_bytes'] for r in results if r['scientific_arrays_reused']),scientific_npz_total_bytes=sum(r['scientific_npz_bytes'] for r in results),total_evaluation_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),label_access=accesses,minimum_inference_dependencies=['numpy','threadpoolctl'],test_scope='Previously seen original nine test conditions; retrospective replay after final freeze, no test reselection',scope_notes=dict(original42='train42/validation9/reused_test9',final51='original train42+validation9 are training; no independent validation; reused_test9',full60='inherited V3 all60 training fit only'),floating_limit_caution='All60 numerical equality is training-label reproduction, not new-condition accuracy or zero Maxwell error.',chronology_note='No V4 full60 head update; inherited V3 heads retain their V3 prediction-freeze metadata. No scientific arrays are copied from V3.')
    for path in [OUT/'scientific_qa.json',ROOT/'qa/scientific_qa.json']:
        with path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (ROOT/'qa/scientific_qa.md').open('x',encoding='utf-8') as f:
        f.write('# V4 冻结模型独立科学复评\n\n'+f'{len(checks)}项核验通过。19个全场指标分区，13个科学数组分区；新增6个NPZ，其余7个只读复核V3原文件。训练42/51只保存CSV和指标，不另存大型训练数组。\n\n')
        f.write('|模型|训练范围|分区|L2/%|RMSE/T|最大绝对误差/T|\n|---|---|---|---:|---:|---:|\n')
        for r in results:f.write(f'|{r["id"]}|{r["training_scope"]}|{r["partition"]}|{r["metrics"]["relative_L2_pct"]:.9g}|{r["metrics"]["rmse_T"]:.9g}|{r["metrics"]["max_abs_T"]:.9g}|\n')
        f.write('\n51训练版的原验证9已参加训练，没有独立验证分数。测试9为已见工况的冻结后复用回放；不得用于重新选型。全60拟合仅说明训练内标签数值重现。所有指标使用原float32组件转float64后计算，没有使用网页q16。推理依赖NumPy+threadpoolctl，已阻断Torch/sklearn/SciPy及源标签读取。\n')
    print(json.dumps(dict(all_checks_pass=True,pass_count=len(checks),new_scientific_npz_total_bytes=report['new_scientific_npz_total_bytes'],reused_scientific_npz_total_bytes=report['reused_scientific_npz_total_bytes'],report=str(ROOT/'qa/scientific_qa.json')),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-choices-sha256');parser.add_argument('--expected-prediction-sha256')
    parser.add_argument('--allow-final-label-replay',action='store_true',help='Use only after root explicitly authorizes finalized replay')
    main(parser.parse_args())

