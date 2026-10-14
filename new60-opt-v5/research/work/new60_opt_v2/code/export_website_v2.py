"""Convert completed frozen QA scientific arrays to relative, per-case web payloads.

No inference, fitting, model selection or label-cache access. Every displayed
metric is calculated from the saved float64 scientific prediction and truth,
before uint16 display quantization. Existing manifest/data are never replaced.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')

def metrics(t,p):
    t=np.asarray(t,dtype=np.float64);e=np.asarray(p,dtype=np.float64)-t
    den=float(np.sum(t*t));sse=float(np.sum(e*e))
    return dict(rmse_T=float(np.sqrt(sse/e.size)),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),relative_L2_pct=float(np.sqrt(sse/den)*100) if den else None)

def components(quantity,t,p):
    if quantity=='Br,Bt':return {'Br':(t[...,0::2],p[...,0::2]),'Bt':(t[...,1::2],p[...,1::2]),'magnitude':(np.hypot(t[...,0::2],t[...,1::2]),np.hypot(p[...,0::2],p[...,1::2]))}
    return {'Br' if quantity=='Br' else 'magnitude':(t,p)}

def quantize(a):
    a=np.asarray(a,dtype=np.float64);lo=float(a.min());hi=float(a.max());step=(hi-lo)/65535 if hi>lo else 1e-12
    q=np.rint((a-lo)/step).clip(0,65535).astype('<u2');back=q.astype(np.float64)*step+lo
    maximum=float(np.max(np.abs(a-back)))
    if maximum>step/2+4*np.finfo(float).eps*max(1,abs(lo),abs(hi)):raise AssertionError('uint16 rounding bound failed')
    return base64.b64encode(q.tobytes()).decode('ascii'),lo,step,maximum

def canonical(q):
    return {'vector':'Br,Bt','Br,Bt':'Br,Bt','planar_magnitude':'|B|','|B|':'|B|','Br':'Br'}[q]

LABELS={
 'prediction_rotor_magnitude':'主云图 · 核模型 Br/Bt → |B|',
 'prediction_stator_vector':'有符号 Br/Bt · 核模型',
 'prediction_stator_magnitude':'主云图 · 核模型 Br/Bt → |B|',
 'prediction_airgap_Br':'主云图 · 核模型 Br',
 'prediction_rotor_vector':'有符号 Br/Bt · 纯神经网络',
 'prediction_rotor_magnitude_nn':'纯神经网络 Br/Bt → |B|',
 'prediction_rotor_magnitude_direct_nn':'直接 |B| 神经网络（原始输出）',
 'prediction_rotor_magnitude_neural_kernel':'神经网络特征 + 核模型 |B|',
 'fitting_rotor':'全部60训练 · |B| 神经网络',
 'fitting_stator':'全部60训练 · |B| 神经网络',
 'fitting_airgap':'全部60训练 · Br 神经网络'}

def build(root=ROOT,qa_reports=(),output=None):
    root=Path(root).resolve();output=Path(output or root/'website').resolve()
    choices_path=root/'both_versions_choices_frozen.json';choices=read(choices_path);choices_sha=sha(choices_path)
    sidecar=root/'both_versions_choices_frozen.sha256'
    if not sidecar.is_file() or choices_sha not in sidecar.read_text():raise ValueError('Frozen choices SHA missing/mismatch')
    if not qa_reports:raise ValueError('Pass completed QA reports with --qa-report')
    reports=[]
    for path in qa_reports:
        path=Path(path).resolve();report=read(path)
        if report.get('all_checks_pass') is not True or report.get('choices_sha256')!=choices_sha:raise ValueError(f'QA incomplete/failed or different freeze: {path}')
        reports.append(dict(path=str(path.relative_to(root)),sha256=sha(path),report=report))
    passed={r['id'] for rep in reports for r in rep['report'].get('results',[]) if r.get('pass') is True}
    if any(m['id'] not in passed for m in choices['models']):raise ValueError('QA does not cover every frozen model')
    for name in ['manifest.json','data','results.json','display_precision.json']:
        if (output/name).exists():raise FileExistsError(output/name)
    output.mkdir(parents=True,exist_ok=True)
    split_path=(root/choices['split_relative_path']).resolve()
    if sha(split_path)!=choices['split_sha256']:raise ValueError('Split SHA mismatch')
    split=read(split_path)
    split_lookup={str(case):part for part in ['train','validation','test'] for case in split[part]}
    manifest=dict(schema_version=2,dataset='NEW60_OPT_V2',created_utc=datetime.now(timezone.utc).isoformat(),choices_sha256=choices_sha,
       defaults=dict(mode='fitting_full60',region='rotor',case_id=2,frame=0),cases=[],frames={},modes={mode:dict(regions={}) for mode in ['prediction','fitting_full60']},
       previous_test_already_seen=True,test_is_fresh_blind=False,notes=['Fitting: all60 training-in-sample only.','Prediction: original42/9/9; old test reused.','3D is field height / T, not a volume field.','Fixed sampled sector and 60 frames; no mirroring or unseen-position claim.'])
    results=dict(choices_sha256=choices_sha,scope=manifest['notes'],qa=[{k:v for k,v in r.items() if k!='report'} for r in reports],models=[])
    precision=[];cases={};geometries={};common=None
    for choice in choices['models']:
        mid=choice['id'];region=choice['region'];mode=choice['branch'];quantity=canonical(choice['quantity'])
        if sha(root/choice['model_relative_path'])!=choice['model_sha256']:raise ValueError('Model changed after freeze: '+mid)
        parts=['training_fit'] if mode=='fitting_full60' else ['train','validation','reused_test']
        loaded=[];sources=[];partition_metrics={}
        for part in parts:
            path=root/'evaluation'/mid/(part+'_predictions.npz')
            with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            meta=json.loads(str(a['metadata_json']))
            if meta.get('choices_sha256')!=choices_sha or meta.get('model_sha256')!=choice['model_sha256']:raise ValueError('Scientific artifact identity mismatch: '+str(path))
            if canonical(str(a['quantity']))!=quantity:raise ValueError('Quantity mismatch')
            t=np.asarray(a['truth'],dtype=np.float64);p=np.asarray(a['prediction'],dtype=np.float64)
            if t.shape!=p.shape or t.ndim!=3 or t.shape[1]!=60 or not np.isfinite(t).all() or not np.isfinite(p).all():raise ValueError('Scientific arrays invalid')
            n=len(a['point_id']);d=n*(2 if quantity=='Br,Bt' else 1)
            if t.shape[2]!=d:raise ValueError('Point count does not match field dimension')
            xy=np.asarray(a['xy']);ids=np.asarray(a['point_id']);theta=np.asarray(a['theta']);time_s=np.asarray(a['time_s'])
            if xy.shape!=(n,2):raise ValueError('xy must be [points,2]')
            g=dict(point_id=ids.tolist(),xy=xy.tolist(),units='mm',sample_scope='Actual sampled sector only; no mirrored completion or cross-hole interpolation')
            if region in geometries and geometries[region]!=g:raise ValueError('Geometry mismatch across models')
            geometries[region]=g
            frame_dict=dict(theta_absolute_deg=theta.tolist(),time_s=time_s.tolist())
            if common is not None and common!=frame_dict:raise ValueError('Frame coordinates mismatch')
            common=frame_dict
            partition_metrics[part]={key:metrics(truth,prediction) for key,(truth,prediction) in components(quantity,t,p).items()}
            if quantity=='Br,Bt':partition_metrics[part]['signed_vector']=metrics(t,p)
            for i,cid in enumerate(a['case_ids'].tolist()):
                original=split_lookup.get(str(cid))
                if original is None:raise ValueError('Original case ID missing in split')
                if part!='training_fit' and original!=('test' if part=='reused_test' else part):raise ValueError('Partition differs from original split')
                condition=np.asarray(a['conditions'][i],dtype=float)
                c=dict(case_id=cid,split=original,I_rms_A=float(condition[0]),Thet_deg=float(condition[1]))
                if str(cid) in cases and cases[str(cid)]!=c:raise ValueError('Case ID/input mismatch')
                cases[str(cid)]=c;loaded.append((cid,t[i],p[i]))
            sources.append(dict(path=str(path.relative_to(root)),sha256=sha(path),partition=part,shape=list(t.shape)))
        if len(loaded)!=60 or len({str(x[0]) for x in loaded})!=60:raise ValueError('Each model must have exactly all60 case IDs')
        r=manifest['modes'][mode]['regions'].setdefault(region,dict(geometry=f'data/{region}_geometry.json',models=[],cases={},default_model=None))
        note=''
        if mode=='fitting_full60':note='所有工况参加训练；仅训练内拟合误差。'
        if mode=='fitting_full60' and quantity=='|B|':note+='网页遵循冻结模型 nonnegative=True 的非负输出；原始未裁剪输出误差及负值计数另见拟合报告。'
        if mid=='prediction_rotor_magnitude_direct_nn':note+='直接大小网络使用冻结的原始输出，未静默裁剪负值。'
        m=dict(key=mid,label=LABELS.get(mid,mid),quantity=quantity,model_sha256=choice['model_sha256'],origin=choice.get('family',choice.get('inference_module_relative_path','frozen model')),note=note,scales={})
        ranges={}
        for cid,t,p in loaded:
            for key,(ct,cp) in components(quantity,t,p).items():
                lo=min(float(ct.min()),float(cp.min()),0.);hi=max(float(ct.max()),float(cp.max()));err=float(np.max(np.abs(cp-ct)))
                old=ranges.setdefault(key,[lo,hi,err]);old[:]=[min(old[0],lo),max(old[1],hi),max(old[2],err)]
        for key,(lo,hi,err) in ranges.items():
            if key in ['Br','Bt']:hi=max(abs(lo),abs(hi),1e-12);lo=-hi
            m['scales'][key]=dict(field=[lo,max(hi,lo+1e-12)],error=[0,max(err,1e-12)])
        r['models'].append(m)
        if r['default_model'] is None or mid in ['prediction_rotor_magnitude','prediction_stator_magnitude','prediction_airgap_Br']:r['default_model']=mid
        for cid,t,p in loaded:
            tq,to,ts,te=quantize(t);pq,po,ps,pe=quantize(p)
            ms={key:dict(aggregate=metrics(ct,cp),frames=[metrics(ct[i],cp[i]) for i in range(60)]) for key,(ct,cp) in components(quantity,t,p).items()}
            payload=dict(mode=mode,region=region,model_key=mid,model_sha256=choice['model_sha256'],case_id=cid,quantity=quantity,units='T',original_split=cases[str(cid)]['split'],
                point_id=geometries[region]['point_id'],shape=list(t.shape),layout='interleaved' if quantity=='Br,Bt' else 'scalar',
                truth_q16_b64=tq,prediction_q16_b64=pq,offset=dict(truth=to,prediction=po),scale=dict(truth=ts,prediction=ps),metrics=ms,
                quantization_max_abs_T=dict(truth=te,prediction=pe),metric_precision='unquantized float64 scientific export')
            rel=f'data/{mode}/{mid}/case_{cid}.json.gz';path=output/rel;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8'),compresslevel=7,mtime=0))
            r['cases'].setdefault(str(cid),{})[mid]=rel
            precision.append(dict(path=rel,truth_step_T=ts,prediction_step_T=ps,truth_max_abs_T=te,prediction_max_abs_T=pe,bytes=path.stat().st_size,sha256=sha(path)))
        results['models'].append(dict(id=mid,branch=mode,region=region,quantity=quantity,model_sha256=choice['model_sha256'],sources=sources,metrics=partition_metrics,note=note))
    if len(cases)!=60:raise ValueError('Expected60 original cases')
    manifest['cases']=sorted(cases.values(),key=lambda x:x['case_id']);manifest['frames']=common
    for region,g in geometries.items():write(output/'data'/f'{region}_geometry.json',g)
    write(output/'results.json',results)
    write(output/'display_precision.json',dict(quantization='affine uint16 little-endian; float64 round-to-nearest',metrics='computed before quantization',payload_count=len(precision),payloads=precision))
    write(output/'manifest.json',manifest) # Written last; incomplete exports cannot look ready.
    return dict(payload_count=len(precision),payload_bytes=sum(p['bytes'] for p in precision),models=len(choices['models']),manifest_sha256=sha(output/'manifest.json'),choices_sha256=choices_sha,output=str(output))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);p.add_argument('--qa-report',type=Path,action='append',required=True)
    a=p.parse_args();print(json.dumps(build(a.root,a.qa_report,a.output),ensure_ascii=False))

if __name__=='__main__':main()
