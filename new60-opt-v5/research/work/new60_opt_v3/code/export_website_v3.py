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
import html
from datetime import datetime, timezone
import numpy as np
from comparison_narrative_v3 import EXPERIMENT_COUNTS,comparison_text

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

def signed_residuals(quantity,t,p):
    """Use unquantized component errors, including norm error, not rounded p-t."""
    return {key:prediction-truth for key,(truth,prediction) in components(quantity,t,p).items()}

def packed_residuals(quantity,t,p):
    packed={}
    for key,error in signed_residuals(quantity,t,p).items():
        b64,offset,scale,maximum=quantize(error)
        packed[key]=dict(q16_b64=b64,offset=offset,scale=scale,max_abs_quantization_T=maximum,
                         shape=list(error.shape),definition='prediction minus truth before display quantization')
    return packed

def model_label(choice):
    # Never infer NN/kernel attribution from an ID: V3 may reuse V2 model IDs.
    if not choice.get('display_label') or not choice.get('family'):
        raise ValueError('Each frozen model needs explicit display_label and family: '+choice['id'])
    return str(choice['display_label'])

def safe_artifact(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.parent) or not p.is_file() or p.is_symlink():
        raise ValueError('Frozen artifact missing/outside the study workspace: '+str(relative))
    return p

def generate_guide(output,manifest,results,root=None):
    """Generate final tables only from completed original-precision exports."""
    rows=[]
    for model in results['models']:
        part='training_fit' if model['branch']=='fitting_full60' else 'reused_test'
        key='Br' if model['quantity']=='Br' else 'magnitude'
        m=model['metrics'][part][key]
        esc=lambda x:html.escape(str(x))
        rows.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [model['label'],
            '全部60训练内拟合' if part=='training_fit' else '9个复用测试工况',
            model['quantity'],f"{m['relative_L2_pct']:.10g}",f"{m['rmse_T']:.10g}",f"{m['max_abs_T']:.10g}"])+'</tr>')
    template=(Path(__file__).parent/'templates/website/guide.html').read_text(encoding='utf-8')
    template=template.replace('<!-- ORIGINAL_PRECISION_RESULT_ROWS -->','\n'.join(rows))
    template=template.replace('<!-- FROZEN_CHOICES_SHA -->',html.escape(manifest['choices_sha256']))
    template=template.replace('<!-- EXPERIMENT_COUNTS -->',html.escape(EXPERIMENT_COUNTS))
    template=template.replace('<!-- V2_TO_V3_COMPARISON -->',html.escape(comparison_text(root)) if root else '')
    (output/'guide.html').write_text(template,encoding='utf-8')

def build(root=ROOT,qa_reports=(),output=None):
    root=Path(root).resolve();output=Path(output or root/'website').resolve()
    choices_path=root/'both_versions_choices_frozen.json';choices=read(choices_path);choices_sha=sha(choices_path)
    sidecar=root/'both_versions_choices_frozen.sha256'
    if not sidecar.is_file() or choices_sha not in sidecar.read_text():raise ValueError('Frozen choices SHA missing/mismatch')
    for ref in choices.get('support_files',[]):
        if sha(safe_artifact(root,ref['relative_path']))!=ref['sha256']:raise ValueError('Frozen support source changed: '+ref['relative_path'])
    if not qa_reports:raise ValueError('Pass completed QA reports with --qa-report')
    reports=[]
    for path in qa_reports:
        path=Path(path).resolve();report=read(path)
        if report.get('all_checks_pass') is not True or report.get('choices_sha256')!=choices_sha:raise ValueError(f'QA incomplete/failed or different freeze: {path}')
        reports.append(dict(path=str(path.relative_to(root)),sha256=sha(path),report=report))
    passed={r['id'] for rep in reports for r in rep['report'].get('results',[]) if r.get('pass') is True}
    if any(m['id'] not in passed for m in choices['models']):raise ValueError('QA does not cover every frozen model')
    for name in ['manifest.json','data','results.json','display_precision.json','index.html','app.js']:
        if (output/name).exists():raise FileExistsError(output/name)
    output.mkdir(parents=True,exist_ok=True)
    split_path=safe_artifact(root,choices['split_relative_path'])
    if sha(split_path)!=choices['split_sha256']:raise ValueError('Split SHA mismatch')
    split=read(split_path)
    original_ids=split.get('original_ids',split)
    split_lookup={str(case):part for part in ['train','validation','test'] for case in original_ids[part]}
    manifest=dict(schema_version=3,dataset='NEW60_OPT_V3',created_utc=datetime.now(timezone.utc).isoformat(),choices_sha256=choices_sha,
       defaults=dict(mode='fitting_full60',region='rotor',case_id=2,frame=0),cases=[],frames={},modes={mode:dict(regions={}) for mode in ['prediction','fitting_full60']},
       previous_test_already_seen=True,test_is_fresh_blind=False,
       error_display='Signed residual is computed from original-precision arrays and quantized independently; error plots use its absolute value, never rounded prediction minus rounded truth.',
       notes=['Fitting: all60 training-in-sample; existing NN hidden features, final linear head calibrated by least squares. Near-machine-precision fitting is not predictive accuracy.',
              'Prediction: original42/9/9; old test reused.','3D is field height / T, not a volume field.',
              'Fixed sampled 45-degree sector and 60 frames; no mirroring or unseen-position claim.'])
    results=dict(choices_sha256=choices_sha,scope=manifest['notes'],qa=[{k:v for k,v in r.items() if k!='report'} for r in reports],models=[])
    precision=[];cases={};geometries={};common=None
    for choice in choices['models']:
        mid=choice['id'];region=choice['region'];mode=choice['branch'];quantity=canonical(choice['quantity'])
        label=model_label(choice)
        if sha(safe_artifact(root,choice['model_relative_path']))!=choice['model_sha256']:raise ValueError('Model changed after freeze: '+mid)
        for pk,hk in [('inference_module_relative_path','inference_module_sha256'),('model_source_module_relative_path','model_source_module_sha256')]:
            if pk in choice and sha(safe_artifact(root,choice[pk]))!=choice[hk]:raise ValueError('Inference source changed: '+mid)
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
        if mode=='fitting_full60':note='全部60工况参与线性头最小二乘校准；冻结已训练NN隐层，仅报告训练内拟合误差，不能代表未见工况预测。'
        note+=str(choice.get('display_note',choice.get('note','')))
        m=dict(key=mid,label=label,quantity=quantity,model_sha256=choice['model_sha256'],origin=choice['family'],note=note,scales={})
        ranges={}
        for cid,t,p in loaded:
            for key,(ct,cp) in components(quantity,t,p).items():
                lo=min(float(ct.min()),float(cp.min()),0.);hi=max(float(ct.max()),float(cp.max()));err=float(np.max(np.abs(cp-ct)))
                old=ranges.setdefault(key,[lo,hi,err]);old[:]=[min(old[0],lo),max(old[1],hi),max(old[2],err)]
        for key,(lo,hi,err) in ranges.items():
            if key in ['Br','Bt']:hi=max(abs(lo),abs(hi),1e-12);lo=-hi
            m['scales'][key]=dict(field=[lo,max(hi,lo+1e-12)],error=[0,err if err>0 else 1e-16])
        r['models'].append(m)
        if r['default_model'] is None or choice.get('role')=='primary':r['default_model']=mid
        for cid,t,p in loaded:
            tq,to,ts,te=quantize(t);pq,po,ps,pe=quantize(p)
            ms={key:dict(aggregate=metrics(ct,cp),frames=[metrics(ct[i],cp[i]) for i in range(60)]) for key,(ct,cp) in components(quantity,t,p).items()}
            payload=dict(mode=mode,region=region,model_key=mid,model_sha256=choice['model_sha256'],case_id=cid,quantity=quantity,units='T',original_split=cases[str(cid)]['split'],
                point_id=geometries[region]['point_id'],shape=list(t.shape),layout='interleaved' if quantity=='Br,Bt' else 'scalar',
                truth_q16_b64=tq,prediction_q16_b64=pq,offset=dict(truth=to,prediction=po),scale=dict(truth=ts,prediction=ps),metrics=ms,
                residuals=packed_residuals(quantity,t,p),residual_display='original-precision signed error, independently quantized',
                quantization_max_abs_T=dict(truth=te,prediction=pe),metric_precision='unquantized float64 scientific export')
            rel=f'data/{mode}/{mid}/case_{cid}.json.gz';path=output/rel;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8'),compresslevel=7,mtime=0))
            r['cases'].setdefault(str(cid),{})[mid]=rel
            precision.append(dict(path=rel,truth_step_T=ts,prediction_step_T=ps,truth_max_abs_T=te,prediction_max_abs_T=pe,
                                  residuals={key:{k:v for k,v in r.items() if k!='q16_b64'} for key,r in payload['residuals'].items()},
                                  bytes=path.stat().st_size,sha256=sha(path)))
        results['models'].append(dict(id=mid,label=label,family=choice['family'],branch=mode,region=region,quantity=quantity,model_sha256=choice['model_sha256'],sources=sources,metrics=partition_metrics,note=note))
    if len(cases)!=60:raise ValueError('Expected60 original cases')
    manifest['cases']=sorted(cases.values(),key=lambda x:x['case_id']);manifest['frames']=common
    for region,g in geometries.items():write(output/'data'/f'{region}_geometry.json',g)
    write(output/'results.json',results)
    write(output/'display_precision.json',dict(quantization='affine uint16 little-endian; float64 round-to-nearest',metrics='computed before quantization',
          residuals='Computed before field quantization, independent per-component signed residual encoding.',payload_count=len(precision),payloads=precision))
    for name in ['index.html','app.js','styles.css','favicon.svg']:
        shutil.copyfile(Path(__file__).parent/'templates/website'/name,output/name)
    generate_guide(output,manifest,results,root)
    write(output/'manifest.json',manifest) # Written last; incomplete exports cannot look ready.
    return dict(payload_count=len(precision),payload_bytes=sum(p['bytes'] for p in precision),models=len(choices['models']),manifest_sha256=sha(output/'manifest.json'),choices_sha256=choices_sha,output=str(output))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);p.add_argument('--qa-report',type=Path,action='append',required=True)
    a=p.parse_args();print(json.dumps(build(a.root,a.qa_report,a.output),ensure_ascii=False))

if __name__=='__main__':main()
