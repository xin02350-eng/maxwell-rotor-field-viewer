"""Independently compare every V3 web pack with frozen scientific exports.

Run only after final freeze/evaluation and website export are authorized.
This checks display serialization; it does not infer, train or select models.
"""
from pathlib import Path
import argparse,base64,gzip,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(q):return {'vector':'Br,Bt','Br,Bt':'Br,Bt','planar_magnitude':'|B|','|B|':'|B|','Br':'Br'}[q]
def by_quantity(q,t,p):
    if q=='Br,Bt':return {'Br':(t[...,0::2],p[...,0::2]),'Bt':(t[...,1::2],p[...,1::2]),'magnitude':(np.hypot(t[...,0::2],t[...,1::2]),np.hypot(p[...,0::2],p[...,1::2]))}
    return {'Br' if q=='Br' else 'magnitude':(t,p)}
def metric(t,p):
    e=p-t;den=float(np.sum(t*t));sse=float(np.sum(e*e))
    return dict(rmse_T=float(np.sqrt(sse/e.size)),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),relative_L2_pct=100*float(np.sqrt(sse/den)) if den else None)

def run(root=ROOT):
    root=Path(root);site=root/'website';choices=read(root/'both_versions_choices_frozen.json');ch=sha(root/'both_versions_choices_frozen.json')
    manifest=read(site/'manifest.json');precision=read(site/'display_precision.json');results=read(site/'results.json')
    checks=[];count=0;max_residual_rounding=0.;max_field_rounding=0.
    def check(name,ok):
        checks.append({'name':name,'pass':bool(ok)})
        if not ok:raise AssertionError(name)
    def close_metrics(name,stored,expected):
        for key,v in expected.items():
            check(name+'/'+key,stored[key] is None if v is None else np.isclose(stored[key],v,rtol=2e-11,atol=1e-25))
    def decode(b64,offset,step,shape,original,name):
        nonlocal max_field_rounding,max_residual_rounding
        q=np.frombuffer(base64.b64decode(b64,validate=True),dtype='<u2')
        check(name+'/length',q.size==original.size and list(original.shape)==shape)
        check(name+'/affine parameters',np.isfinite(offset) and np.isfinite(step) and step>0)
        q=q.reshape(original.shape);expected=np.rint((original-offset)/step).clip(0,65535).astype('<u2')
        check(name+'/round-to-nearest LE',np.array_equal(q,expected))
        restored=q.astype(np.float64)*step+offset;maximum=float(np.max(np.abs(restored-original)))
        roundoff=8*np.finfo(float).eps*max(float(np.max(np.abs(original))),abs(offset),1e-300)
        check(name+'/half-step error bound',maximum<=step/2+roundoff)
        if '/residual/' in name:max_residual_rounding=max(max_residual_rounding,maximum)
        else:max_field_rounding=max(max_field_rounding,maximum)
        return maximum
    check('schema3',manifest['schema_version']==3)
    check('same freeze',manifest['choices_sha256']==ch==results['choices_sha256'])
    check('case60',len(manifest['cases'])==60 and len({str(x['case_id']) for x in manifest['cases']})==60)
    check('fitting default',manifest['defaults']['mode']=='fitting_full60')
    index={x['path']:x for x in precision['payloads']};scientific_sources=[]
    cases={str(c['case_id']):c for c in manifest['cases']}
    for model in choices['models']:
        mid=model['id'];mode=model['branch'];region=model['region'];quantity=canonical(model['quantity'])
        rm=manifest['modes'][mode]['regions'][region];geo=read(site/rm['geometry']);mm=next(m for m in rm['models'] if m['key']==mid)
        check(mid+'/model identity',mm['model_sha256']==model['model_sha256'] and mm['quantity']==quantity and mm['label']==model['display_label'])
        parts=['training_fit'] if mode=='fitting_full60' else ['train','validation','reused_test'];seen=[]
        for part in parts:
            src=root/'evaluation'/mid/(part+'_predictions.npz');scientific_sources.append({'path':src.relative_to(root).as_posix(),'sha256':sha(src)})
            with np.load(src,allow_pickle=False) as z:
                t=z['truth'];p=z['prediction'];ids=z['case_ids'];points=z['point_id'];xy=z['xy'];theta=z['theta'];conditions=z['conditions'];meta=read_json_string(z['metadata_json'])
            check(mid+'/'+part+'/source freeze',meta['choices_sha256']==ch and meta['model_sha256']==model['model_sha256'])
            check(mid+'/'+part+'/geometry',np.array_equal(points,geo['point_id']) and np.array_equal(xy,geo['xy']) and np.array_equal(theta,manifest['frames']['theta_absolute_deg']))
            for i,cid in enumerate(ids.tolist()):
                name=mid+'/'+str(cid);rel=rm['cases'][str(cid)][mid];path=site/rel;row=index[rel]
                raw=json.loads(gzip.decompress(path.read_bytes()));seen.append(cid);count+=1
                check(name+'/file SHA and bytes',sha(path)==row['sha256'] and path.stat().st_size==row['bytes'])
                expected_split='test' if part=='reused_test' else part
                if part!='training_fit':check(name+'/partition',raw['original_split']==expected_split)
                c=cases[str(cid)]
                check(name+'/native input and original split',np.array_equal(conditions[i],[c['I_rms_A'],c['Thet_deg']]) and raw['original_split']==c['split'])
                check(name+'/identity and dimensions',raw['mode']==mode and raw['region']==region and raw['units']=='T' and raw['model_key']==mid and raw['case_id']==cid and raw['quantity']==quantity and raw['model_sha256']==model['model_sha256'] and raw['point_id']==points.tolist() and raw['shape']==list(t[i].shape))
                for key,a in [('truth',t[i]),('prediction',p[i])]:
                    maximum=decode(raw[key+'_q16_b64'],raw['offset'][key],raw['scale'][key],raw['shape'],a,name+'/'+key)
                    check(name+'/'+key+'/recorded rounding',np.isclose(maximum,raw['quantization_max_abs_T'][key],rtol=1e-12,atol=1e-25))
                for component,(ct,cp) in by_quantity(quantity,t[i],p[i]).items():
                    r=raw['residuals'][component];error=cp-ct
                    maximum=decode(r['q16_b64'],r['offset'],r['scale'],r['shape'],error,name+'/residual/'+component)
                    check(name+'/'+component+'/recorded residual rounding',np.isclose(maximum,r['max_abs_quantization_T'],rtol=1e-10,atol=1e-25))
                    close_metrics(name+'/'+component+'/case',raw['metrics'][component]['aggregate'],metric(ct,cp))
                    expected_frames=[metric(ct[f],cp[f]) for f in range(60)]
                    check(name+'/'+component+'/60 frame metrics',len(raw['metrics'][component]['frames'])==60)
                    for key in ['rmse_T','mae_T','max_abs_T','relative_L2_pct']:
                        check(name+'/'+component+'/all60frames/'+key,
                            all(s[key] is None if e[key] is None else np.isclose(s[key],e[key],rtol=2e-11,atol=1e-25)
                                for s,e in zip(raw['metrics'][component]['frames'],expected_frames)))
                    lo,hi=mm['scales'][component]['field'];elo,ehi=mm['scales'][component]['error']
                    check(name+'/'+component+'/color bounds',min(ct.min(),cp.min())>=lo and max(ct.max(),cp.max())<=hi and elo==0 and np.max(np.abs(error))<=ehi)
        check(mid+'/all60 unique',len(seen)==60 and len(set(seen))==60)
    check('every registered payload checked',count==len(index)==len(choices['models'])*60)
    out={'status':'PASS','all_checks_pass':True,'scope':'Real full web serialization audit against frozen original-precision scientific arrays; not browser interaction or model selection',
         'choices_sha256':ch,'payloads_checked':count,'checks_count':len(checks),'checks':checks,
         'maximum_field_display_error_T':max_field_rounding,'maximum_independent_residual_display_error_T':max_residual_rounding,
         'manifest_sha256':sha(site/'manifest.json'),'app_sha256':sha(site/'app.js'),'source_sha256':sha(Path(__file__)),
         'scientific_sources':scientific_sources}
    target=root/'qa/website_payload_v3_check.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'qa/website_payload_v3_check.md').write_text(f'# V3 网页载荷核验\n\nPASS：{count} 个工况包、{len(checks)} 项检查。逐点检查真值、预测与原精度独立残差的LE编码/舍入/误差界限，逐帧核对原精度指标、工况身份、点序、坐标、SHA和色标。此报告不替代真实浏览器交互检查。\n',encoding='utf-8')
    return {k:v for k,v in out.items() if k not in {'checks','scientific_sources'}}

def read_json_string(value):return json.loads(str(value))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    print(json.dumps(run(args.root),ensure_ascii=True))
