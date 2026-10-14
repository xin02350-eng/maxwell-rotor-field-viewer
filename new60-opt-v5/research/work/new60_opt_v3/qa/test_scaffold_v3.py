"""Temporary synthetic components only: no real labels, training, or evaluation."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import export_website_v3 as exporter
import build_delivery_v3 as builder
import check_web_payloads_v3 as payload_checker

checks=[]
def check(name,ok):
    checks.append({'name':name,'pass':bool(ok)})
    if not ok:raise AssertionError(name)

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False),encoding='utf-8')

for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'));check('Python AST '+p.name,True)
node=shutil.which('node') or r'C:\Program Files\nodejs\node.exe'
subprocess.run([node,'--check',str(ROOT/'code/templates/website/app.js')],check=True)
check('JS syntax',True)

with tempfile.TemporaryDirectory(prefix='maxwell_v3_synthetic_components_') as temp:
    temp=Path(temp);root=temp/'v3';root.mkdir()
    val=[1,9,14,15,24,27,35,40,51];test=[2,4,12,16,17,18,38,41,58]
    split={'train':[i for i in range(60) if i not in val+test],'validation':val,'test':test}
    dump(root/'split.json',split)
    model_specs=[('prediction_rotor_magnitude','prediction','rotor'),('prediction_stator_magnitude','prediction','stator'),
      ('prediction_airgap_Br','prediction','airgap'),('prediction_rotor_gated_nn','prediction','rotor'),
      ('prediction_rotor_v2_kernel','prediction','rotor'),('fitting_rotor_magnitude','fitting_full60','rotor'),
      ('fitting_stator_magnitude','fitting_full60','stator'),('fitting_airgap_Br','fitting_full60','airgap')]
    models=[]
    for i,(mid,branch,region) in enumerate(model_specs):
        model=root/'models'/f'{mid}.npz';model.parent.mkdir(exist_ok=True);model.write_bytes(b'opaque synthetic model, never decoded '+mid.encode())
        models.append({'id':mid,'branch':branch,'region':region,'quantity':'Br' if region=='airgap' else 'planar_magnitude',
            'model_relative_path':f'models/{mid}.npz','model_sha256':exporter.sha(model),'display_label':f'合成组件模型 {mid}',
            'family':'synthetic_component_only','role':'primary' if i<3 or i>4 else 'control'})
    choices={'models':models,'split_relative_path':'split.json','split_sha256':exporter.sha(root/'split.json')}
    dump(root/'both_versions_choices_frozen.json',choices);ch=exporter.sha(root/'both_versions_choices_frozen.json')
    (root/'both_versions_choices_frozen.sha256').write_text(ch)
    dump(root/'qa.json',{'all_checks_pass':True,'choices_sha256':ch,'results':[{'id':m['id'],'pass':True} for m in models]})
    theta=9+1.5*np.arange(60);time=np.arange(1,61)/22000.;ids=np.array([101,102,103]);xy=np.array([[1.,0.],[2.,.5],[1.5,1.]])
    for m in models:
        parts={'training_fit':list(range(60))} if m['branch']=='fitting_full60' else {'train':split['train'],'validation':val,'reused_test':test}
        for part,cids in parts.items():
            case=np.asarray(cids);idx=np.arange(len(cids)*60*3).reshape(len(cids),60,3)
            truth=(.2 if m['region']=='airgap' else 2.)+np.sin(idx*.11)*.1+case[:,None,None]*.001
            pred=truth+np.cos(idx*.23)*(1e-12 if m['branch']=='fitting_full60' else .007)
            out=root/'evaluation'/m['id']/(part+'_predictions.npz');out.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out,truth=truth,prediction=pred,case_ids=case,conditions=np.column_stack([case*3.,case*.5]),
                point_id=ids,xy=xy,theta=theta,time_s=time,quantity=m['quantity'],
                metadata_json=json.dumps({'choices_sha256':ch,'model_sha256':m['model_sha256']}))
    result=exporter.build(root,[root/'qa.json'])
    check('8 synthetic models x60 payloads',result['payload_count']==480)
    check('manifest schema3',json.loads((root/'website/manifest.json').read_text(encoding='utf-8'))['schema_version']==3)
    check('guide generated final table', '<!-- ORIGINAL_PRECISION_RESULT_ROWS -->' not in (root/'website/guide.html').read_text(encoding='utf-8'))
    check('does not publish a real V3 website',not (ROOT/'website').exists())
    (root/'qa').mkdir()
    web_audit=payload_checker.run(root)
    check('all480 synthetic packs independent serialization audit',web_audit['all_checks_pass'] and web_audit['payloads_checked']==480)
    try:exporter.build(root,[root/'qa.json'])
    except FileExistsError:check('exporter refuses replacement',True)
    else:check('exporter refuses replacement',False)
    vector=np.arange(120,dtype=float).reshape(60,2)/100
    pred=vector+np.cos(vector)*1e-12
    residual=exporter.packed_residuals('Br,Bt',vector,pred)
    check('component and norm residuals',set(residual)=={'Br','Bt','magnitude'})
    expected=np.hypot(pred[:,0::2],pred[:,1::2])-np.hypot(vector[:,0::2],vector[:,1::2])
    r=residual['magnitude'];import base64
    actual=np.frombuffer(base64.b64decode(r['q16_b64']),dtype='<u2').astype(float)*r['scale']+r['offset']
    check('near-zero magnitude residual precision',np.max(np.abs(actual.reshape(expected.shape)-expected))<=r['scale']/2+1e-30)
    node_result=subprocess.run([node,str(ROOT/'qa/test_viewer_components_v3.js'),str(root/'website')],capture_output=True,text=True)
    print(node_result.stdout,flush=True)
    if node_result.returncode:print(node_result.stderr);raise RuntimeError('Node component simulation failed')
    check('Node async interaction simulation',True)
    for unsafe in ['../outside','new60_opt_v3/../../escape','/absolute','C:/x','new60_opt_v3/.git/config','new60_opt_v3/__pycache__/x.pyc']:
        try:builder.safe_relative(unsafe)
        except ValueError:check('reject unsafe '+unsafe,True)
        else:check('reject unsafe '+unsafe,False)
    context={'choices_sha256':ch,'website_deployment':None}
    markdown=builder.report(root,context,{})
    word=temp/'preview.docx';builder.docx(word,markdown)
    with zipfile.ZipFile(word) as z:
        for name in z.namelist():ET.fromstring(z.read(name))
        doc=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        rows=doc.findall('.//w:tbl/w:tr',ns)
        check('Word rows including pipe quantities have6columns',len(rows)==9 and all(len(r.findall('w:tc',ns))==6 for r in rows))
    check('all Word XML parses',True)
    # Only streaming mechanics are mocked at readiness boundary; real hash,
    # opaque bytes copy, CRC, SHA, final rename and refusal are executed.
    plan_path=root/'delivery_dependencies.json';dump(plan_path,{'omitted':[]})
    src=root/'models'/f"{models[0]['id']}.npz";h=builder.digest(src)
    builder.preflight=lambda *args,**kwargs:dict(context,ready=True,bytes=src.stat().st_size)
    builder.verify_choices=lambda *args,**kwargs:({},ch)
    builder.source_plan=lambda *args,**kwargs:[(src,'v3/models/model.npz',{'role':'model'})]
    out=temp/'small_delivery';packed=builder.build(root,plan_path,out)
    check('stream verifies CRC/SHA/bytes',packed['all_member_sha_crc_pass'])
    check('opaque source unchanged',builder.digest(src)==h)
    check('no duplicated directory tree',not (out/'v3').exists())
    with zipfile.ZipFile(packed['zip']) as z:
        check('single archive root',all(n.startswith('small_delivery/') for n in z.namelist()))
    try:builder.build(root,plan_path,out)
    except FileExistsError:check('builder refuses overwrite',True)
    else:check('builder refuses overwrite',False)

result={'status':'PASS','validation_stage':'SCAFFOLD','all_checks_pass':True,'scope':'Temporary synthetic 480-payload export, Node component simulation, tiny opaque streaming ZIP and Word XML; not real model/label evaluation or browser QA',
        'real_all60_labels_read':False,'real_website_exported':False,'checks':checks,
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'code').glob('*.py'))+[ROOT/'code/templates/website/app.js']}}
(ROOT/'qa/scaffold_v3_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','checks':len(checks),'scope':result['scope']}))
