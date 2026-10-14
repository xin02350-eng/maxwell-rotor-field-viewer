"""Temporary synthetic V5 scope/reuse/CLI/UI/serialization checks only."""
import ast,base64,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import export_website_v5 as exporter
import build_delivery_v5 as builder
import check_web_payloads_v5 as checker
from scope_utils_v5 import normalize_scope,case_role,select_choice

checks=[]
def check(name,ok):
    checks.append({'name':name,'pass':bool(ok)})
    if not ok:raise AssertionError(name)
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
def freeze(root,registry):
    dump(root/'both_versions_choices_frozen.json',registry);h=exporter.sha(root/'both_versions_choices_frozen.json')
    (root/'both_versions_choices_frozen.sha256').write_text(h,encoding='utf-8');return h

for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
check('all V5 Python code parses',True)
node=shutil.which('node') or r'C:\Program Files\nodejs\node.exe'
subprocess.run([node,'--check',str(ROOT/'code/templates/website/app.js')],check=True)
check('JS syntax',True)
snapshot=json.loads((ROOT/'qa/v4_scaffold_source_snapshot.json').read_text(encoding='utf-8'))
check('copied V4 inputs remain byte-identical',all(exporter.sha(ROOT.parent/'new60_opt_v4'/r['path'])==r['sha256'] for r in snapshot['files']))

with tempfile.TemporaryDirectory(prefix='maxwell_v5_scope_components_') as tmp:
    work=Path(tmp);root=work/'v5';old=work/'old';root.mkdir();old.mkdir()
    val=[1,9,14,15,24,27,35,40,51];test=[2,4,12,16,17,18,38,41,58]
    train=[i for i in range(60) if i not in val+test];split={'train':train,'validation':val,'test':test}
    dump(root/'split.json',split);dump(root/'protocol.json',{'synthetic_component_only':True})
    module='import numpy as np\ndef predict(path,conditions,frame_index=None):\n c=np.asarray(conditions);single=c.ndim==1;c=np.atleast_2d(c)\n with np.load(path,allow_pickle=False) as z:n=int(z["points"])\n p=np.broadcast_to((2+c[:,0,None,None]*.001), (len(c),60,n)).copy()\n if frame_index is not None:p=p[:,frame_index]\n return p[0] if single else p\n'
    for base in [root,old]:(base/'infer.py').write_text(module,encoding='utf-8')
    specs=[('prediction_rotor_magnitude','rotor','original42'),('prediction_rotor_pure_nn','rotor','original42'),
       ('prediction_rotor_final51_magnitude','rotor','final51'),('prediction_rotor_final51_pure_nn','rotor','final51'),
       ('prediction_stator_magnitude','stator','original42'),('prediction_airgap_Br','airgap','original42'),
       ('fitting_rotor_magnitude','rotor','full60'),('fitting_stator_magnitude','stator','full60'),('fitting_airgap_Br','airgap','full60')]
    models=[];oldmodels=[]
    for i,(mid,region,scope) in enumerate(specs):
        source=root if i<4 else old;oldid=mid if source==root else 'legacy_'+mid
        mp=source/'models'/(oldid+'.npz');mp.parent.mkdir(exist_ok=True);np.savez(mp,points=np.array(3))
        choice=dict(id=mid,branch='fitting_full60' if scope=='full60' else 'prediction',region=region,quantity='Br' if region=='airgap' else 'planar_magnitude',
           role='control' if i in [1,2,3] else 'primary',family='synthetic_fixture',method='synthetic_only',display_label='合成组件 '+mid,
           training_scope=scope,train_case_ids=train if scope=='original42' else train+val if scope=='final51' else list(range(60)),
           validation_case_ids=val if scope=='original42' else [],reused_test_case_ids=test if scope!='full60' else [],
           scientific_partitions=['training_fit'] if scope=='full60' else ['reused_test'] if scope=='final51' else ['validation','reused_test'],
           model_relative_path=mp.relative_to(root).as_posix() if source==root else '../old/'+mp.relative_to(old).as_posix(),model_sha256=exporter.sha(mp),
           inference_module_relative_path='infer.py' if source==root else '../old/infer.py',inference_module_sha256=exporter.sha(source/'infer.py'))
        if source==old:
            oldmodel=dict(choice,id=oldid,model_relative_path=mp.relative_to(old).as_posix(),inference_module_relative_path='infer.py')
            oldmodels.append(oldmodel);choice.update(scientific_source_relative_root='old',scientific_model_id=oldid)
        models.append(choice)
    oldsha=freeze(old,{'models':oldmodels})
    for m in models:
        if 'scientific_source_relative_root' in m:m['expected_scientific_choices_sha256']=oldsha
    # One cache deliberately includes an unreadable object label entry. The CLI
    # must access inputs/geometry only, never allow_pickle or numeric field data.
    theta=9+1.5*np.arange(60);points=np.array([101,102,103]);xy=np.array([[1.,0.],[2.,.5],[1.5,1.]])
    conditions=np.column_stack([np.arange(60)*3.,np.arange(60)*.5])
    cache=root/'inputs.npz';np.savez(cache,conditions=conditions,case_ids=np.arange(60),point_id=points,xy=xy,theta=theta,time_s=np.arange(1,61)/22000.,solver_index=np.arange(1,61),theta_m_relative_deg=theta-7.5,field=np.array([object()],dtype=object))
    for m in models:m.update(cache_relative_path='inputs.npz',cache_sha256=exporter.sha(cache))
    registry={'models':models,'split_relative_path':'split.json','split_sha256':exporter.sha(root/'split.json'),'protocol_sha256':exporter.sha(root/'protocol.json')}
    ch=freeze(root,registry);proofs=[]
    for i,m in enumerate(models):
        source=root if i<4 else old;sid=m.get('scientific_model_id',m['id']);sh=ch if source==root else oldsha
        scope=normalize_scope(m,split)
        for part in m['scientific_partitions']:
            ids=list(range(60)) if part=='training_fit' else val if part=='validation' else test
            idx=np.arange(len(ids)*60*3).reshape(len(ids),60,3)
            truth=(.2 if m['region']=='airgap' else 2.)+np.sin(idx*.11)*.1+np.array(ids)[:,None,None]*.001
            pred=truth+np.cos(idx*.23)*(1e-12 if part=='training_fit' else .007)
            path=source/'evaluation'/sid/(part+'_predictions.npz');path.parent.mkdir(parents=True,exist_ok=True)
            meta={'choices_sha256':sh,'model_sha256':m['model_sha256'],'inference_module_sha256':m['inference_module_sha256'],'source_sha256':'synthetic-source-sha'}
            np.savez_compressed(path,truth=truth,prediction=pred,case_ids=np.array(ids),conditions=conditions[ids],point_id=points,xy=xy,theta=theta,time_s=np.arange(1,61)/22000.,quantity=m['quantity'],metadata_json=json.dumps(meta))
            proofs.append(dict(id=m['id'],partition=part,pass_=True,scientific_source_relative_root=source.name,scientific_model_id=sid,
                expected_scientific_choices_sha256=sh,scientific_npz_sha256=exporter.sha(path),source_sha256='synthetic-source-sha',model_sha256=m['model_sha256'],scientific_reuse_explicitly_verified=source!=root))
    for p in proofs:p['pass']=p.pop('pass_')
    dump(root/'qa.json',{'all_checks_pass':True,'choices_sha256':ch,'results':proofs})
    final51=models[2];scope=normalize_scope(final51,split)
    check('original validation is train in final51',case_role(scope,1)=='train' and case_role(scope,2)=='reused_test')
    check('model ID overrides implicit defaults',select_choice(models,models[4]['id'])['region']=='stator')
    try:select_choice(models,models[4]['id'],region='rotor')
    except ValueError:check('explicit conflicting model selectors rejected',True)
    else:check('explicit conflicting model selectors rejected',False)
    result=exporter.build(root,[root/'qa.json'])
    check('scope-aware270 synthetic packs',result['payload_count']==270)
    check('no copied old scientific arrays',not (root/'evaluation'/'legacy_fitting_rotor_magnitude').exists())
    (root/'qa').mkdir();audit=checker.run(root);check('all270 independent serialization checks',audit['all_checks_pass'] and audit['payloads_checked']==270)
    # A stale original PASS cannot authorize reuse in a new V5 registry.
    bad=json.loads(json.dumps(proofs));next(r for r in bad if r['scientific_source_relative_root']=='old')['scientific_reuse_explicitly_verified']=False
    badproof=next(r for r in bad if r['scientific_source_relative_root']=='old');choice=next(m for m in models if m['id']==badproof['id'])
    path=old/'evaluation'/choice['scientific_model_id']/(badproof['partition']+'_predictions.npz')
    with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z['metadata_json']))
    try:exporter.scientific_binding(root,choice,badproof['partition'],path,meta,[{'report':{'results':bad}}],ch)
    except ValueError:check('old science without new explicit reuse proof rejected',True)
    else:check('old science without new explicit reuse proof rejected',False)
    badproof.update(scientific_arrays_reused=True,scientific_arrays_saved=True)
    accepted=exporter.scientific_binding(root,choice,badproof['partition'],path,meta,[{'report':{'results':bad}}],ch)
    check('formal reuse and saved flags accepted with complete SHA proof',accepted['reused'] is True)
    badproof['scientific_arrays_saved']=False
    try:exporter.scientific_binding(root,choice,badproof['partition'],path,meta,[{'report':{'results':bad}}],ch)
    except ValueError:check('reuse flag without saved scientific artifact rejected',True)
    else:check('reuse flag without saved scientific artifact rejected',False)
    js=subprocess.run([node,str(ROOT/'qa/test_viewer_components_v5.js'),str(root/'website')],capture_output=True,text=True)
    print(js.stdout)
    if js.returncode:raise AssertionError(js.stderr)
    check('Node async interaction and final51 label simulation',True)
    output=root/'case1.csv'
    cli=[sys.executable,str(ROOT/'code/predict_v5.py'),'--study',str(root),'--model-id',final51['id'],'--case-id','1','--frame','0','--output',str(output)]
    run=subprocess.run(cli,capture_output=True,text=True)
    if run.returncode:raise AssertionError(run.stderr)
    meta=json.loads(output.with_suffix('.csv.json').read_text(encoding='utf-8'))
    check('CLI original-val case1 marked train underfinal51',meta['effective_role']=='train' and meta['old_prediction_split_role']=='validation' and meta['training_scope']['key']=='final51')
    check('CLI does not access poison field labels',meta['field_labels_read'] is False and len(output.read_text(encoding='utf-8').splitlines())==4)
    check('CLI refuses replacement',subprocess.run(cli,capture_output=True).returncode!=0)
    markdown=builder.report(root,{'choices_sha256':ch,'website_deployment':None},{})
    word=root/'report.docx';builder.docx(word,markdown)
    with zipfile.ZipFile(word) as z:
        for name in z.namelist():ET.fromstring(z.read(name))
        tree=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        rows=tree.findall('.//w:tbl/w:tr',ns);check('scope report10rows and6columns',len(rows)==10 and all(len(r.findall('w:tc',ns))==6 for r in rows))
    check('no fictional V5 counts', '13次完整神经网络训练' not in markdown and '尚无最终实验计数' in markdown)
    check('no real website or data generated',not (ROOT/'website').exists())

out={'status':'PASS','validation_stage':'SCAFFOLD','all_checks_pass':True,'scope':'Temporary synthetic scope/reuse export and270packs, CLI with poison label field, Node component simulation, and Word XML; no real models/data/test evaluation or browser QA',
     'real_scientific_arrays_read':False,'checks':checks,'checks_count':len(checks),
     'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'code').glob('*.py'))+[ROOT/'code/templates/website/app.js']}}
(ROOT/'qa/scaffold_v5_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','stage':'SCAFFOLD','checks':len(checks)}))
