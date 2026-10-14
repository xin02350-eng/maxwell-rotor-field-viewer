"""Explicitly authorized, label-free original versus hardlink-relocated inference.

Without --run this only reports prerequisites. --run does not score models or
select them. It compares all nine frozen models at original/native new inputs,
frames0/59, runs the real CLI, and enforces file/NPZ/import guards in subprocesses.
The temporary hardlinks retain work-relative sibling paths and are never edited.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, os, pathlib, runpy, subprocess, sys, uuid
from datetime import datetime,timezone
Path=pathlib.Path
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False)

def worker(config_path):
    cfg=read(config_path);root=Path(cfg['study']).resolve();work=root.parent;out=Path(cfg['output']).resolve();out.mkdir(parents=True,exist_ok=False)
    allowed={str((work/p).resolve()).casefold() for p in cfg['allowed']};runtime=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
    sys.dont_write_bytecode=True;sys.path.insert(0,str(root/'code'))
    opened=[];blocked=[];npz_access=[];skipped_bytecode=[]
    import importlib.machinery
    source_get_data=importlib.machinery.SourceFileLoader.get_data
    def audited_source(loader,path):
        if str(path).endswith('.pyc') and Path(path).resolve().is_relative_to(work):
            skipped_bytecode.append(str(path));raise FileNotFoundError('Force hashed source instead of optional bytecode')
        return source_get_data(loader,path)
    importlib.machinery.SourceFileLoader.get_data=audited_source
    class RejectFramework:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in {'torch','sklearn','scipy'}:
                blocked.append({'type':'import','name':fullname});raise ImportError('Portable inference forbids '+fullname)
    sys.meta_path.insert(0,RejectFramework())
    def audit(event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if p.is_relative_to(out):return
        if writing:blocked.append({'type':'write','path':str(p)});raise PermissionError('Only new QA outputs may be written')
        if any(p.is_relative_to(q) for q in runtime):return
        if str(p).casefold() not in allowed:
            blocked.append({'type':'read','path':str(p)});raise PermissionError('Outside retained project dependency whitelist: '+str(p))
        opened.append(str(p.relative_to(work)))
    sys.addaudithook(audit)
    import numpy as np
    from threadpoolctl import threadpool_limits
    from numpy.lib.npyio import NpzFile
    from inference_numpy_v5 import read_freeze,predict
    input_keys={'conditions','case_ids','point_id','xy','theta','time_s','solver_index','theta_m_relative_deg'}
    registry=read_freeze(root);choices=registry['models'];cache_paths={(root/m.get('cache_relative_path',f'../new60_20261014/cache_v1/{m["region"]}.npz')).resolve() for m in choices}
    original_get=NpzFile.__getitem__
    def safe_get(z,key):
        p=Path(z.zip.filename).resolve()
        if p.is_relative_to(out):return original_get(z,key)
        if (p in cache_paths and key not in input_keys) or key in {'field','coeff','truth','target','targets','prediction'}:
            blocked.append({'type':'npz_label','path':str(p),'key':key});raise PermissionError('Numeric label entry is forbidden')
        npz_access.append({'path':p.relative_to(work).as_posix(),'key':key});return original_get(z,key)
    NpzFile.__getitem__=safe_get
    checks=[];arrays={};descriptions=[]
    def check(name,ok):
        if not bool(ok):raise AssertionError(name)
        checks.append(name)
    def cli(args,success=True):
        before=list(sys.argv);sys.argv=[str(root/'code/predict_v5.py'),*list(map(str,args))]
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):runpy.run_path(sys.argv[0],run_name='__main__')
        except (ValueError,FileExistsError,SystemExit) as e:
            if success:raise
            return type(e).__name__
        finally:sys.argv[:]=before
        if not success:raise AssertionError('Invalid CLI request unexpectedly succeeded')
    with threadpool_limits(limits=2):
        for index,m in enumerate(choices):
            cache=(root/m.get('cache_relative_path',f'../new60_20261014/cache_v1/{m["region"]}.npz')).resolve()
            with np.load(cache,allow_pickle=False) as z:
                cond=z['conditions'];ids=z['case_ids'];xy=z['xy'];theta=z['theta'];points=z['point_id']
            c=cond[np.flatnonzero(ids==2)[0]];new=np.array([317.25,47.625])
            check(m['id']+'/unseen input is not an original condition',not np.any(np.all(np.isclose(cond,new,rtol=0,atol=1e-10),axis=1)))
            for kind,condition in [('case02',c),('unseen',new)]:
                for frame in [0,59]:
                    key=f'm{index}_{kind}_f{frame}';value=predict(root,m,condition,frame_index=frame)
                    check(key+'/finite expected points',value.shape==(len(points),) and np.isfinite(value).all());arrays[key]=value
                    descriptions.append({'key':key,'model_id':m['id'],'input_kind':kind,'condition':condition.tolist(),'frame':frame,'model_sha256':m['model_sha256']})
            dest=out/(m['id']+'.npz');cli(['--model-id',m['id'],'--case-id','2','--frame','59','--coordinates','world','--output',dest])
            with np.load(dest,allow_pickle=False) as z:
                actual=z['prediction'];coords=z['xy'];meta=json.loads(str(z['metadata_json']))
            check(m['id']+'/real CLI equals source prediction',np.array_equal(actual[0],arrays[f'm{index}_case02_f59']))
            angle=np.deg2rad(theta[59]);expected=xy@np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]) if m['region']=='rotor' else xy
            check(m['id']+'/world coordinates',np.max(np.abs(coords[0]-expected))<1e-11)
            check(m['id']+'/scope and quantity preserved',meta['model_id']==m['id'] and meta['quantity']==m['quantity'] and meta['field_labels_read'] is False and meta['effective_role']==('training_fit' if m['branch']=='fitting_full60' else 'reused_test'))
            arrays[f'm{index}_world_xy']=coords
        selected=next(m for m in choices if m['id']=='prediction_rotor_final51_magnitude')
        d=out/'final51_case01.csv';cli(['--model-id',selected['id'],'--case-id','1','--frame','0','--output',d])
        check('final51 original validation case01 is training',read(d.with_suffix('.csv.json'))['effective_role']=='train')
        check('existing output rejected',cli(['--model-id',selected['id'],'--case-id','1','--frame','0','--output',d],False)=='FileExistsError')
        for name,args in [('invalid frame',['--frame','60']),('negative current',['--current-rms','-1','--thet-deg','20']),('nonfinite current',['--current-rms','nan','--thet-deg','20']),('conflicting model region',['--model-id',selected['id'],'--region','stator'])]:
            cli([*args,'--output',out/(name.replace(' ','_')+'.csv')],False);check(name+' rejected',True)
        # Explicit guard probes are separated from actual inference accesses.
        inference_block_count=len(blocked)
        if inference_block_count:write(out/'unexpected_accesses.json',{'blocked':blocked})
        check('actual inference needed no forbidden import/file/label access: '+json.dumps(blocked),inference_block_count==0)
        for name in ['torch','sklearn','scipy']:
            try:__import__(name)
            except ImportError:check('guard rejects '+name,True)
            else:check('guard rejects '+name,False)
        try:
            with np.load(next(iter(cache_paths)),allow_pickle=False) as z:z['field']
        except PermissionError:check('guard rejects cache field labels',True)
        else:check('guard rejects cache field labels',False)
        try:(work/'outside_manifest_probe.txt').read_bytes()
        except PermissionError:check('guard rejects undeclared physical source',True)
        else:check('guard rejects undeclared physical source',False)
    np.savez_compressed(out/'predictions.npz',**arrays)
    report=dict(all_checks_pass=True,choices_sha256=sha(root/'both_versions_choices_frozen.json'),checks=checks,case_frame_outputs=descriptions,
                project_reads=sorted(set(opened)),npz_access=npz_access,guard_probes=blocked,inference_block_count=inference_block_count,optional_bytecode_not_read=sorted(set(skipped_bytecode)),
                runtime_allowlist=[str(x) for x in runtime],runtime_note='Only Python/NumPy/threadpoolctl installation files may be read outside retained project paths; no original project fallback.',
                labels_read=False,framework_imported=False,model_selection_performed=False)
    write(out/'worker_report.json',report)

def run(root=ROOT):
    root=Path(root).resolve();sys.path.insert(0,str(root/'code'))
    from prepare_delivery_dependencies_v5 import collect_inference
    inv,registry,ch=collect_inference(root,include_all_source=False)
    report_path=root/'qa/portable_v5_check.json'
    if report_path.exists():raise FileExistsError(report_path)
    stage=root/'qa/portable_delivery_check'/('run_'+uuid.uuid4().hex[:10]);work=stage/'work';work.mkdir(parents=True,exist_ok=False)
    # Only inference closure and its exact source files; no labels are decoded.
    for rel,row in inv.rows.items():
        source=root.parent/rel;target=work/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
        if not os.path.samefile(source,target) or target.stat().st_size!=row['bytes']:raise AssertionError('Hardlink staging failed')
    allowed=sorted(inv.rows);runs={}
    for label,study in [('original',root),('relocated',work/root.name)]:
        cfg=stage/(label+'_config.json');out=stage/(label+'_outputs');write(cfg,{'study':str(study),'output':str(out),'allowed':allowed})
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(cfg)]
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        p=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=env,cwd=stage)
        (stage/(label+'_stdout.log')).write_text(p.stdout,encoding='utf-8');(stage/(label+'_stderr.log')).write_text(p.stderr,encoding='utf-8')
        if p.returncode:raise RuntimeError(label+' guarded worker failed: '+p.stderr[-4000:])
        runs[label]=read(out/'worker_report.json')
    import numpy as np
    parity=[]
    with np.load(stage/'original_outputs/predictions.npz',allow_pickle=False) as a,np.load(stage/'relocated_outputs/predictions.npz',allow_pickle=False) as b:
        if a.files!=b.files:raise AssertionError('Parity output keys differ')
        for key in a.files:
            delta=float(np.max(np.abs(a[key]-b[key])))
            if not np.allclose(a[key],b[key],rtol=1e-12,atol=1e-11):raise AssertionError('Relocation changed '+key)
            parity.append({'key':key,'shape':list(a[key].shape),'max_abs_difference':delta,'pass':True})
    unchanged=[]
    for rel,row in inv.rows.items():
        ok=sha(root.parent/rel)==row['sha256']==sha(work/rel)
        if not ok:raise AssertionError('Original/hardlink source changed: '+rel)
        unchanged.append({'path':rel,'sha256':row['sha256'],'same_ntfs_file':os.path.samefile(root.parent/rel,work/rel)})
    report=dict(all_checks_pass=True,choices_sha256=ch,created_utc=datetime.now(timezone.utc).isoformat(),models=len(registry['models']),
                hardlinked_files=len(inv.rows),hardlinked_logical_bytes=sum(x['bytes'] for x in inv.rows.values()),source_sha256=sha(__file__),
                dependency_source_sha256=sha(root/'code/prepare_delivery_dependencies_v5.py'),stage_relative_path=stage.relative_to(root).as_posix(),
                original_and_relocated=runs,parity=parity,source_immutability=unchanged,
                maximum_prediction_delta_T=max(x['max_abs_difference'] for x in parity if not x['key'].endswith('world_xy')),
                labels_read=False,model_selection_performed=False,fresh_test_evaluation=False,
                limitation='Hardlink relocation proves relative dependency resolution under a guarded different path on this machine. ZIP member SHA/CRC/bytes must be checked independently after archive creation; this is not a second-machine installation test.')
    write(report_path,report)
    (root/'qa/portable_v5_check.md').write_text('# V5 搬迁推理检查\n\nPASS：全部冻结模型在原路径与新硬链接目录使用原工况和新输入、帧0/59预测一致。真实CLI、坐标、训练范围、错误输入和防覆盖检查通过。逐文件SHA未变，禁止读取标签和未声明项目路径，禁止Torch/sklearn/SciPy；系统Python与NumPy/threadpoolctl安装文件为必要运行时白名单。\n\n此检查不重新选择模型或评价预测误差，也不替代ZIP逐成员SHA/CRC/字节核验或另一台机器的安装测试。\n',encoding='utf-8')
    return {'all_checks_pass':True,'models':len(registry['models']),'hardlinked_files':len(inv.rows),'parity_checks':len(parity),'maximum_prediction_delta_T':report['maximum_prediction_delta_T']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--run',action='store_true');p.add_argument('--worker',type=Path);a=p.parse_args()
    if a.worker:worker(a.worker);return
    if a.run:result=run(a.root)
    else:result={'ready':(a.root/'both_versions_choices_frozen.json').is_file(),'numeric_inference_run':False,'instruction':'Run --run only after final freeze and explicit parity-QA authorization; no science labels are needed.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
