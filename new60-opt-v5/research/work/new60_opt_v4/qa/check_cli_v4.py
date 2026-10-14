"""Real CLI and source replay with label-cache entries and ML frameworks blocked."""
from pathlib import Path
import builtins,csv,importlib.util,json,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from inference_numpy_v4 import read_freeze,predict,digest
import predict_v4

def main():
    registry=read_freeze(ROOT);checks=[];out=ROOT/'qa/cli_validation_final';out.mkdir(exist_ok=True)
    def check(name,ok,detail=None):
        checks.append(dict(name=name,pass_=bool(ok),details=detail))
        if not ok:raise AssertionError(name+': '+str(detail))
    old_import=builtins.__import__;old_load=np.load
    forbidden_reads=[];frameworks=[]
    class InputsOnly:
        def __init__(self,z):self.z=z
        def __enter__(self):return self
        def __exit__(self,*args):self.z.close()
        def __getitem__(self,key):
            if key not in predict_v4.INPUT_KEYS:
                forbidden_reads.append(key);raise AssertionError('Input cache numeric label access '+key)
            return self.z[key]
    def safe_load(path,*args,**kw):
        z=old_load(path,*args,**kw)
        return InputsOnly(z) if isinstance(path,(str,Path)) and 'cache_v1' in str(path) and str(path).endswith('.npz') else z
    def safe_import(name,*args,**kw):
        if name.split('.')[0] in ('torch','tensorflow','jax','sklearn','scipy'):
            frameworks.append(name);raise AssertionError('Inference requested training framework '+name)
        return old_import(name,*args,**kw)
    np.load=safe_load;builtins.__import__=safe_import
    try:
        c=np.array([[121.,41.],[360.,54.4],[9.817283,49.85252032]])
        for i,row in enumerate(registry['models']):
            source=(ROOT/row['inference_module_relative_path']).resolve()
            spec=importlib.util.spec_from_file_location('_v4_source_parity_'+str(i),source);m=importlib.util.module_from_spec(spec)
            sys.path.insert(0,str(source.parent))
            try:
                spec.loader.exec_module(m)
                expected=m.predict(ROOT/row['model_relative_path'],c,**row.get('predict_kwargs',{}))
            finally:sys.path.pop(0)
            actual=predict(ROOT,row,c)
            delta=float(np.max(np.abs(actual-expected)));check(row['id']+' source forward parity',np.allclose(actual,expected,rtol=2e-12,atol=2e-12),delta)
            for frame in (0,44,59):
                a=predict(ROOT,row,c,frame);delta=float(np.abs(a-expected[:,frame]).max())
                check(row['id']+f' frame {frame}',np.allclose(a,expected[:,frame],rtol=1e-7,atol=1e-7),delta)
        scenarios=[('fitting_full60','rotor',44,'npz',None),('fitting_full60','stator',0,'csv',None),('fitting_full60','airgap',59,'npz',None),
            ('prediction','rotor',0,'npz',None),('prediction','rotor',59,'csv','prediction_rotor_pure_nn'),('prediction','stator',44,'npz',None),('prediction','airgap',59,'csv',None)]
        for branch,region,frame,extension,model_id in scenarios:
            dest=out/f'{branch}_{region}_{frame}.{extension}'
            sys.argv=['predict_v4.py','--branch',branch,'--region',region,'--case-id','2','--frame',str(frame),'--output',str(dest)]
            if model_id:sys.argv+=['--model-id',model_id]
            predict_v4.main();meta=json.loads(dest.with_suffix(dest.suffix+'.json').read_text(encoding='utf8'))
            check(dest.name+' labels blocked',meta['field_labels_read'] is False)
            check(dest.name+' honest score scope',meta['effective_role']==('training_fit' if branch=='fitting_full60' else 'reused_test'))
            row=next(r for r in registry['models'] if r['id']==meta['model_id']);expect=predict(ROOT,row,[360.,54.4],frame)
            if extension=='npz':
                with np.load(dest,allow_pickle=False) as z:
                    check(dest.name+' data replay',np.allclose(z['prediction'][0],expect,rtol=0,atol=1e-7))
                    if region=='rotor':
                        ref=z['xy_reference'];angle=np.deg2rad(float(z['theta'][0]));xy=np.column_stack((ref[:,0]*np.cos(angle)-ref[:,1]*np.sin(angle),ref[:,0]*np.sin(angle)+ref[:,1]*np.cos(angle)))
                        check(dest.name+' absolute world rotation',np.allclose(z['xy'][0],xy,rtol=0,atol=2e-12))
            else:
                with dest.open(encoding='utf8',newline='') as f:table=list(csv.reader(f))
                check(dest.name+' point rows',len(table)==len(expect)+1)
                check(dest.name+' field replay',np.allclose([float(r[-1]) for r in table[1:]],expect,rtol=0,atol=1e-7))
        unseen=out/'unseen_input.npz';sys.argv=['predict_v4.py','--branch','prediction','--region','rotor','--current-rms','120.1234567','--thet-deg','41.1234567','--frame','0','--output',str(unseen)];predict_v4.main()
        meta=json.loads(unseen.with_suffix('.npz.json').read_text(encoding='utf8'))
        check('new input scope',meta['original_case_id'] is None and meta['score_scope']=='new input; accuracy not established')
        for model_id in ('prediction_rotor_final51_magnitude','prediction_rotor_final51_pure_nn'):
            dest=out/(model_id+'_former_validation.npz')
            sys.argv=['predict_v4.py','--model-id',model_id,'--case-id','1','--frame','0','--output',str(dest)]
            predict_v4.main();meta=json.loads(dest.with_suffix('.npz.json').read_text('utf8'))
            check(model_id+' former validation now training',meta['old_prediction_split_role']=='validation' and meta['effective_role']=='train' and len(meta['training_scope']['train_case_ids'])==51)
        invalid=[['--frame','60'],['--current-rms','-1','--thet-deg','41'],['--current-rms','nan','--thet-deg','41'],['--case-id','99'],['--current-rms','1'],['--case-id','2','--current-rms','1','--thet-deg','41']]
        for i,arguments in enumerate(invalid):
            dest=out/f'invalid_{i}.npz';sys.argv=['predict_v4.py','--branch','prediction','--region','rotor','--output',str(dest),*arguments]
            try:predict_v4.main()
            except ValueError:rejected=True
            else:rejected=False
            check(f'invalid request {i}',rejected and not dest.exists())
        before=digest(unseen);sys.argv=['predict_v4.py','--branch','prediction','--region','rotor','--output',str(unseen)]
        try:predict_v4.main()
        except FileExistsError:rejected=True
        else:rejected=False
        check('existing export preserved',rejected and digest(unseen)==before)
        check('no cache label attempts',not forbidden_reads)
        check('no training framework attempts',not frameworks)
    finally:builtins.__import__=old_import;np.load=old_load
    result=dict(all_checks_pass=True,choices_sha256=digest(ROOT/'both_versions_choices_frozen.json'),check_count=len(checks),checks=checks,
        field_labels_read=False,ML_frameworks_imported=False,minimal_declared_dependencies=['numpy','threadpoolctl'],
        results=[dict(id=r['id'],pass_=True) for r in registry['models']])
    # JSON uses the shared QA vocabulary, despite pass being a Python keyword.
    text=json.dumps(result,ensure_ascii=False,indent=2).replace('"pass_":','"pass":')+'\n'
    (ROOT/'qa/cli_v4_check.json').write_text(text,encoding='utf8')
    print(json.dumps(dict(all_checks_pass=True,check_count=len(checks)),ensure_ascii=False))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
