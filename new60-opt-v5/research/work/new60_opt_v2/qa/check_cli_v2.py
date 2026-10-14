"""Forward-equation parity, actual CLI outputs and world rotation checks."""
import importlib.util,json,sys,subprocess,tempfile,builtins
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
from inference_numpy_v2 import digest,load_model,predict
from predict_v2 import verified_choices

def main():
    choices=verified_choices(ROOT);results=[]
    original_import=builtins.__import__
    def safeimport(name,*args,**kwargs):
        if name.split('.')[0] in ('torch','tensorflow','jax','sklearn'):raise AssertionError('Inference imported training framework '+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=safeimport
    try:
      for i,row in enumerate(choices['models']):
        path=ROOT/row['inference_module_relative_path'];sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location(f'_original_predict_{i}',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=load_model(ROOT/row['model_relative_path'],row['model_sha256']);c=np.array([[121.,41.],[360.,54.4],[9.817283,49.85252032]])
        expected=module.predict(ROOT/row['model_relative_path'],c,**row.get('predict_kwargs',{}));actual=predict(model,c,row)
        delta=float(np.max(np.abs(expected-actual)));np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
        for frame in (0,44,59):np.testing.assert_allclose(predict(model,c,row,frame),expected[:,frame],rtol=2e-12,atol=2e-12)
        results.append(dict(id=row['id'],passed=True,max_forward_difference_T=delta,source_sha256=digest(path),model_sha256=row['model_sha256']))
    finally:builtins.__import__=original_import
    directory=ROOT/'qa/cli_validation';directory.mkdir(exist_ok=True)
    commands=[('fitting_full60','rotor',44,'npz'),('prediction','stator',0,'csv'),('prediction','airgap',59,'npz')]
    cli=[]
    for branch,region,frame,ext in commands:
        output=directory/f'{branch}_{region}_{frame}.{ext}'
        if output.exists():raise FileExistsError(output)
        cmd=[sys.executable,str(ROOT/'code/predict_v2.py'),'--branch',branch,'--region',region,'--case-id','2','--frame',str(frame),'--output',str(output)]
        subprocess.run(cmd,check=True,capture_output=True,text=True,encoding='utf8')
        info=json.loads(output.with_suffix(output.suffix+'.json').read_text(encoding='utf8'));assert info['field_labels_read'] is False
        if ext=='npz':
          with np.load(output,allow_pickle=False) as z:
            if region=='airgap':assert z['prediction'].shape==(1,900)
            else:
              ref=z['xy_reference'];angle=np.deg2rad(z['theta'][0]);expected=np.column_stack((ref[:,0]*np.cos(angle)-ref[:,1]*np.sin(angle),ref[:,0]*np.sin(angle)+ref[:,1]*np.cos(angle)));np.testing.assert_allclose(z['xy'][0],expected,rtol=0,atol=2e-12)
        cli.append(dict(branch=branch,region=region,frame=frame,bytes=output.stat().st_size,sha256=digest(output),passed=True,scope=info['score_scope']))
    report=dict(passed=True,model_count=len(results),models=results,actual_cli_exports=cli,max_forward_difference_T=max(r['max_forward_difference_T'] for r in results),numpy_only_forward_equations=True,field_labels_loaded=False,code_sha256={n:digest(ROOT/'code'/n) for n in ('inference_numpy_v2.py','predict_v2.py')})
    (ROOT/'qa/cli_v2_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (ROOT/'qa/cli_v2_check.md').write_text(f"# V2两版推理检查\n\n全部{len(results)}个模型前向方程与原始冻结入口一致，最大差异{report['max_forward_difference_T']:.3g} T。实际CSV/NPZ、转子世界坐标与帧0/44/59接口通过。没有读取field/coeff磁场标签；前向实现仅需NumPy，不导入PyTorch或sklearn。详见JSON。\n",encoding='utf8')
    print(json.dumps(dict(passed=True,max_delta=report['max_forward_difference_T'],model_count=len(results)),ensure_ascii=False))
if __name__=='__main__':main()
