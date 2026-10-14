"""Finite whole-field blends; validation only, with a predeclared peak constraint.

No target lookup at deployment. Bound components are actual trained networks
or the independently identified kernel control. Weights are common to every
condition, frame and spatial point, never a validation-derived point mask.
"""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(Path(path).parent))
    try:spec.loader.exec_module(m)
    finally:sys.path.pop(0)
    return m
def predict(model,conditions,frame_index=None):
    p=Path(model)
    with np.load(p,allow_pickle=False) as z:meta=json.loads(str(z['metadata_json']))
    signed=[]
    for i,row in enumerate(meta['components']):
        mp=(p.parent/row['model_relative_path']).resolve();sp=(p.parent/row['module_relative_path']).resolve()
        if digest(mp)!=row['model_sha256'] or digest(sp)!=row['module_sha256']:raise ValueError('Component SHA mismatch')
        v=module(sp,'v4_field_component_'+str(i)).predict(mp,conditions,frame_index=frame_index,**row.get('kwargs',{}))
        signed.append(np.asarray(v,dtype=np.float64))
    if meta['blend_route']=='signed_vector':
        v=sum(r['weight']*a for r,a in zip(meta['components'],signed))
        return np.hypot(v[...,0::2],v[...,1::2])
    return sum(r['weight']*np.hypot(a[...,0::2],a[...,1::2]) for r,a in zip(meta['components'],signed))
def metrics(value,truth):
    e=value-truth;sse=float(np.sum(e*e));te=float(np.sum(truth*truth))
    loc=tuple(int(x) for x in np.unravel_index(np.argmax(np.abs(e)),e.shape))
    return dict(relative_L2_pct=100*np.sqrt(sse/te),rmse_T=np.sqrt(sse/e.size),
        max_abs_T=float(np.abs(e).max()),mae_T=float(np.abs(e).mean()),
        worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()),max_abs_array_location=loc)
def main():
    out=HERE/'global_field_blend';out.mkdir(exist_ok=False)
    v2=HERE.parent/'new60_opt_v2';nn=HERE/'point_regime_nn'
    defs=[('gp',v2/'kernel_route/rotor_cycle_refined/model.npz',v2/'kernel_route/kernel_model.py',{}),
        ('old_nn',v2/'neural_route/rotor/selected_model.npz',v2/'neural_route/cycle_neural.py',{}),
        ('nn2',nn/'runs/regime_sigmoid_affine2_principal_retry1_seed42/model.npz',nn/'regime_point_nn.py',{'return_signed':True}),
        ('nn3',nn/'runs/regime_sigmoid_affine3_principal_seed42/model.npz',nn/'regime_point_nn.py',{'return_signed':True}),
        ('quad2',nn/'runs/regime_quadratic_sigmoid_affine2_principal_seed42/model.npz',nn/'regime_quadratic_skip_nn.py',{'return_signed':True}),
        ('quad3',nn/'runs/regime_quadratic_sigmoid_affine3_principal_seed42/model.npz',nn/'regime_quadratic_skip_nn.py',{'return_signed':True})]
    bindings={name:dict(name=name,model_relative_path=os.path.relpath(mp,out).replace('\\','/'),module_relative_path=os.path.relpath(sp,out).replace('\\','/'),model_sha256=digest(mp),module_sha256=digest(sp),kwargs=kw) for name,mp,sp,kw in defs}
    recipes=[];seen=set()
    def add(weights,route):
        parts=tuple(sorted((k,v) for k,v in weights.items() if v>0));key=(route,parts)
        if key in seen:return
        seen.add(key);recipes.append(dict(weights=dict(parts),blend_route=route))
    names=list(bindings)
    for name in names:add({name:1.},'signed_vector')
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for alpha in [.25,.5,.75]:
                for route in ['signed_vector','scalar_magnitude']:add({a:1-alpha,b:alpha},route)
    # The same fixed family can be refitted on51 without reading test labels.
    for i in range(5):
        for j in range(5-i):
            for route in ['signed_vector','scalar_magnitude']:add({'gp':i/4,'nn2':j/4,'nn3':(4-i-j)/4},route)
    write(out/'protocol.json',dict(source_sha256=digest(__file__),components=bindings,recipes=recipes,
        candidate_count=len(recipes),selection='Full validation9 RMSE; also retain lowest RMSE with maxAbs<=0.75T, and minimum maxAbs. No point masks, no cropping.',
        final51_recipe='Among fixed gp/nn2/nn3 recipes only, choose validation42-model RMSE with peak<=0.75T; if unavailable minimize RMSE. Use identical weights/route with independently trained51 models.',
        test_labels_opened=False,full60_fitting_used=False))
    truthpath=HERE.parent/'new60_opt_v3/evaluation/prediction_rotor_magnitude/validation_predictions.npz'
    with np.load(truthpath,allow_pickle=False) as z:truth=z['truth'];c=z['conditions'];ids=z['case_ids']
    if ids.tolist()!=[1,9,14,15,24,27,35,40,51]:raise ValueError('Validation IDs mismatch')
    signed={name:module(sp,'v4_blend_develop_'+name).predict(mp,c,**kw) for name,mp,sp,kw in defs}
    norms={name:np.hypot(a[...,0::2],a[...,1::2]) for name,a in signed.items()}
    rows=[]
    for index,recipe in enumerate(recipes):
        if recipe['blend_route']=='signed_vector':
            v=sum(w*signed[name] for name,w in recipe['weights'].items());value=np.hypot(v[...,0::2],v[...,1::2])
        else:value=sum(w*norms[name] for name,w in recipe['weights'].items())
        rows.append(dict(index=index,**recipe,validation=metrics(value,truth)))
    rmse=lambda r:r['validation']['rmse_T']
    bounded=[r for r in rows if r['validation']['max_abs_T']<=.75]
    final51=[r for r in rows if set(r['weights']).issubset({'gp','nn2','nn3'})]
    final51bounded=[r for r in final51 if r['validation']['max_abs_T']<=.75]
    chosen={'average_best':min(rows,key=rmse),'peak_bounded':min(bounded,key=rmse) if bounded else min(rows,key=rmse),
        'peak_best':min(rows,key=lambda r:(r['validation']['max_abs_T'],rmse(r))),
        'final51_fixed_recipe':min(final51bounded or final51,key=rmse)}
    for label,row in chosen.items():
        meta=dict(format='v4_whole_field_finite_blend',blend_route=row['blend_route'],
            components=[dict(bindings[name],weight=w) for name,w in row['weights'].items()],
            validation_selection_metrics=row['validation'],source_sha256=digest(__file__),
            test_labels_used=False,training_case_count=42,selection_label=label)
        mp=out/(label+'_model.npz');np.savez_compressed(mp,metadata_json=np.array(json.dumps(meta)))
        value=predict(mp,c)
        if abs(metrics(value,truth)['rmse_T']-row['validation']['rmse_T'])>1e-12:raise AssertionError('Portable replay mismatch')
        mp.with_suffix('.npz.sha256').write_text(digest(mp)+'\n',encoding='ascii')
    write(out/'results.json',dict(candidate_count=len(rows),rows=rows,chosen=chosen,test_labels_opened=False))
    print(json.dumps(dict(candidate_count=len(rows),chosen=chosen),ensure_ascii=False),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
