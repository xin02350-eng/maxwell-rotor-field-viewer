"""Two low-capacity physical response NNs, finite whole-field controls only."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s)
    sys.path.insert(0,str(Path(p).parent))
    try:s.loader.exec_module(m)
    finally:sys.path.pop(0)
    return m
def predict(model,conditions,frame_index=None):
    mp=Path(model)
    with np.load(mp,allow_pickle=False) as z:meta=json.loads(str(z['metadata_json']))
    parts=[]
    for index,row in enumerate(meta['components']):
        p=(mp.parent/row['model_relative_path']).resolve();s=(mp.parent/row['module_relative_path']).resolve()
        if sha(p)!=row['model_sha256'] or sha(s)!=row['module_sha256']:raise ValueError('SHA mismatch')
        v=load(s,'v4_radial_component_'+str(index)).predict(p,conditions,frame_index=frame_index,**row.get('kwargs',{}));parts.append((row,v))
    if meta['blend_route']=='signed_vector':
        if any(row['quantity']!='Br,Bt' for row,v in parts):raise ValueError('Unsigned branch in signed blend')
        v=sum(row['weight']*v for row,v in parts);return np.hypot(v[...,0::2],v[...,1::2])
    return sum(row['weight']*(np.hypot(v[...,0::2],v[...,1::2]) if row['quantity']=='Br,Bt' else v) for row,v in parts)
def metric(v,y):
    e=v-y;sse=float(np.sum(e*e));return dict(relative_L2_pct=float(100*np.sqrt(sse/np.sum(y*y))),rmse_T=float(np.sqrt(sse/e.size)),
        max_abs_T=float(np.abs(e).max()),mae_T=float(np.abs(e).mean()),worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()))
def protocol():
    out=ROOT/'radial_combo';out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(candidate_count=30,alphas=[0,.25,.5,.75,1],radial_candidate_count=2,
        controls='Each radial versusGP42 in signed/scalar routes(20), versus previously selectedpureNN42 norm in scalar route(10). Endpoints may be numerically duplicated.',
        selection='Include preexistingmixed42 andpure42. Primary minimum fullvalidation9 RMSEwithmaxAbs<=.75T; pureNN same amongneuralrecipes. If no boundedcandidate useminimumRMSE.',
        no_per_point_weights=True,no_test_labels_read=True,no_full60_fitting_used=True,source_sha256=sha(__file__)))
def main():
    out=ROOT/'radial_combo';plan=read(out/'protocol.json')
    if sha(__file__)!=plan['source_sha256']:raise ValueError('Protocol/source changed')
    rm=read(ROOT/'radial_saturation_nn/final_manifest.json');radial=rm['models']
    if len(radial)!=2:raise ValueError('Exactly2 predeclared radial networks required')
    v2=ROOT.parent/'new60_opt_v2';base=ROOT/'global_field_blend'
    def binding(name,mp,sp,q,kw,family):return dict(name=name,model_relative_path=os.path.relpath(mp,out).replace('\\','/'),module_relative_path=os.path.relpath(sp,out).replace('\\','/'),quantity=q,kwargs=kw,family=family,model_sha256=sha(mp),module_sha256=sha(sp))
    gp=binding('gp42',v2/'kernel_route/rotor_cycle_refined/model.npz',v2/'kernel_route/kernel_model.py','Br,Bt',{},'kernel')
    pure=binding('oldpure42',base/'peak_best_model.npz',ROOT/'global_field_blend.py','|B|',{},'neural')
    mixed=binding('oldmixed42',base/'peak_bounded_model.npz',ROOT/'global_field_blend.py','|B|',{},'mixed')
    entries=[]
    for row in radial:
        entries.append(binding(row['id'],ROOT/'radial_saturation_nn'/row['model_relative_path'],ROOT/'radial_saturation_nn'/row['inference_module_relative_path'],'Br,Bt',row.get('signed_output_kwargs',{'return_signed':True}),'neural'))
    with np.load(ROOT.parent/'new60_opt_v3/evaluation/prediction_rotor_magnitude/validation_predictions.npz',allow_pickle=False) as z:y=z['truth'];c=z['conditions'];ids=z['case_ids']
    if ids.tolist()!=[1,9,14,15,24,27,35,40,51]:raise ValueError('Validation IDs changed')
    branches=[gp,pure,mixed,*entries];values={r['name']:load(out/r['module_relative_path'],'v4_radial_develop_'+r['name']).predict(out/r['model_relative_path'],c,**r['kwargs']) for r in branches}
    def norm(row):
        v=values[row['name']];return np.hypot(v[...,0::2],v[...,1::2]) if row['quantity']=='Br,Bt' else v
    rows=[]
    def add(components,route,family):
        components=[r for r in components if r['weight']>0]
        if route=='signed_vector':v=sum(r['weight']*values[r['name']] for r in components);v=np.hypot(v[...,0::2],v[...,1::2])
        else:v=sum(r['weight']*norm(r) for r in components)
        rows.append(dict(index=len(rows),components=components,blend_route=route,family=family,validation=metric(v,y)))
    for r in entries:
        for alpha in plan['alphas']:
            for route in ['signed_vector','scalar_magnitude']:add([dict(gp,weight=1-alpha),dict(r,weight=alpha)],route,'neural' if alpha==1 else ('kernel' if alpha==0 else 'mixed'))
        for alpha in plan['alphas']:add([dict(pure,weight=1-alpha),dict(r,weight=alpha)],'scalar_magnitude','neural')
    if len(rows)!=30:raise ValueError('Finite count mismatch')
    add([dict(mixed,weight=1.)],'scalar_magnitude','mixed');add([dict(pure,weight=1.)],'scalar_magnitude','neural')
    bounded=[r for r in rows if r['validation']['max_abs_T']<=.75];nn=[r for r in rows if r['family']=='neural'];nnb=[r for r in nn if r['validation']['max_abs_T']<=.75]
    selected={'primary':min(bounded or rows,key=lambda r:r['validation']['rmse_T']),'pure_nn':min(nnb or nn,key=lambda r:r['validation']['rmse_T'])}
    choices={}
    for label,row in selected.items():
        mp=out/(label+'_model.npz');np.savez_compressed(mp,metadata_json=np.array(json.dumps(dict(format='v4_radial_finite_blend',**row,source_sha256=sha(__file__),test_labels_used=False))))
        mp.with_suffix('.npz.sha256').write_text(sha(mp)+'\n',encoding='ascii')
        if abs(metric(predict(mp,c),y)['rmse_T']-row['validation']['rmse_T'])>1e-12:raise AssertionError('Replay mismatch')
        note='；'.join(f"{r['weight']:.0%}×{r['name']}" for r in row['components'])+'；'+row['blend_route']+'；完整验证9选择全局比例'
        choices[label]=dict(model_relative_path=str(mp.relative_to(ROOT)).replace('\\','/'),inference_module_relative_path='radial_combo.py',validation=row['validation'],family=row['family'],note=note)
    write(out/'results.json',dict(new_combination_count=30,existing_controls=2,rows=rows,selected=selected))
    sources=['radial_combo.py']+list(dict.fromkeys('radial_saturation_nn/'+r['inference_module_relative_path'] for r in radial))
    write(ROOT/'final_selection_validation_only.json',dict(version='v4_final_validation_only_selection',primary=choices['primary'],pure_nn=choices['pure_nn'],
        final51_recipe_remains_original_predeclared=True,test_labels_used=False,full60_fitting_used=False,
        support_files=[dict(relative_path=p,sha256=sha(ROOT/p)) for p in sources]))
    print(json.dumps(choices,ensure_ascii=False),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--protocol-only',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):protocol() if a.protocol_only else main()
