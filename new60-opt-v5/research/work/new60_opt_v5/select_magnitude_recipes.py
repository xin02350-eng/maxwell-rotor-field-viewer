"""Finite V5 whole-field recipes, declared before candidate outcomes.

Primary and pure-neural choices use full validation9 only. A later final51
refit keeps the selected objective/structure/weights, with no validation left.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    p=Path(p)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
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
    for index,r in enumerate(meta['components']):
        p=(mp.parent/r['model_relative_path']).resolve();s=(mp.parent/r['module_relative_path']).resolve()
        if sha(p)!=r['model_sha256'] or sha(s)!=r['module_sha256']:raise ValueError('Component SHA mismatch')
        v=load(s,'v5_norm_component_'+str(index)).predict(p,conditions,frame_index=frame_index,**r.get('kwargs',{}));parts.append((r,v))
    if meta['blend_route']=='signed_vector':
        if any(r['quantity']!='Br,Bt' for r,v in parts):raise ValueError('Unsigned component in vector recipe')
        v=sum(r['weight']*v for r,v in parts);return np.hypot(v[...,0::2],v[...,1::2])
    return sum(r['weight']*(np.hypot(v[...,0::2],v[...,1::2]) if r['quantity']=='Br,Bt' else v) for r,v in parts)
def metrics(v,y):
    e=v-y;sse=float(np.sum(e*e));return dict(relative_L2_pct=float(100*np.sqrt(sse/np.sum(y*y))),rmse_T=float(np.sqrt(sse/e.size)),mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()))
def protocol():
    out=ROOT/'magnitude_recipes';out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(source_sha256=sha(__file__),new_recipe_count=60,existing_control_count=2,
        alpha_grid=[0,.25,.5,.75,1],recipe_family='4magnitude-NNs vsGP42 in signed/scalar routes(40), vsoldpureNN42 scalar(20); endpoints duplicated; retain V4mixed/pure2controls',
        selection='Fullvalidation9 RMSEsubject tomaxAbs<=.75T, retain boundedV4 baseline; pureNN same amongneuralonly recipes. No pointweights or cropping.',
        final51='Ifprimary containsone newNN, train exactlyone51NN withsame selectedstructure/signedlambda from51warmstart. Bindsame ratio with51GP; scalaroldpure42 component becomesfixed.25NN2+.75NN3 51pure. Ifnewroutefails primaryvalidation, retainpredeclaredV4 51fixedrecipe.',
        test_labels_read=False,full60fit_used=False))
def main():
    out=ROOT/'magnitude_recipes';plan=read(out/'protocol.json')
    if sha(__file__)!=plan['source_sha256']:raise ValueError('Declared source changed')
    manifest=read(ROOT/'magnitude_objective_nn/final_manifest.json');nets=manifest['models']
    if len(nets)!=4:raise ValueError('Exactly4 declared fine-tuning models required')
    def binding(name,mp,sp,q,kw,family):return dict(name=name,model_relative_path=os.path.relpath(mp,out).replace('\\','/'),module_relative_path=os.path.relpath(sp,out).replace('\\','/'),quantity=q,kwargs=kw,family=family,model_sha256=sha(mp),module_sha256=sha(sp))
    v4=WORK/'new60_opt_v4';v2=WORK/'new60_opt_v2'
    gp=binding('gp42',v2/'kernel_route/rotor_cycle_refined/model.npz',v2/'kernel_route/kernel_model.py','Br,Bt',{},'kernel')
    pure=binding('V4pure42',v4/'radial_combo/pure_nn_model.npz',v4/'radial_combo.py','|B|',{},'neural')
    old=binding('V4mixed42',v4/'radial_combo/primary_model.npz',v4/'radial_combo.py','|B|',{},'mixed')
    nn=[binding(row['id'],ROOT/'magnitude_objective_nn'/row['model_relative_path'],ROOT/'magnitude_objective_nn'/row['inference_module_relative_path'],'Br,Bt',row.get('signed_output_kwargs',{'return_signed':True}),'neural') for row in nets]
    with np.load(v4/'evaluation/prediction_rotor_magnitude/validation_predictions.npz',allow_pickle=False) as z:y=z['truth'];c=z['conditions'];ids=z['case_ids']
    if ids.tolist()!=[1,9,14,15,24,27,35,40,51]:raise ValueError('Validation IDs changed')
    values={r['name']:load(out/r['module_relative_path'],'v5_selection_'+r['name']).predict(out/r['model_relative_path'],c,**r['kwargs']) for r in [gp,pure,old,*nn]}
    def norm(r):
        v=values[r['name']];return np.hypot(v[...,0::2],v[...,1::2]) if r['quantity']=='Br,Bt' else v
    rows=[]
    def add(parts,route,family):
        parts=[r for r in parts if r['weight']>0]
        if route=='signed_vector':v=sum(r['weight']*values[r['name']] for r in parts);v=np.hypot(v[...,0::2],v[...,1::2])
        else:v=sum(r['weight']*norm(r) for r in parts)
        rows.append(dict(index=len(rows),components=parts,blend_route=route,family=family,validation=metrics(v,y)))
    for r in nn:
        for a in plan['alpha_grid']:
            for route in ['signed_vector','scalar_magnitude']:add([dict(gp,weight=1-a),dict(r,weight=a)],route,'neural' if a==1 else 'kernel' if a==0 else 'mixed')
        for a in plan['alpha_grid']:add([dict(pure,weight=1-a),dict(r,weight=a)],'scalar_magnitude','neural')
    if len(rows)!=60:raise AssertionError('Recipe count changed')
    add([dict(old,weight=1)],'scalar_magnitude','mixed');add([dict(pure,weight=1)],'scalar_magnitude','neural')
    bounded=[r for r in rows if r['validation']['max_abs_T']<=.75];pures=[r for r in rows if r['family']=='neural'];pb=[r for r in pures if r['validation']['max_abs_T']<=.75]
    selected=dict(primary=min(bounded or rows,key=lambda r:r['validation']['rmse_T']),pure_nn=min(pb or pures,key=lambda r:r['validation']['rmse_T']))
    choices={}
    for label,row in selected.items():
        mp=out/(label+'_model.npz');np.savez_compressed(mp,metadata_json=np.array(json.dumps(dict(format='v5_originalT_wholefield_magnitude_recipe',**row,source_sha256=sha(__file__),test_labels_used=False))))
        mp.with_suffix('.npz.sha256').write_text(sha(mp)+'\n',encoding='ascii')
        if abs(metrics(predict(mp,c),y)['rmse_T']-row['validation']['rmse_T'])>1e-12:raise AssertionError('Model replay changed')
        choices[label]=dict(model_relative_path=mp.relative_to(ROOT).as_posix(),inference_module_relative_path='select_magnitude_recipes.py',family=row['family'],validation=row['validation'],components=row['components'],blend_route=row['blend_route'])
    write(out/'results.json',dict(new_recipe_count=60,existing_controls=2,rows=rows,selected=selected))
    write(ROOT/'selection_validation_only.json',dict(version='V5validation-only-selection',**choices,test_labels_used=False,full60_fit_used=False,
        selected_new_primary_neural_ids=[r['name'] for r in selected['primary']['components'] if r['name'] not in ['gp42','V4pure42','V4mixed42']]))
    print(json.dumps(choices,ensure_ascii=False),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--protocol-only',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):protocol() if a.protocol_only else main()
