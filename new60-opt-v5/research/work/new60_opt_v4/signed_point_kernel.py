"""Train-LOO output-specific signed response kernels; no test/full60 access.

Independent point kernels escape the whole-cycle spatial affine-span limit.
Input hyperparameters are selected per point using training-only LOO errors;
validation chooses a finite common shrinkage recipe, never a point routing map.
The unchanged gradient-trained signed-cycle NN supplies zero-mean time variation.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def coordinates(c,kind,scale):
    c=np.atleast_2d(np.asarray(c,dtype=float));r=c[:,0]/scale;a=np.deg2rad(c[:,1])
    if kind.startswith('saturated_'):
        factor=float(kind.split('_')[1]);r=factor*np.tanh(r/factor)
    return np.column_stack((r*np.cos(a),r*np.sin(a)))
def kernel(x,y,kind,length):
    r=np.linalg.norm(x[:,None]-y[None],axis=-1)/length
    if kind=='gaussian':return np.exp(-.5*r*r)
    if kind=='matern32':z=np.sqrt(3)*r;return (1+z)*np.exp(-z)
    if kind=='inverse_multiquadric':return 1/np.sqrt(1+r*r)
    raise ValueError(kind)
def polynomial(x,d):return np.column_stack([x[:,0]**i*x[:,1]**(t-i) for t in range(d+1) for i in range(t+1)])
def base_features(c,kind,scale):
    c=np.atleast_2d(np.asarray(c,dtype=float));r=c[:,0]/scale;a=np.deg2rad(c[:,1])
    if kind=='native':return np.column_stack((r,c[:,1]/90))
    x=r*np.cos(a);y=r*np.sin(a)
    if kind=='cartesian':return np.column_stack((x,y))
    if kind=='mixed':return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))
    return np.column_stack((x,y,x*x,x*y,y*y))
def baseline_predict(m,c):
    c=np.atleast_2d(c);s=float(m['current_scale']);f=base_features(c,str(m['feature_kind']),s)
    v=(f-m['feature_mean'])/m['feature_scale']
    for j in range(int(m['layer_count'])):
        v=v@m[f'w{j}']+m[f'b{j}']
        if j<int(m['layer_count'])-1:v=np.tanh(v) if str(m['activation'])=='tanh' else v/(1+np.exp(-np.clip(v,-700,700)))
    x=base_features(c,'cartesian',s);pol=polynomial(x,int(m['base_degree']))
    latent=v*m['output_scale']+pol@m['base_weights']
    coefficient=(m['cycle_mean']+latent@m['cycle_basis']).reshape(len(c),int(m['frame_count']),-1)
    return m['spatial_mean']+coefficient@m['spatial_basis'].T
def load_model(path):
    if isinstance(path,dict):return path
    p=Path(path)
    if sha(p)!=p.with_suffix('.npz.sha256').read_text().strip():raise ValueError('Model SHA mismatch')
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def predict(model,conditions,frame_index=None,return_signed=False):
    m=load_model(model);meta=json.loads(str(m['metadata_json']));c=np.asarray(conditions,dtype=float);single=c.ndim==1;c=np.atleast_2d(c)
    if c.shape[1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Expected native RMS/Thet')
    mean=np.empty((len(c),int(m['point_count']),2))
    for i,spec in enumerate(meta['groups']):
        pre=f'g{i}_';z=(coordinates(c,spec['coordinate'],meta['current_scale'])-m[pre+'center'])/m[pre+'scale']
        z=z*np.array([np.sqrt(spec['anisotropy']),1/np.sqrt(spec['anisotropy'])])
        value=kernel(z,m[pre+'centers'],spec['kernel'],spec['lengthscale'])@m[pre+'weights']+polynomial(z,spec['degree'])@m[pre+'trend']
        mean[:,m[pre+'points'],:]=value.reshape(len(c),-1,2)
    base={k[len('baseline_'):]:v for k,v in m.items() if k.startswith('baseline_')}
    vector=baseline_predict(base,c).reshape(len(c),60,int(m['point_count']),2)
    vector=mean[:,None,:,:]+float(m['dynamic_scale'])*(vector-vector.mean(1,keepdims=True))
    result=vector.reshape(len(c),60,-1) if return_signed else np.hypot(vector[...,0],vector[...,1])
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<60:raise ValueError('Fixed frame index0..59 only')
        result=result[:,frame_index]
    return result[0] if single else result
def metrics(value,truth):
    e=value-truth
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(truth)),mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()))
def run(out):
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    prepared=ROOT.parent/'new60_opt_v3/joint_neural/point_prepared'
    configs=[dict(coordinate=coordinate,kernel=kind,lengthscale=length,ridge=ridge,degree=degree,anisotropy=2.8)
        for coordinate in ['polar_xy','saturated_0.5','saturated_1'] for kind in ['gaussian','matern32','inverse_multiquadric']
        for length in [1.,2.,4.] for ridge in [1e-5,1e-3] for degree in [1,2]]
    # 108 settings and one exact V2 shared-kernel setting; finite before labels.
    configs.append(dict(coordinate='saturated_1',kernel='inverse_multiquadric',lengthscale=3.72,ridge=.0001,degree=2,anisotropy=2.8))
    alphas=[0.,.25,.5,.75,1.]
    dump(out/'protocol.json',dict(configs=configs,global_shrinkage_alphas=alphas,source_sha256=sha(__file__),
        labels='train42 signed means only; validation9 fullmag selects common recipe',perpoint_selector='Training LOO component MSE sum over Br/Bt',
        test_labels_read=False,all60_fit_labels_read=False,wholefield_points_retained=True,new_gradient_NN_trainings=0))
    with np.load(prepared/'mean_data.npz',allow_pickle=False) as z:
        ids=z['case_ids'];c=z['conditions'];means=z['signed_mean'][:42];dynamic=z['validation_signed_dynamic'];dynamic_scale=float(z['signed_dynamic_scale'])
    if ids[:42].tolist()!=json.loads((ROOT.parent/'new60_20261014/cache_v1/split.json').read_text())['original_ids']['train']:raise ValueError('Training identity mismatch')
    with np.load(ROOT.parent/'new60_opt_v3/evaluation/prediction_rotor_magnitude/validation_predictions.npz',allow_pickle=False) as z:
        if z['case_ids'].tolist()!=ids[42:].tolist():raise ValueError('Validation identity mismatch')
        truth=z['truth']
    with np.load(prepared/'baseline_model.npz',allow_pickle=False) as z:base={k:z[k] for k in z.files}
    with np.load(ROOT.parent/'new60_opt_v2/kernel_route/rotor_cycle_refined/model.npz',allow_pickle=False) as z:old_gp=json.loads(str(z['metadata_json']))
    current_scale=float(old_gp['blocks'][0]['current_scale']);target=means.reshape(42,-1);solutions=[];errors=[];rows=[];best=float('inf');best_candidate=None
    def score(mean):v=mean.reshape(9,1800,2)[:,None]+dynamic_scale*dynamic;return np.hypot(v[...,0],v[...,1])
    for number,spec in enumerate(configs):
        raw=coordinates(c[:42],spec['coordinate'],current_scale);center=raw.mean(0);scale=np.maximum(raw.std(0),1e-12)
        x=(raw-center)/scale*np.array([np.sqrt(spec['anisotropy']),1/np.sqrt(spec['anisotropy'])]);pol=polynomial(x,spec['degree'])
        A=np.block([[kernel(x,x,spec['kernel'],spec['lengthscale'])+spec['ridge']*np.eye(42),pol],[pol.T,np.zeros((pol.shape[1],pol.shape[1]))]])
        try:
            inv=np.linalg.inv(A);sol=inv@np.concatenate((target,np.zeros((pol.shape[1],target.shape[1]))))
            loo=(sol[:42]/np.diag(inv)[:42,None]).reshape(42,1800,2)
            mse=np.mean(np.sum(loo*loo,axis=-1),axis=0)
            if not np.isfinite(mse).all():raise ValueError('Nonfinite LOO')
            xv=(coordinates(c[42:],spec['coordinate'],current_scale)-center)/scale*np.array([np.sqrt(spec['anisotropy']),1/np.sqrt(spec['anisotropy'])])
            val=kernel(xv,x,spec['kernel'],spec['lengthscale'])@sol[:42]+polynomial(xv,spec['degree'])@sol[42:]
            value=score(val);metric=metrics(value,truth)
            solutions.append(dict(spec=spec,center=center,scale=scale,centers=x,weights=sol[:42],trend=sol[42:],validation_mean=val))
            errors.append(mse);row=dict(candidate=number,config=spec,train_loo_mse_T2=float(mse.mean()),validation=metric,status='PASS')
            if metric['rmse_T']<best:best=metric['rmse_T'];best_candidate=number
        except (np.linalg.LinAlgError,ValueError) as exc:
            solutions.append(None);errors.append(np.full(1800,np.inf));row=dict(candidate=number,config=spec,status='FAILED',reason=str(exc))
        rows.append(row)
        if (number+1)%36==0:print(f'Completed{number+1}/{len(configs)} shared controls; best val RMSE{best:.7g}T',flush=True)
    errors=np.array(errors);choice=errors.argmin(0);shared=int(np.mean(errors,axis=1).argmin());selection_rows=[]
    pointmean=np.stack([solutions[int(choice[p])]['validation_mean'].reshape(9,1800,2)[:,p,:] for p in range(1800)],axis=1)
    def save(choice_indices,name):
        arrays=dict(point_count=np.array(1800),dynamic_scale=np.array(dynamic_scale))
        groups=[]
        for i,idx in enumerate(sorted(set(choice_indices.tolist()))):
            sol=solutions[idx];points=np.flatnonzero(choice_indices==idx);cols=np.column_stack((2*points,2*points+1)).reshape(-1)
            pre=f'g{i}_';arrays.update({pre+'points':points,pre+'center':sol['center'],pre+'scale':sol['scale'],pre+'centers':sol['centers'],pre+'weights':sol['weights'][:,cols],pre+'trend':sol['trend'][:,cols]});groups.append(sol['spec'])
        arrays.update({'baseline_'+k:v for k,v in base.items()})
        meta=dict(format='v4-output-specific-signed-kernel-response',groups=groups,current_scale=current_scale,training_original_ids=ids[:42].tolist(),validation_original_ids=ids[42:].tolist(),test_labels_read=False,all60_fit_labels_read=False,
            source_sha256=sha(__file__),prepared_sha256=sha(prepared/'mean_data.npz'),baseline_model_sha256=sha(prepared/'baseline_model.npz'),method='analytic point-response kernels plus actual frozen signed neural time variation, not new neural training')
        arrays['metadata_json']=np.array(json.dumps(meta));np.savez_compressed(out/name,**arrays);(out/name).with_suffix('.npz.sha256').write_text(sha(out/name)+'\n')
    # Save both reconstructable primitive routes; shrinkage metadata stays explicit.
    save(choice,'point_model.npz');save(np.full(1800,shared),'shared_model.npz');save(np.full(1800,best_candidate),'validation_best_shared_model.npz')
    for alpha in alphas:
        mean=(1-alpha)*solutions[shared]['validation_mean'].reshape(9,1800,2)+alpha*pointmean
        selection_rows.append(dict(alpha=alpha,validation=metrics(score(mean.reshape(9,-1)),truth)))
    np.savez_compressed(out/'training_loo_selectors.npz',train_loo_point_mse_T2=errors,point_config_index=choice,shared_config_index=np.array(shared),case_ids=ids[:42])
    report=dict(status='PASS',config_count=len(configs),rows=rows,shared_train_loo_candidate=shared,best_validation_shared_candidate=best_candidate,
        best_validation_shared=rows[best_candidate],perpoint_shrinkage=selection_rows,point_model_validation=selection_rows[-1]['validation'],
        point_model_replay=metrics(predict(out/'point_model.npz',c[42:]),truth),elapsed_seconds=time.perf_counter()-start,test_labels_read=False,
        all60_fit_labels_read=False,new_gradient_NN_trainings=0)
    dump(out/'results.json',report);print(json.dumps({k:report[k] for k in ['config_count','best_validation_shared','perpoint_shrinkage','elapsed_seconds']}),flush=True)
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(limits=2):run(a.output)
