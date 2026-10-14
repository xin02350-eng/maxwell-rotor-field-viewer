"""Finite nonlinear axis-saturation warps of the native current plane.
Uses immutable train-only 24-mode signed cycle POD; no test/full60 access.
Mathematical axis rotation is not a claim that native Thet is a dq angle.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def warp(c,spec,current_scale):
    c=np.atleast_2d(c);a=np.deg2rad(c[:,1]-spec['rotation_deg']);xy=np.column_stack((c[:,0]*np.cos(a),c[:,0]*np.sin(a)))/current_scale
    for j,cap in enumerate(spec['axis_caps']):
        if cap is not None:xy[:,j]=cap*np.tanh(xy[:,j]/cap)
    return xy
def poly(x):return np.column_stack((np.ones(len(x)),x[:,1],x[:,0],x[:,1]**2,x[:,0]*x[:,1],x[:,0]**2))
def kernel(x,y,length):return 1/np.sqrt(1+np.sum((x[:,None,:]-y[None,:,:])**2,axis=-1)/length**2)
def metrics(p,t):
    e=p-t;return dict(rmse_T=float(np.sqrt(np.mean(e*e))),relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(t)),mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()))
def predict(model,conditions,frame_index=None,return_signed=False):
    p=Path(model)
    if sha(p)!=p.with_suffix('.npz.sha256').read_text().strip():raise ValueError('Model SHA mismatch')
    with np.load(p,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    spec=json.loads(str(m['metadata_json']));c=np.asarray(conditions,dtype=float);single=c.ndim==1;c=np.atleast_2d(c)
    if c.shape[1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Expected native RMS/Thet')
    x=(warp(c,spec,float(m['current_scale']))-m['center'])/m['scale'];x=x*np.array([np.sqrt(2.8),1/np.sqrt(2.8)])
    coefficient=kernel(x,m['centers'],spec['lengthscale'])@m['weights']+poly(x)@m['trend']
    signed=(m['mean']+coefficient@m['basis'].T).reshape(len(c),60,3600)
    result=signed if return_signed else np.hypot(signed[...,0::2],signed[...,1::2])
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<60:raise ValueError('Stored frame index0..59 only')
        result=result[:,frame_index]
    return result[0] if single else result
def run(out):
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    configs=[dict(rotation_deg=angle,axis_caps=list(caps),lengthscale=length,ridge=ridge,degree=2,anisotropy=2.8,rank=24)
        for angle in [0,30,60,90] for caps in [(.35,None),(.7,None),(1.4,None),(.7,1.4),(1.4,.7),(.7,.7)]
        for length in [1.,2.,4.] for ridge in [1e-4,1e-2]]
    dump(out/'protocol.json',dict(configs=configs,selection='Complete validation9 original-T planar magnitude RMSE; max also reported',
        all_basis_and_scales='Training42 only',test_labels_read=False,all60_fitting_labels_read=False,new_NN_trainings=0,source_sha256=sha(__file__)))
    old=ROOT.parent/'new60_opt_v2/kernel_route/rotor_cycle_refined/model.npz'
    with np.load(old,allow_pickle=False) as z:mean=z['b0_mean'];basis=z['b0_basis'];oldmeta=json.loads(str(z['metadata_json']))
    prepared=ROOT.parent/'new60_opt_v2/kernel_route/prepared_rotor/cycles.npz'
    with np.load(prepared,allow_pickle=False) as z:
        c=z['train_conditions'];cv=z['validation_conditions'];ids=z['train_case_ids'];coeff=z['train_coeff'][:,:24]
        if not np.array_equal(z['basis'][:,:24],basis) or not np.array_equal(z['mean'],mean):raise ValueError('Original train-only POD differs')
        true=z['validation'].reshape(9,60,3600);truth=np.hypot(true[...,0::2],true[...,1::2])
    current_scale=float(oldmeta['blocks'][0]['current_scale']);best=float('inf');best_arrays=None;best_config=None;rows=[]
    for i,spec in enumerate(configs):
        raw=warp(c,spec,current_scale);center=raw.mean(0);scale=np.maximum(raw.std(0),1e-12);x=(raw-center)/scale*np.array([np.sqrt(2.8),1/np.sqrt(2.8)]);P=poly(x)
        try:
            A=np.block([[kernel(x,x,spec['lengthscale'])+spec['ridge']*np.eye(42),P],[P.T,np.zeros((6,6))]])
            solution=np.linalg.solve(A,np.concatenate((coeff,np.zeros((6,24)))))
            xv=(warp(cv,spec,current_scale)-center)/scale*np.array([np.sqrt(2.8),1/np.sqrt(2.8)])
            valcoef=kernel(xv,x,spec['lengthscale'])@solution[:42]+poly(xv)@solution[42:]
            signed=(mean+valcoef@basis.T).reshape(9,60,3600);value=np.hypot(signed[...,0::2],signed[...,1::2])
            if not np.isfinite(value).all():raise ValueError('Nonfinite field')
            metric=metrics(value,truth);row=dict(config=spec,validation=metric,status='PASS')
            if metric['rmse_T']<best:
                best=metric['rmse_T'];best_config=spec;best_arrays=dict(mean=mean,basis=basis,center=center,scale=scale,centers=x,weights=solution[:42],trend=solution[42:],current_scale=np.array(current_scale));best_value=value
        except (ValueError,np.linalg.LinAlgError) as exc:row=dict(config=spec,status='FAILED',reason=str(exc))
        rows.append(row)
        if (i+1)%36==0:print(f'Axis-warp {i+1}/{len(configs)} best val RMSE{best:.8g}T',flush=True)
    best_config.update(format='v4-axis-warp-signed-cycle-pod',source_sha256=sha(__file__),training_original_ids=ids.tolist(),
        test_labels_read=False,all60_fitting_labels_read=False,prepared_sha256=sha(prepared),base_model_sha256=sha(old),
        method='Analytic axis-saturation kernel with fixed training-only signed POD; not a neural network')
    best_arrays['metadata_json']=np.array(json.dumps(best_config));np.savez_compressed(out/'selected_model.npz',**best_arrays);(out/'selected_model.npz.sha256').write_text(sha(out/'selected_model.npz')+'\n')
    np.savez_compressed(out/'validation_predictions.npz',prediction=best_value)
    replay=predict(out/'selected_model.npz',cv);delta=float(np.abs(replay-best_value).max())
    report=dict(config_count=len(configs),rows=rows,best=min((r for r in rows if r['status']=='PASS'),key=lambda r:r['validation']['rmse_T']),source_sha256=sha(__file__),model_sha256=sha(out/'selected_model.npz'),
        replay_max_abs_T=delta,elapsed_seconds=time.perf_counter()-start,test_labels_read=False,all60_fitting_labels_read=False,new_NN_trainings=0)
    dump(out/'results.json',report);print(json.dumps({k:report[k] for k in ['config_count','best','replay_max_abs_T','elapsed_seconds']}),flush=True)
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(limits=2):run(a.output)
