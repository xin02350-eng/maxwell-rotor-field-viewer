"""Low-capacity train-only signed Br/Bt mean NN with radial tanh saturation.

The affine effective drive is a learned proxy, not a measured material H field.
Isotropic (7 parameters/point) and determinant-one elliptic (9 parameters/point)
forms couple the two signed outputs through one radial nonlinear activation.
Immutable train42 old neural dynamics supplies all60 saved frame variations.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np

VERSION='v4-signed-mean-radial-tanh-affine-drive-neural'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def features(c,kind,scale):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64))
    if c.shape[1]!=2 or not np.isfinite(c).all():raise ValueError('Expected native [I_RMS_A, Thet_deg]')
    r=c[:,0]/scale;a=np.deg2rad(c[:,1]);x=r*np.cos(a);y=r*np.sin(a)
    if kind=='cartesian':return np.column_stack((x,y))
    if kind=='mixed':return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))
    if kind=='cartesian2':return np.column_stack((x,y,x*x,x*y,y*y))
    raise ValueError('Unknown feature specification')

def sigmoid(v):
    z=np.exp(-np.abs(v));return np.where(v>=0,1/(1+z),z/(1+z))

def softplus(v):return np.where(v>20,v,np.logaddexp(0,v))

def baseline_predict(m,c):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));scale=float(m['current_scale'])
    v=(features(c,str(m['feature_kind']),scale)-m['feature_mean'])/m['feature_scale']
    for j in range(int(m['layer_count'])):
        v=v@m[f'w{j}']+m[f'b{j}']
        if j<int(m['layer_count'])-1:v=np.tanh(v) if str(m['activation'])=='tanh' else v*sigmoid(v)
    x=features(c,'cartesian',scale);degree=int(m['base_degree'])
    poly=np.column_stack([x[:,0]**i*x[:,1]**j for total in range(degree+1) for i in range(total+1) for j in [total-i]])
    latent=v*m['output_scale']+poly@m['base_weights']
    coef=(m['cycle_mean']+latent@m['cycle_basis']).reshape(len(c),int(m['frame_count']),-1)
    return m['spatial_mean']+coef@m['spatial_basis'].T

def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if sha(path)!=path.with_suffix(path.suffix+'.sha256').read_text().strip():raise ValueError('Radial neural model SHA mismatch')
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def predict_mean(m,c):
    x=(features(c,'cartesian',float(m['current_scale']))-m['feature_mean'])/m['feature_scale']
    h=np.einsum('cf,pfh->pch',x,m['drive_weight'])+m['drive_bias'][:,None,:]
    if str(m['radial_kind'])=='elliptic':
        s=float(m['max_log_stretch'])*np.tanh(m['raw_stretch']);a=float(m['max_shear'])*np.tanh(m['raw_shear'])
        q0=np.exp(s)[:,None]*h[...,0]+a[:,None]*h[...,1];q1=np.exp(-s)[:,None]*h[...,1]
        rho=np.sqrt(q0*q0+q1*q1+1e-20)
    else:rho=np.sqrt(np.sum(h*h,axis=-1)+1e-20)
    bs=softplus(m['raw_saturation'])+float(m['minimum_saturation'])
    z=rho/bs[:,None];factor=np.tanh(z)/z
    return (h*factor[...,None]*m['point_output_scale'][:,None,None]).transpose(1,0,2)

def predict(model,conditions,frame_index=None,nonnegative=False,return_signed=False,return_mean=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c);mean=predict_mean(m,c)
    if return_mean:return mean[0] if single else mean
    base={k[9:]:v for k,v in m.items() if k.startswith('baseline_')}
    old=baseline_predict(base,c).reshape(len(c),int(m['frame_count']),int(m['point_count']),2)
    signed=mean[:,None,:,:]+float(m['dynamic_scale'])*(old-old.mean(1,keepdims=True))
    out=signed.reshape(len(c),int(m['frame_count']),-1) if return_signed else np.hypot(signed[...,0],signed[...,1])
    if nonnegative and not return_signed:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m['frame_count']):raise ValueError('Invalid saved frame index')
        out=out[:,frame_index]
    return out[0] if single else out

def metrics(p,y,ids,point_ids):
    e=p-y;se=np.sum(e*e,axis=(1,2));energy=np.sum(y*y,axis=(1,2));rm=np.sqrt(se/np.prod(y.shape[1:]));worst=int(np.argmax(rm));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape)
    return dict(relative_L2_pct=float(100*np.sqrt(se.sum()/energy.sum())),rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        worst_case_id=int(ids[worst]),worst_case_rmse_T=float(rm[worst]),max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),point_id=int(point_ids[loc[2]]),prediction_T=float(p[loc]),truth_T=float(y[loc]),signed_error_T=float(e[loc])),
        per_case=[dict(case_id=int(k),rmse_T=float(r),relative_L2_pct=float(100*np.sqrt(s/den))) for k,r,s,den in zip(ids,rm,se,energy)])

def mean_metrics(pred,target):
    e=pred-target;pm=np.hypot(pred[...,0],pred[...,1]);tm=np.hypot(target[...,0],target[...,1]);me=pm-tm
    return dict(signed_static_mean_relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(target)),signed_static_mean_component_rmse_T=float(np.sqrt(np.mean(e*e))),
        signed_static_mean_component_max_abs_T=float(np.abs(e).max()),norm_of_signed_mean_relative_L2_pct=float(100*np.linalg.norm(me)/np.linalg.norm(tm)),
        norm_of_signed_mean_rmse_T=float(np.sqrt(np.mean(me*me))),norm_of_signed_mean_max_abs_T=float(np.abs(me).max()),
        quantity_note='Static signed mean and its vector norm, not framewise |B| and not mean(|B|)')

def train(prepared,truth_path,out,config,protocol_path):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    prepared=Path(prepared);out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    identity=json.loads((prepared/'identity.json').read_text(encoding='utf-8'))
    if identity['test_labels_loaded'] or identity['full60_fit_labels_loaded'] or len(identity['train_case_ids'])!=42 or len(identity['validation_case_ids'])!=9:raise ValueError('Allowed label guard')
    with np.load(prepared/'mean_data.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(prepared/'geometry.npz',allow_pickle=False) as z:geometry={k:z[k] for k in z.files}
    with np.load(prepared/'baseline_model.npz',allow_pickle=False) as z:base={k:z[k] for k in z.files}
    with np.load(truth_path,allow_pickle=False) as z:truth=z['truth'];ids=z['case_ids']
    protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if sha(__file__)!=protocol['source_sha256'] or sha(prepared/'identity.json')!=protocol['prepared_identity_sha256'] or sha(prepared/'mean_data.npz')!=protocol['prepared_means_sha256']:raise ValueError('Predeclared source/data binding mismatch')
    if config not in protocol['candidates'] or a['case_ids'][:42].tolist()!=identity['train_case_ids'] or ids.tolist()!=identity['validation_case_ids']:raise ValueError('Protocol and partition guard')
    with threadpool_limits(limits=2):
        c=a['conditions'];target=a['signed_mean'];cs=float(c[:42,0].max());f=features(c,'cartesian',cs);fm=f[:42].mean(0);fs=np.maximum(f[:42].std(0),1e-8);x=(f-fm)/fs
        points=target.shape[1];den=float(a['truth_energy'][:42].sum());alpha=float(a['signed_dynamic_scale']);scale=np.sqrt(np.mean(np.sum(target[:42]**2,axis=-1),axis=0));scale=np.maximum(scale,scale.max()*.001)
        yn=target[:42]/scale[None,:,None];stretch=np.zeros(points);shear=np.zeros(points)
        if config['radial_kind']=='elliptic':
            centered=yn-yn.mean(0,keepdims=True);cov=np.einsum('cph,cpj->phj',centered,centered)/42
            eig,vec=np.linalg.eigh(cov);eig=np.maximum(eig,eig.max(axis=1,keepdims=True)*1e-3+1e-8)
            logs=np.log(eig);logs=logs-logs.mean(axis=1,keepdims=True);inveig=np.exp(np.clip(-logs,-np.log(2),np.log(2)))
            metric=np.einsum('phi,pi,pji->phj',vec,inveig,vec)
            upper=np.linalg.cholesky(metric).transpose(0,2,1)
            stretch=np.log(upper[:,0,0]);shear=upper[:,0,1]
        q0=np.exp(stretch)[None,:]*yn[...,0]+shear[None,:]*yn[...,1];q1=np.exp(-stretch)[None,:]*yn[...,1]
        rho=np.sqrt(q0*q0+q1*q1+1e-20);bs0=1.25*rho.max(axis=0)+config['minimum_saturation']
        ratio=np.minimum(rho/bs0[None,:],.999);inverse=np.arctanh(ratio)/np.maximum(ratio,1e-20);drive_target=yn*inverse[...,None]
        design=np.column_stack((np.ones(42),x[:42]));coef=np.linalg.solve(design.T@design+np.diag([0,1e-8,1e-8]),design.T@drive_target.reshape(42,-1)).reshape(3,points,2)
        initial=dict(drive_weight=coef[1:].transpose(1,0,2),drive_bias=coef[0],raw_saturation=np.log(np.expm1(bs0-config['minimum_saturation'])))
        if config['radial_kind']=='elliptic':initial.update(raw_stretch=np.arctanh(stretch/config['max_log_stretch']),raw_shear=np.arctanh(shear/config['max_shear']))
        params={k:torch.nn.Parameter(torch.tensor(v,dtype=torch.float32).contiguous()) for k,v in initial.items()}
        xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(target[:42].transpose(1,0,2),dtype=torch.float32);st=torch.tensor(scale[:,None,None],dtype=torch.float32)
        priors={k:torch.tensor(v,dtype=torch.float32) for k,v in initial.items() if k.startswith('raw_')}
        def forward(inp):
            h=torch.einsum('cf,pfh->pch',inp,params['drive_weight'])+params['drive_bias'][:,None,:]
            if config['radial_kind']=='elliptic':
                s=config['max_log_stretch']*torch.tanh(params['raw_stretch']);a0=config['max_shear']*torch.tanh(params['raw_shear']);q0=torch.exp(s)[:,None]*h[...,0]+a0[:,None]*h[...,1];q1=torch.exp(-s)[:,None]*h[...,1];rho=torch.sqrt(q0*q0+q1*q1+1e-20)
            else:rho=torch.sqrt(torch.sum(h*h,axis=-1)+1e-20)
            bs=torch.nn.functional.softplus(params['raw_saturation'])+config['minimum_saturation'];z=rho/bs[:,None]
            return h*(torch.tanh(z)/z)[...,None]*st
        def objective():
            e=forward(xt[:42])-yt
            prior=sum(torch.mean((params[k]-v)**2) for k,v in priors.items())
            return 60*torch.sum(e*e)/den+config['radial_prior']*prior
        history=[];best=float('inf');best_state=None;best_step=0;closures=0
        def checkpoint(step,stage,loss):
            nonlocal best,best_state,best_step
            with torch.no_grad():mean=forward(xt).numpy().transpose(1,0,2)
            signed=mean[42:,None,:,:]+alpha*a['validation_signed_dynamic'];p=np.hypot(signed[...,0],signed[...,1]);e=p-truth;rmse=float(np.sqrt(np.mean(e*e)))
            row=dict(step=step,stage=stage,objective=float(loss),train_signed_mean_relative_L2_pct=float(100*np.sqrt(60*np.sum((mean[:42]-target[:42])**2)/den)),validation_magnitude_relative_L2_pct=float(100*np.sqrt(np.sum(e*e)/np.sum(truth*truth))),validation_rmse_T=rmse,validation_max_abs_T=float(np.max(np.abs(e))),elapsed_seconds=time.perf_counter()-start);history.append(row)
            if rmse<best:best=rmse;best_step=step;best_state={k:p.detach().clone() for k,p in params.items()}
            return row
        def restore():
            with torch.no_grad():
                for k,p in params.items():p.copy_(best_state[k])
        opt=torch.optim.Adam(list(params.values()),lr=config['learning_rate']);checkpoint(0,'train_inverse_radial_OLS',objective().detach())
        for step in range(1,config['adam_steps']+1):
            opt.zero_grad();loss=objective();loss.backward();torch.nn.utils.clip_grad_norm_(list(params.values()),5);opt.step()
            if step%50==0:
                row=checkpoint(step,'Adam',loss.detach())
                if step%1000==0:print(json.dumps(dict(name=config['name'],progress=row)),flush=True)
        final_adam=history[-1].copy();restore()
        with torch.no_grad():
            for p in params.values():p.data=p.data.double()
        xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(target[:42].transpose(1,0,2),dtype=torch.float64);st=torch.tensor(scale[:,None,None],dtype=torch.float64);priors={k:torch.tensor(v,dtype=torch.float64) for k,v in initial.items() if k.startswith('raw_')}
        opt=torch.optim.LBFGS(list(params.values()),lr=1,max_iter=10,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
        for outer in range(1,config['lbfgs_blocks']+1):
            def closure():
                nonlocal closures
                opt.zero_grad();loss=objective();loss.backward()
                for p in params.values():
                    if p.grad is not None and not p.grad.is_contiguous():p.grad=p.grad.contiguous()
                closures+=1;return loss
            opt.step(closure);row=checkpoint(config['adam_steps']+outer*10,'LBFGS',objective().detach())
            if outer%10==0:print(json.dumps(dict(name=config['name'],progress=row)),flush=True)
        final_optimizer=history[-1].copy();restore()
        arrays=dict(radial_kind=np.asarray(config['radial_kind']),current_scale=np.asarray(cs),feature_mean=fm,feature_scale=fs,point_output_scale=scale,
            minimum_saturation=np.asarray(config['minimum_saturation']),max_log_stretch=np.asarray(config['max_log_stretch']),max_shear=np.asarray(config['max_shear']),
            point_count=np.asarray(points),frame_count=np.asarray(60),dynamic_scale=np.asarray(alpha),**{k:p.detach().numpy() for k,p in params.items()},**geometry)
        arrays.update({'baseline_'+k:v for k,v in base.items()})
        meta=dict(version=VERSION,configuration=config,train_case_ids=identity['train_case_ids'],validation_case_ids=identity['validation_case_ids'],excluded_case_ids=identity['excluded_case_ids'],
            source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),prepared_identity_sha256=sha(prepared/'identity.json'),prepared_means_sha256=sha(prepared/'mean_data.npz'),baseline_model_sha256=sha(prepared/'baseline_model.npz'),
            initialization='train42 scalar output energy and input scalers; train covariance determinant-one ellipse for elliptic; inverse tanh effective-drive OLS; all7/9 trainable parameters PyTorch gradients',
            checkpoint='minimum full9x60x1800 raw magnetic magnitude RMSE',point_weighting='uniform all1800',dynamic='immutable old train42 pure neural signed zero-temporal-mean output, train-only scale1.0',
            quantity='planar sqrt(Br^2+Bt^2), no Bz',units='T',xy_units='mm',frame_scope='all60 supplied first-cycle saved frames only',best_step=best_step,
            physics_scope='affine effective drive is a learned proxy, not observed H; radial saturation is a structural B-H analogy, not measured material law',test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
        arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False));np.savez_compressed(out/'model.npz',**arrays);(out/'model.npz.sha256').write_text(sha(out/'model.npz')+'\n')
        actual=predict_mean(arrays,c)
        with torch.no_grad():expected=forward(xt).numpy().transpose(1,0,2)
        replay=float(np.max(np.abs(actual-expected)))
        if replay>1e-10:raise AssertionError('NumPy and Torch64 static signed replay disagree')
        pred=predict(out/'model.npz',c[42:]);validation=metrics(pred,truth,ids,geometry['point_id'])
        if not np.isclose(validation['rmse_T'],best,atol=1e-7,rtol=2e-4):raise AssertionError('Saved full-field replay differs from selected checkpoint')
        with (out/'history.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        result=dict(name=config['name'],configuration=config,validation=validation,train_static_mean=mean_metrics(actual[:42],target[:42]),validation_static_mean=mean_metrics(actual[42:],target[42:]),
            train_signed_mean_relative_L2_pct=float(100*np.sqrt(60*np.sum((actual[:42]-target[:42])**2)/den)),new_trainable_parameters=sum(p.numel() for p in params.values()),new_parameters_per_point=sum(p.numel() for p in params.values())//points,
            best_step=best_step,final_adam=final_adam,final_optimizer=final_optimizer,lbfgs_closure_evaluations=closures,history_checkpoint_count=len(history),elapsed_seconds=time.perf_counter()-start,
            numpy_torch_float64_replay_max_abs_difference_T=replay,model_sha256=sha(out/'model.npz'),source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),
            test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',type=Path,required=True);parser.add_argument('--truth',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();train(args.prepared,args.truth,args.output,json.loads(args.config.read_text(encoding='utf-8')),args.protocol)
