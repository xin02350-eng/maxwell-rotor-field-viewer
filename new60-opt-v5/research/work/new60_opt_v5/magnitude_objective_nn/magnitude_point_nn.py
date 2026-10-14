"""V5 actual warmstart NN fine-tuning on original-T full magnetic magnitude.

All trainable parameters belong to the signed mean two/three-gate network.
The old train42 neural dynamics and all train42 scalers stay frozen. No test,
all60 fitting, or train51 model is loaded. NumPy inference is self-contained.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import zipfile
import numpy as np

VERSION='v5-signed-regime-NN-warmstart-fullfield-magnitude-objective'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def selected_rows(path,key,indices):
    mapping={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as archive,archive.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or len(shape)!=3 or shape[0]!=60 or shape[1:]!=(60,3600) or dtype.kind!='f':raise ValueError('Invalid full source field geometry')
        out=np.empty((len(indices),*shape[1:]),dtype=dtype);block=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(shape[0]):
            if i in mapping:
                raw=f.read(block)
                if len(raw)!=block:raise ValueError('Short selected field row')
                out[mapping[i]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(block,1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite selected train labels')
    return out

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

def baseline_predict(m,c):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));scale=float(m['current_scale'])
    v=(features(c,str(m['feature_kind']),scale)-m['feature_mean'])/m['feature_scale']
    for j in range(int(m['layer_count'])):
        v=v@m[f'w{j}']+m[f'b{j}']
        if j<int(m['layer_count'])-1:v=np.tanh(v) if str(m['activation'])=='tanh' else v*sigmoid(v)
    x=features(c,'cartesian',scale);degree=int(m['base_degree'])
    poly=np.column_stack([x[:,0]**i*x[:,1]**j for total in range(degree+1) for i in range(total+1) for j in [total-i]])
    latent=v*m['output_scale']+poly@m['base_weights'];coef=(m['cycle_mean']+latent@m['cycle_basis']).reshape(len(c),int(m['frame_count']),-1)
    return m['spatial_mean']+coef@m['spatial_basis'].T

def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if sha(path)!=path.with_suffix(path.suffix+'.sha256').read_text().strip():raise ValueError('Magnitude neural model SHA mismatch')
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def predict_mean(m,c):
    x=(features(c,'cartesian',float(m['current_scale']))-m['feature_mean'])/m['feature_scale']
    v=np.einsum('cf,pfh->pch',x,m['linear_weight'])+m['linear_bias'][:,None,:]
    z=np.einsum('cf,pfk->pck',x,m['gate_weight'])+m['gate_bias'][:,None,:]
    aug=np.column_stack((np.ones(len(x)),x));expert=np.einsum('cf,pkfh->pckh',aug,m['expert_weight'])
    v=v+np.sum(sigmoid(z)[...,None]*expert,axis=2)
    return (v*m['output_scale'][:,None,:]+m['output_mean'][:,None,:]).transpose(1,0,2)

def predict(model,conditions,frame_index=None,nonnegative=False,return_signed=False,return_mean=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c);mean=predict_mean(m,c)
    if return_mean:return mean[0] if single else mean
    base={k[9:]:v for k,v in m.items() if k.startswith('baseline_')};old=baseline_predict(base,c).reshape(len(c),int(m['frame_count']),int(m['point_count']),2)
    signed=mean[:,None,:,:]+float(m['dynamic_scale'])*(old-old.mean(1,keepdims=True))
    out=signed.reshape(len(c),int(m['frame_count']),-1) if return_signed else np.hypot(signed[...,0],signed[...,1])
    if nonnegative and not return_signed:out=np.maximum(out,0)
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<int(m['frame_count']):raise ValueError('Invalid saved frame index')
        out=out[:,frame_index]
    return out[0] if single else out

def metrics(p,y,ids,point_ids):
    e=p-y;se=np.sum(e*e,axis=(1,2));energy=np.sum(y*y,axis=(1,2));rm=np.sqrt(se/np.prod(y.shape[1:]));worst=int(np.argmax(rm));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape)
    return dict(relative_L2_pct=float(100*np.sqrt(se.sum()/energy.sum())),rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),worst_case_id=int(ids[worst]),worst_case_rmse_T=float(rm[worst]),
        max_abs_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2]),point_id=int(point_ids[loc[2]]),prediction_T=float(p[loc]),truth_T=float(y[loc]),signed_error_T=float(e[loc])),
        per_case=[dict(case_id=int(k),rmse_T=float(r),relative_L2_pct=float(100*np.sqrt(s/den))) for k,r,s,den in zip(ids,rm,se,energy)])

def mean_metrics(pred,target):
    e=pred-target;pm=np.hypot(pred[...,0],pred[...,1]);tm=np.hypot(target[...,0],target[...,1]);me=pm-tm
    return dict(signed_static_mean_relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(target)),signed_static_mean_component_rmse_T=float(np.sqrt(np.mean(e*e))),signed_static_mean_component_max_abs_T=float(np.abs(e).max()),
        norm_of_signed_mean_relative_L2_pct=float(100*np.linalg.norm(me)/np.linalg.norm(tm)),norm_of_signed_mean_rmse_T=float(np.sqrt(np.mean(me*me))),norm_of_signed_mean_max_abs_T=float(np.abs(me).max()),quantity_note='Static signed mean/norm(mean), distinct from fullframe |B| and mean(|B|)')

def train(prepared,truth_path,cache,out,config,protocol_path):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);torch.manual_seed(config['seed']);prepared=Path(prepared);out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text(encoding='utf-8'));identity=json.loads((prepared/'identity.json').read_text(encoding='utf-8'))
    if sha(__file__)!=protocol['source_sha256'] or config not in protocol['candidates']:raise ValueError('Predeclared source and config binding')
    if identity['test_labels_loaded'] or identity['full60_fit_labels_loaded'] or len(identity['train_case_ids'])!=42 or len(identity['validation_case_ids'])!=9:raise ValueError('Allowed labels guard')
    if sha(prepared/'identity.json')!=protocol['prepared_identity_sha256'] or sha(cache)!=protocol['raw_cache_sha256']:raise ValueError('Prepared/raw cache provenance mismatch')
    with np.load(prepared/'mean_data.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(truth_path,allow_pickle=False) as z:truth=z['truth'];ids=z['case_ids']
    warm_path=Path(config['warmstart_model']);m=load_model(warm_path);warmmeta=json.loads(str(m['metadata_json']))
    if sha(warm_path)!=config['warmstart_model_sha256'] or warmmeta['source_sha256']!=config['warmstart_source_sha256'] or warmmeta['train_case_ids']!=identity['train_case_ids'] or warmmeta['test_labels_loaded'] or warmmeta['full60_fit_labels_loaded']:raise ValueError('Actualtrain42 warmstart identity')
    if a['case_ids'][:42].tolist()!=identity['train_case_ids'] or ids.tolist()!=identity['validation_case_ids'] or not np.array_equal(ids,a['case_ids'][42:]):raise ValueError('Partition order guard')
    with np.load(cache,allow_pickle=False) as z:
        cacheids=z['case_ids'];cacheconditions=z['conditions'];point_ids=z['point_id'];xy=z['xy'];theta=z['theta'];time_s=z['time_s']
    if cacheids.tolist()!=list(range(60)) or not np.array_equal(cacheconditions[a['case_ids']],a['conditions']) or not np.array_equal(point_ids,m['point_id']) or not np.array_equal(xy,m['xy']) or not np.array_equal(theta,m['theta']) or not np.array_equal(time_s,m['time_s']):raise ValueError('Source geometry/conditions identity')
    # Only these42 numeric rows are interpreted; all other source rows are byte-skipped.
    true_signed=selected_rows(cache,'field',identity['train_case_ids']).astype(np.float64).reshape(42,60,1800,2)
    traintruth=np.hypot(true_signed[...,0],true_signed[...,1])
    if not np.allclose(true_signed.mean(1),a['signed_mean'][:42],atol=1e-12,rtol=1e-12):raise ValueError('True fullframe train rows disagree with prepared static targets')
    if not np.allclose(np.sum(traintruth*traintruth,axis=(1,2)),a['truth_energy'][:42],atol=1e-9,rtol=1e-12):raise ValueError('True fullframe training energy identity')
    with threadpool_limits(limits=2):
        base={k[9:]:v for k,v in m.items() if k.startswith('baseline_')};c=a['conditions'];target=a['signed_mean'];old=baseline_predict(base,c[:42]).reshape(42,60,1800,2);dynamic=float(m['dynamic_scale'])*(old-old.mean(1,keepdims=True));valdynamic=a['validation_signed_dynamic']
        x=(features(c,'cartesian',float(m['current_scale']))-m['feature_mean'])/m['feature_scale'];keys=('linear_weight','linear_bias','gate_weight','gate_bias','expert_weight');params={k:torch.nn.Parameter(torch.tensor(m[k],dtype=torch.float32).contiguous()) for k in keys}
        xt=torch.tensor(x,dtype=torch.float32);cm=torch.tensor(m['output_mean'][:,None,:],dtype=torch.float32);sc=torch.tensor(m['output_scale'][:,None,:],dtype=torch.float32);yt=torch.tensor(target[:42],dtype=torch.float32);dyn=torch.tensor(dynamic,dtype=torch.float32);y=torch.tensor(traintruth,dtype=torch.float32)
        def forward(inp):
            v=torch.einsum('cf,pfh->pch',inp,params['linear_weight'])+params['linear_bias'][:,None,:];z=torch.einsum('cf,pfk->pck',inp,params['gate_weight'])+params['gate_bias'][:,None,:];aug=torch.cat((torch.ones((len(inp),1),dtype=inp.dtype),inp),axis=1);expert=torch.einsum('cf,pkfh->pckh',aug,params['expert_weight'])
            return (v+torch.sum(torch.sigmoid(z)[...,None]*expert,axis=2))*sc+cm
        def objective(frames=None):
            mean=forward(xt[:42]).transpose(0,1)
            s=mean[:,None,:,:]+(dyn if frames is None else dyn[:,frames]);t=y if frames is None else y[:,frames]
            magnetic=torch.linalg.vector_norm(s,dim=-1);mag_mse=torch.mean((magnetic-t)**2);signed_mse=torch.mean((mean-yt)**2)
            return mag_mse+config['lambda_signed']*signed_mse,mag_mse,signed_mse
        history=[];best=float('inf');best_state=None;best_step=0;closures=0
        def checkpoint(step,stage,batch,loss):
            nonlocal best,best_state,best_step
            with torch.no_grad():mean=forward(xt).numpy().transpose(1,0,2)
            trsigned=mean[:42,None,:,:]+dynamic;trpred=np.hypot(trsigned[...,0],trsigned[...,1]);tre=trpred-traintruth;v=mean[42:,None,:,:]+valdynamic;pred=np.hypot(v[...,0],v[...,1]);e=pred-truth;rmse=float(np.sqrt(np.mean(e*e)));reg=float(np.mean((mean[:42]-target[:42])**2))
            row=dict(step=step,stage=stage,frame_batch='' if batch is None else ','.join(map(str,batch)),sampled_objective_T2=float(loss),train_full60_magnitude_mse_T2=float(np.mean(tre*tre)),train_signed_mean_regularizer_T2=reg,train_full_objective_T2=float(np.mean(tre*tre))+config['lambda_signed']*reg,
                train_full60_magnitude_relative_L2_pct=float(100*np.linalg.norm(tre)/np.linalg.norm(traintruth)),train_full60_magnitude_rmse_T=float(np.sqrt(np.mean(tre*tre))),train_full60_magnitude_max_abs_T=float(np.abs(tre).max()),
                validation_magnitude_relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(truth)),validation_rmse_T=rmse,validation_max_abs_T=float(np.abs(e).max()),elapsed_seconds=time.perf_counter()-start);history.append(row)
            if rmse<best:best=rmse;best_step=step;best_state={k:p.detach().clone() for k,p in params.items()}
            return row
        def restore():
            with torch.no_grad():
                for k,p in params.items():p.copy_(best_state[k])
        optimizer=torch.optim.Adam([dict(params=[params[k] for k in ('linear_weight','linear_bias','expert_weight')],lr=config['learning_rate']),dict(params=[params[k] for k in ('gate_weight','gate_bias')],lr=config['gate_learning_rate'])])
        checkpoint(0,'warmstart_train42_val_selected',None,objective()[0].detach())
        for step in range(1,config['adam_steps']+1):
            offset=(step-1)%10;frames=list(range(offset,60,10));optimizer.zero_grad();loss,_,_=objective(frames);loss.backward();torch.nn.utils.clip_grad_norm_(list(params.values()),5);optimizer.step()
            if step%50==0:
                row=checkpoint(step,'Adam_fixed6frame_cycle',frames,loss.detach())
                if step%500==0:print(json.dumps(dict(name=config['name'],progress=row)),flush=True)
        final_adam=history[-1].copy();restore()
        with torch.no_grad():
            for p in params.values():p.data=p.data.double()
        xt=torch.tensor(x,dtype=torch.float64);cm=torch.tensor(m['output_mean'][:,None,:],dtype=torch.float64);sc=torch.tensor(m['output_scale'][:,None,:],dtype=torch.float64);yt=torch.tensor(target[:42],dtype=torch.float64);dyn=torch.tensor(dynamic,dtype=torch.float64);y=torch.tensor(traintruth,dtype=torch.float64)
        optimizer=torch.optim.LBFGS(list(params.values()),lr=1,max_iter=10,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
        for outer in range(1,config['lbfgs_blocks']+1):
            def closure():
                nonlocal closures
                optimizer.zero_grad();loss=objective()[0];loss.backward()
                for p in params.values():
                    if p.grad is not None and not p.grad.is_contiguous():p.grad=p.grad.contiguous()
                closures+=1;return loss
            optimizer.step(closure);row=checkpoint(config['adam_steps']+outer*10,'LBFGS_all60_true_frames',None,objective()[0].detach())
            if outer%5==0:print(json.dumps(dict(name=config['name'],progress=row)),flush=True)
        final_optimizer=history[-1].copy();restore();arrays=dict(m);arrays.update({k:p.detach().numpy() for k,p in params.items()})
        meta=dict(version=VERSION,configuration=config,train_case_ids=identity['train_case_ids'],validation_case_ids=identity['validation_case_ids'],excluded_case_ids=identity['excluded_case_ids'],
            source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),prepared_identity_sha256=sha(prepared/'identity.json'),prepared_means_sha256=sha(prepared/'mean_data.npz'),raw_cache_sha256=sha(cache),
            raw_label_rows_loaded=identity['train_case_ids'],warmstart_model_sha256=config['warmstart_model_sha256'],warmstart_source_sha256=config['warmstart_source_sha256'],warmstart_kind='actualtrain42 V4 signedmean NN, checkpoint selected using repeatedly reused oldval9; nottrain51/notfit60',
            fine_tuning_objective='OriginalTesla fullframe magnitude MSE +lambda_signed originalTesla static signedmean component MSE; no perpointstd loss scaling',frame_sampling='deterministic6savedframes perAdamstep [offset,offset+10,...offset+50]; offset=(step-1)%10; coversall60 each10steps; LBFGS full60',
            checkpoint='minimum full9x60x1800 raw magnetic magnitude RMSE',point_weighting='uniform all1800',dynamic='immutable original train42 V2 signed pureNN zero-temporal-mean, fixedscale1.0',scalers='allinput/output scalers unchanged from actualtrain42 warmstart',
            baseline_model_sha256=warmmeta['baseline_model_sha256'],quantity='planar sqrt(Br^2+Bt^2), no Bz',units='T',xy_units='mm',frame_scope='all60 supplied first-cycle savedframes only',best_step=best_step,
            validation_scope='old9 validation repeatedly reused for prior model selection; not fresh validation',test_scope='original9 test already seen in project V1-V4; future scoring is reusedtest, notblind; thistraining branch neverloads testlabels',test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,train51_labels_loaded=False)
        arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False));np.savez_compressed(out/'model.npz',**arrays);(out/'model.npz.sha256').write_text(sha(out/'model.npz')+'\n')
        actual=predict_mean(arrays,c)
        with torch.no_grad():expected=forward(xt).numpy().transpose(1,0,2)
        replay=float(np.max(np.abs(actual-expected)))
        if replay>1e-10:raise AssertionError('NumPy and Torch64 signedmean replay mismatch')
        trainpred=predict(out/'model.npz',c[:42]);valpred=predict(out/'model.npz',c[42:]);validation=metrics(valpred,truth,ids,point_ids)
        if not np.isclose(validation['rmse_T'],best,atol=1e-7,rtol=2e-4):raise AssertionError('Saved fullvalidation disagrees with checkpoint')
        train_signed=actual[:42,None,:,:]+dynamic;ts_err=train_signed-true_signed
        with (out/'history.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        result=dict(name=config['name'],configuration=config,validation=validation,train_full60_magnitude=metrics(trainpred,traintruth,a['case_ids'][:42],point_ids),train_static_mean=mean_metrics(actual[:42],target[:42]),validation_static_mean=mean_metrics(actual[42:],target[42:]),
            train_full60_signed_components=dict(relative_L2_pct=float(100*np.linalg.norm(ts_err)/np.linalg.norm(true_signed)),component_rmse_T=float(np.sqrt(np.mean(ts_err*ts_err))),component_max_abs_T=float(np.abs(ts_err).max())),
            new_gradient_fine_tuning_run=True,from_scratch_training=False,new_trainable_parameters=sum(p.numel() for p in params.values()),new_parameters_per_point=sum(p.numel() for p in params.values())//1800,
            best_step=best_step,final_adam=final_adam,final_optimizer=final_optimizer,history_checkpoint_count=len(history),lbfgs_closure_evaluations=closures,elapsed_seconds=time.perf_counter()-start,
            raw_label_rows_loaded=identity['train_case_ids'],numpy_torch_float64_replay_max_abs_difference_T=replay,model_sha256=sha(out/'model.npz'),source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),
            test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,train51_labels_loaded=False)
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('prepared','truth','cache','output','config','protocol'):parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args();train(args.prepared,args.truth,args.cache,args.output,json.loads(args.config.read_text(encoding='utf-8')),args.protocol)
