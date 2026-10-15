"""Gradient-trained Gaussian RBF residual NN on immutable V5 rotor NN.

Trainable pointwise RBF centers, anisotropic widths and output weights. The
original NN cycle/mean is frozen. Only train42 labels train this new network.
Old validation9 selects checkpoints; no test/full60 fitting labels are read.
"""
from pathlib import Path
import argparse, importlib.util, json, time
import numpy as np
import rbf_pointwise as common

ROOT=Path(__file__).resolve().parent
WARM=ROOT.parent/'new60_opt_v5/magnitude_objective_nn/runs/magnitude_ft_gate2_lambda0p02_seed42/model.npz'
MODULE=ROOT.parent/'new60_opt_v5/magnitude_objective_nn/magnitude_point_nn.py'

def load_warm():
    spec=importlib.util.spec_from_file_location('v6_immutable_warm_nn',MODULE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,module.load_model(WARM)

def farthest_centers(x,count):
    ids=[int(np.argmin(np.sum(x*x,axis=1)))];dist=np.full(len(x),np.inf)
    for _ in range(count-1):
        dist=np.minimum(dist,np.sum((x-x[ids[-1]])**2,axis=1));ids.append(int(np.argmax(dist)))
    return x[ids]

def residual(m,c):
    x=(common.coords(c,float(m['current_scale']),0.)-m['input_mean'])/m['input_scale']
    centers=m['initial_centers']+.5*np.tanh(m['center_delta']);width=np.exp(np.clip(m['log_width'],-1.5,2.))
    z=(x[None,:,None,:]-centers[:,None,:,:])/width[:,None,:,:]
    h=np.exp(-.5*np.sum(z*z,axis=-1))
    aug=np.column_stack((np.ones(len(x)),x))
    return (np.einsum('pnh,phk->pnk',h,m['weights'])+np.einsum('ni,pik->pnk',aug,m['linear'])).transpose(1,0,2)

def predict(model,conditions,frame_index=None):
    path=Path(model)
    if common.sha(path)!=path.with_suffix('.npz.sha256').read_text().strip():raise ValueError('RBF NN SHA mismatch')
    with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    meta=json.loads(str(m['metadata_json']))
    if common.sha(WARM)!=meta['warm_sha256'] or common.sha(MODULE)!=meta['warm_module_sha256']:raise ValueError('Frozen warm model/source mismatch')
    c=np.asarray(conditions,dtype=float);single=c.ndim==1;c=np.atleast_2d(c)
    if c.shape[1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Native nonnegative current and angle required')
    module,warm=load_warm();signed=module.predict(warm,c,return_signed=True).reshape(len(c),60,1800,2)
    signed=signed+residual(m,c)[:,None,:,:];out=np.hypot(signed[...,0],signed[...,1])
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<60:raise ValueError('Saved frame index 0..59 only')
        out=out[:,frame_index]
    return out[0] if single else out

def run():
    import torch
    torch.set_num_threads(2);torch.manual_seed(42);start=time.perf_counter();out=ROOT/'rbf_neural';out.mkdir(exist_ok=False)
    split=json.loads((ROOT.parent/'new60_20261014/cache_v1/split.json').read_text());tr=split['train'];va=split['validation'];cache=ROOT.parent/'new60_20261014/cache_v1/rotor.npz'
    configs=[dict(hidden=h,lambda_signed=l,adam_steps=1200,lr=.002,center_lr=.001,head_prior=1e-4,seed=42) for h in [8,16] for l in [.02,.2]]
    common.dump(out/'protocol.json',dict(configs=configs,train_case_ids=tr,validation_case_ids=va,excluded_test_case_ids=split['test'],cache_sha256=common.sha(cache),source_sha256=common.sha(__file__),warm_sha256=common.sha(WARM),warm_module_sha256=common.sha(MODULE),
        checkpoint='Full9 validation original Tesla magnetic magnitude RMSE',loss='Fullframe raw T magnitude MSE + lambda signed static mean MSE +1e-4 output-weight squared prior',
        trainable='Gaussian RBF pointwise centers, two axis widths, output weights and linear skip; base NN frozen',frames='6 frames per step deterministic cycle covers all60 every10steps',test_labels_read=False,full60_fit_used=False,actual_new_gradient_NN_runs=4))
    with np.load(cache,allow_pickle=False) as z:c=z['conditions'];pid=z['point_id'];xy=z['xy'];theta=z['theta']
    train=common.rows(cache,tr).reshape(42,60,1800,2);valid=common.rows(cache,va).reshape(9,60,1800,2);truth=np.hypot(valid[...,0],valid[...,1]);targetmean=train.mean(1)
    module,warm=load_warm();allc=c[tr+va];base=module.predict(warm,allc,return_signed=True).reshape(51,60,1800,2)
    current_scale=float(np.median(c[tr,0]));raw=common.coords(allc,current_scale,0.);mu=raw[:42].mean(0);sd=raw[:42].std(0);x=(raw-mu)/sd
    xt=torch.tensor(x,dtype=torch.float32);bt=torch.tensor(base[:42],dtype=torch.float32);yt=torch.tensor(np.hypot(train[...,0],train[...,1]),dtype=torch.float32);mean=torch.tensor(targetmean,dtype=torch.float32);basemean=torch.tensor(base[:42].mean(1),dtype=torch.float32)
    rows=[]
    for number,config in enumerate(configs):
        dest=out/f'run{number:02d}';dest.mkdir();h=config['hidden'];centers=farthest_centers(x[:42],h);init=np.broadcast_to(centers,(1800,h,2)).copy()
        initial=torch.tensor(init,dtype=torch.float32);delta=torch.nn.Parameter(torch.zeros_like(initial));logw=torch.nn.Parameter(torch.full_like(initial,np.log(.8)));weight=torch.nn.Parameter(torch.zeros((1800,h,2)));linear=torch.nn.Parameter(torch.zeros((1800,3,2)));params=[delta,logw,weight,linear]
        def forward(inp):
            center=initial+.5*torch.tanh(delta);width=torch.exp(logw.clamp(-1.5,2.));z=(inp[None,:,None,:]-center[:,None,:,:])/width[:,None,:,:];act=torch.exp(-.5*(z*z).sum(-1));aug=torch.cat((torch.ones((len(inp),1)),inp),dim=1)
            return (torch.einsum('pnh,phk->pnk',act,weight)+torch.einsum('ni,pik->pnk',aug,linear)).transpose(0,1)
        optimizer=torch.optim.Adam([dict(params=[weight,linear],lr=config['lr']),dict(params=[delta,logw],lr=config['center_lr'])]);best=np.inf;state=None;best_step=None;hist=[]
        for step in range(config['adam_steps']+1):
            if step%50==0:
                with torch.no_grad():res=forward(xt).numpy()
                value=base[42:]+res[42:,None,:,:];pred=np.hypot(value[...,0],value[...,1]);met=common.metrics(pred,truth)
                hist.append(dict(step=step,validation=met))
                if met['rmse_T']<best:best=met['rmse_T'];state=[p.detach().clone() for p in params];best_step=step
                if step%300==0:print('RBFNN',number,step,'validation',met['relative_L2_pct'],flush=True)
            if step==config['adam_steps']:break
            frames=list(range(step%10,60,10));correction=forward(xt[:42]);signed=bt[:,frames]+correction[:,None,:,:];mag=torch.linalg.vector_norm(signed,dim=-1)
            loss=((mag-yt[:,frames])**2).mean()+config['lambda_signed']*((basemean+correction-mean)**2).mean()+config['head_prior']*((weight*weight).mean()+(linear*linear).mean())
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step()
        arrays=dict(current_scale=np.array(current_scale),input_mean=mu,input_scale=sd,initial_centers=init,point_id=pid,xy=xy,theta=theta)
        arrays.update(dict(zip(['center_delta','log_width','weights','linear'],[v.numpy().astype(np.float64) for v in state])))
        meta=dict(format='V6-actual-gradient-trained-pointwise-Gaussian-RBF-NN',configuration=config,best_step=best_step,train_case_ids=tr,validation_case_ids=va,warm_sha256=common.sha(WARM),warm_module_sha256=common.sha(MODULE),source_sha256=common.sha(__file__),test_labels_read=False,full60_fit_used=False,gradient_trained=True)
        arrays['metadata_json']=np.array(json.dumps(meta));path=dest/'model.npz';np.savez_compressed(path,**arrays);path.with_suffix('.npz.sha256').write_text(common.sha(path)+'\n')
        p=predict(path,c[va]);met=common.metrics(p,truth)
        np.savez_compressed(dest/'validation_predictions.npz',prediction=p,case_ids=np.array(va));common.dump(dest/'history.json',hist)
        row=dict(number=number,configuration=config,best_step=best_step,validation=met,model_sha256=common.sha(path),source_sha256=common.sha(__file__),actual_gradient_NN_training=True);common.dump(dest/'result.json',row);rows.append(row)
    common.dump(out/'results.json',dict(rows=rows,actual_new_gradient_NN_runs=len(rows),best=min(rows,key=lambda v:v['validation']['rmse_T']),elapsed_seconds=time.perf_counter()-start,test_labels_read=False))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run()
