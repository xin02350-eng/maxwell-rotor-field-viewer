"""Independent-frame neural coordinate kernels, optimized on training-only LOO.

The small trainable tanh embeddings change local condition distances. A ridge
kernel head reconstructs train-only spatial POD scores. Validation chooses
checkpoints; no test labels are opened. Pure NumPy prediction is included.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, csv
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
V1=Path(__file__).resolve().parent.parent/'new60_20261014'
sys.path.insert(0,str(V1/'code'))
import hybrid_cycle_rbf as safe

def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def raw_features(c,kind,cap):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64)); a=np.deg2rad(c[:,1]);i=c[:,0]
    if kind=='native': return c
    if kind=='saturated': i=cap*np.tanh(i/cap)
    return np.stack([i*np.cos(a),i*np.sin(a)],axis=1)

def np_embedding(model,x):
    x=(raw_features(x,str(model['coordinate']),float(model['cap']))-model['input_mean'])/model['input_scale']
    h=np.tanh(np.einsum('nd,tdh->tnh',x,model['w1'])+model['b1'][:,None,:])
    warp=np.tanh(np.einsum('tnh,thd->tnd',h,model['w2'])+model['b2'][:,None,:])
    return (x[None,:,:]+.5*warp)/np.exp(model['log_length'])[:,None,:]

def np_kernel(x,c,kind):
    rr=np.sum((x[:,:,None,:]-c[:,None,:,:])**2,axis=-1)
    if kind=='gaussian':return np.exp(-.5*rr)
    q=np.sqrt(5*np.maximum(rr,1e-24));return (1+q+q*q/3)*np.exp(-q)

def predict(model,conditions,frame_index=None):
    if isinstance(model,(str,Path)):
        with np.load(model,allow_pickle=False) as z: model={k:z[k] for k in z.files}
    one=np.asarray(conditions).ndim==1
    x=np_embedding(model,conditions); c=np_embedding(model,model['train_conditions'])
    k=np_kernel(x,c,str(model['kernel']))
    pred=np.einsum('tnc,tcr->ntr',k,model['alpha'])*float(model['output_scale'])+model['cycle_coefficient_mean'][None,:,:]
    if frame_index is not None: pred=pred[:,int(frame_index)]
    out=model['field_mean']+pred@model['spatial_basis'].T
    return out[0] if one else out

def run(region,outdir,quantity):
    import torch
    from sklearn.utils.extmath import randomized_svd
    torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
    out=Path(outdir)/f'{region}_{quantity}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True); start=time.perf_counter()
    a,s,meta=safe._load_cache(V1/'cache_v1'/f'{region}.npz',V1/'cache_v1/split.json')
    tr,va=np.array(s['train']),np.array(s['validation'])
    ytr=np.asarray(a['field'][tr],dtype=np.float64); yva=np.asarray(a['field'][va],dtype=np.float64)
    if quantity=='magnitude':
        if meta['component_count']!=2:raise ValueError('No vector norm for Br-only gap')
        ytr=np.linalg.norm(ytr.reshape(42,60,-1,2),axis=-1)
        yva=np.linalg.norm(yva.reshape(9,60,-1,2),axis=-1)
        meta.update(quantity='sqrt(Br^2+Bt^2)',components=['B_planar_magnitude'],component_count=1,layout='scalar')
        mean=ytr.mean((0,1)); q=ytr.reshape(-1,ytr.shape[-1])-mean
        _,sing,vt=randomized_svd(q,n_components=384,n_iter=5,random_state=42)
        basis=vt.T; tc=(ytr-mean)@basis; vc=(yva-mean)@basis
    else:
        mean=np.asarray(a['mean'],dtype=np.float64);basis=np.asarray(a['basis'],dtype=np.float64)
        tc=(ytr-mean)@basis;vc=(yva-mean)@basis
    cm=tc.mean(0); scale=float(np.sqrt(np.mean((tc-cm)**2)));scale=max(scale,1e-9)
    target=np.transpose((tc-cm)/scale,(1,0,2)); n,t,r=tc.shape
    vcf=vc; projection=mean+vcf@basis.T
    floor=safe._metrics(projection,yva)
    cap=float(np.median(a['conditions'][tr,0])); traincond=a['conditions'][tr];valcond=a['conditions'][va]
    np.savez_compressed(out/'representation.npz',field_mean=mean,spatial_basis=basis,cycle_coefficient_mean=cm,output_scale=np.array(scale),train_conditions=traincond,theta=a['theta'],point_id=a['point_id'],xy=a['xy'],case_ids_train=a['case_ids'][tr],case_ids_validation=a['case_ids'][va])
    specs=[dict(coordinate=co,kernel=k,length=l,ridge=rd,seed=42) for co in ('cartesian','saturated') for k,l,rd in [('gaussian',1.2,.001),('matern52',2.4,.001),('matern52',2.4,.01),('matern52',1.2,.01)]]
    dump(out/'protocol.json',dict(specs=specs,max_steps=800,checkpoint_interval=20,training_objective='conditional leave-one-training-case-out coefficient MSE with fixed train-only POD; not independent CV',neural_embedding='60 independent two-input tanh(8)-tanh(2) residual coordinate warps, temporal smoothness; trained weights affect predictions',quantity=meta['quantity'],metadata=meta,train_ids=a['case_ids'][tr].tolist(),validation_ids=a['case_ids'][va].tolist(),test_labels_accessed=False,source_code_sha256=safe.digest(__file__),cache_sha256=safe.digest(V1/'cache_v1'/f'{region}.npz'),validation_spatial_projection_floor=floor))
    targett=torch.tensor(target); rows=[]; overallbest=np.inf
    for index,spec in enumerate(specs):
        torch.manual_seed(spec['seed']); raw=raw_features(traincond,spec['coordinate'],cap);mu=raw.mean(0);std=np.maximum(raw.std(0),1e-8)
        x=torch.tensor((raw-mu)/std)
        w1=torch.nn.Parameter(torch.randn(60,2,8)*.3);b1=torch.nn.Parameter(torch.zeros(60,8))
        w2=torch.nn.Parameter(torch.randn(60,8,2)*.01);b2=torch.nn.Parameter(torch.zeros(60,2))
        logl=torch.nn.Parameter(torch.full((60,2),np.log(spec['length'])))
        parameters=[w1,b1,w2,b2,logl]; optimizer=torch.optim.Adam(parameters,lr=.01)
        best=np.inf;hist=[];bestmodel=None
        for step in range(801):
            h=torch.tanh(torch.einsum('nd,tdh->tnh',x,w1)+b1[:,None,:]);warp=torch.tanh(torch.einsum('tnh,thd->tnd',h,w2)+b2[:,None,:])
            emb=(x[None,:,:]+.5*warp)/torch.exp(logl.clamp(-3,3))[:,None,:]
            rr=((emb[:,:,None,:]-emb[:,None,:,:])**2).sum(-1)
            if spec['kernel']=='gaussian':k=torch.exp(-.5*rr)
            else:
                q=torch.sqrt(5*rr.clamp_min(1e-24));k=(1+q+q*q/3)*torch.exp(-q)
            k=k+torch.eye(42)[None,:,:]*spec['ridge']
            inv=torch.linalg.inv(k);alpha=inv@targett
            loo=alpha/torch.diagonal(inv,dim1=-2,dim2=-1)[:,:,None]
            smooth=((w1[1:]-w1[:-1])**2).mean()+((w2[1:]-w2[:-1])**2).mean()+((logl[1:]-logl[:-1])**2).mean()
            reg=(w2*w2).mean()
            loss=(loo*loo).mean()+.0005*smooth+.0001*reg
            if step%20==0:
                model=dict(coordinate=np.array(spec['coordinate']),kernel=np.array(spec['kernel']),cap=np.array(cap),input_mean=mu,input_scale=std,w1=w1.detach().numpy().copy(),b1=b1.detach().numpy().copy(),w2=w2.detach().numpy().copy(),b2=b2.detach().numpy().copy(),log_length=logl.detach().clamp(-3,3).numpy().copy(),alpha=alpha.detach().numpy().copy(),train_conditions=traincond,field_mean=mean,spatial_basis=basis,cycle_coefficient_mean=cm,output_scale=np.array(scale),theta=a['theta'],point_id=a['point_id'],xy=a['xy'])
                pred=predict(model,valcond);metrics=safe._metrics(pred,yva)
                hist.append(dict(step=step,loo_objective=float(loss.detach()),validation=metrics))
                if metrics['rmse_T']<best:best=metrics['rmse_T'];bestmodel=model;beststep=step
                if step%100==0:print(region,quantity,index,step,metrics['relative_L2_pct'],float(loss.detach()),flush=True)
            if step==800:break
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.);optimizer.step()
        name=f"frame_kernel_{index:02d}";d=out/name;d.mkdir();pred=predict(bestmodel,valcond);exact=safe._metrics(pred,yva)
        bestmodel['metadata_json']=np.array(json.dumps(dict(format='new60_neural_frame_kernel_v2',region=region,quantity=meta['quantity'],spec=spec,best_step=beststep,validation=exact,test_labels_accessed=False,source_code_sha256=safe.digest(__file__))))
        np.savez_compressed(d/'model.npz',**bestmodel);(d/'model.npz.sha256').write_text(safe.digest(d/'model.npz')+'\n',encoding='ascii');dump(d/'history.json',hist)
        row=dict(name=name,spec=spec,best_step=beststep,validation=exact,model=str(d.relative_to(out)/'model.npz'),model_sha256=safe.digest(d/'model.npz'));rows.append(row)
        dump(out/'ranking.json',sorted(rows,key=lambda z:z['validation']['rmse_T']))
        if best<overallbest:
            overallbest=best;np.savez_compressed(out/'best_validation_predictions.npz',prediction=pred,case_ids=a['case_ids'][va])
    dump(out/'summary.json',dict(rows=rows,best=sorted(rows,key=lambda z:z['validation']['rmse_T'])[0],elapsed_seconds=time.perf_counter()-start,projection_floor=floor,test_labels_accessed=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--region',default='rotor');p.add_argument('--quantity',choices=['vector','magnitude'],default='magnitude');p.add_argument('--out',default=str(Path(__file__).parent/'neural_kernel_route'));q=p.parse_args()
    with threadpool_limits(limits=2):run(q.region,q.out,q.quantity)
if __name__=='__main__':main()
