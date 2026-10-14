"""Shared train-only point response operator: 2 finite true-neural pilots.

Fixed sampled-point IDs carry train42-derived low-dimensional embeddings.
Shared MLP maps native condition features and point features to mean signed
Br/Bt response; frozen V2 signed neural dynamics provide supplied 60 frames.
Inference imports NumPy only; PyTorch is imported exclusively while training.
"""
from pathlib import Path
import argparse,json,time,sys
import numpy as np
from threadpoolctl import threadpool_limits
import rational_control as r

ROOT=Path(__file__).resolve().parent;V1=r.V1;V2=r.V2


def features(conditions,scale):
    c=np.atleast_2d(np.asarray(conditions,dtype=np.float64));rr=c[:,0]/scale;a=np.deg2rad(c[:,1]);x=rr*np.cos(a);y=rr*np.sin(a)
    return np.column_stack((rr,c[:,1]/90,x,y,x*x,x*y,y*y))


def signed_mean(model,conditions):
    """Shared feed-forward equations; float32 matches trained weights."""
    m=model;c=np.atleast_2d(conditions)
    cf=((features(c,float(m['current_scale']))-m['condition_center'])/m['condition_scale']).astype(np.float32)
    attr=np.concatenate((m['point_embedding'],m['point_attributes']),axis=1)
    inp=np.concatenate((np.broadcast_to(cf[:,None,:],(len(c),len(attr),cf.shape[-1])),
                        np.broadcast_to(attr[None,:,:],(len(c),*attr.shape))),axis=-1).reshape(-1,cf.shape[-1]+attr.shape[-1])
    h=inp
    for i in range(int(m['layer_count'])):
        h=h@m[f'w{i}']+m[f'b{i}']
        if i+1<int(m['layer_count']):h=np.tanh(h)
    out=m['zero_current_baseline'][None,:,:]+h.reshape(len(c),len(attr),2)*m['point_response_scale'][None,:,:]
    return out.reshape(len(c),-1).astype(np.float64)


def load_model(path):
    path=Path(path)
    if r.safe.digest(path)!=path.with_suffix('.npz.sha256').read_text().strip():raise ValueError('NN model hash mismatch')
    with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    return m


def predict(model,conditions,frame_index=None,return_signed=False):
    if isinstance(model,(str,Path)):
        path=Path(model);m=load_model(path)
        nnpath=(path.parent/str(m['dynamic_neural_relative_path'])).resolve()
        if r.safe.digest(nnpath)!=str(m['dynamic_neural_sha256']):raise ValueError('Frozen dynamics identity mismatch')
    else:m=model;nnpath=V2/'neural_route/rotor/selected_model.npz'
    c=np.asarray(conditions);single=c.ndim==1;c=np.atleast_2d(c)
    raw=r.safe.predict(nnpath,c);out=signed_mean(m,c)[:,None,:]+raw-raw.mean(1,keepdims=True)
    if not return_signed:out=np.hypot(out[...,0::2],out[...,1::2])
    if frame_index is not None:out=out[:,frame_index]
    return out[0] if single else out


def prepare():
    split=r.safe.read_json(V1/'cache_v1/split.json');tr=split['train'];va=split['validation']
    with np.load(V1/'cache_v1/rotor.npz',allow_pickle=False) as z:
        c=z['conditions'];pid=z['point_id'];xy=z['xy'];ids=z['case_ids']
    _,mean=r.mean_rows(V1/'cache_v1/rotor.npz',tr);mean=mean.reshape(42,1800,2)
    row0=[j for j,i in enumerate(tr) if c[i,0]==0]
    if len(row0)!=1:raise ValueError('Exactly one training zero-current reference required')
    zero=mean[row0[0]];response=mean-zero[None,:,:];ps=np.maximum(response.std(0),.005)
    pointcurve=(response/ps).transpose(1,0,2).reshape(1800,84)
    center=pointcurve.mean(0);u,s,vt=np.linalg.svd(pointcurve-center,full_matrices=False)
    raw=(pointcurve-center)@vt[:8].T
    em=raw.mean(0);es=np.maximum(raw.std(0),1e-6);embedding=((raw-em)/es).astype(np.float32)
    xym=xy.mean(0);xys=np.maximum(xy.std(0),1e-6)
    bm=zero.mean(0);bs=np.maximum(zero.std(0),.005);lg=np.log(ps);lm=lg.mean(0);ls=np.maximum(lg.std(0),1e-6)
    attributes=np.column_stack(((xy-xym)/xys,(zero-bm)/bs,(lg-lm)/ls)).astype(np.float32)
    current_scale=max(float(c[tr,0].max()),1.);f=features(c,current_scale);fm=f[tr].mean(0);fs=np.maximum(f[tr].std(0),1e-8)
    cf=((f-fm)/fs).astype(np.float32)
    valfield=r.safe.selected_rows(V1/'cache_v1/rotor.npz','field',va).astype(np.float64)
    truth=np.hypot(valfield[...,0::2],valfield[...,1::2]);valmean=valfield.mean(1).reshape(9,1800,2);del valfield
    nnpath=V2/'neural_route/rotor/selected_model.npz';nn=r.safe.predict(nnpath,c[va]);dynamic=nn-nn.mean(1,keepdims=True)
    arrays=dict(point_embedding=embedding,point_attributes=attributes,point_response_scale=ps.astype(np.float32),
        zero_current_baseline=zero.astype(np.float32),condition_center=fm,condition_scale=fs,current_scale=np.array(current_scale),
        embedding_pca_center=center,embedding_pca_basis=vt[:8],embedding_pca_mean=em,embedding_pca_scale=es,
        point_id=pid,xy=xy,dynamic_neural_relative_path=np.array('../../../new60_opt_v2/neural_route/rotor/selected_model.npz'),
        dynamic_neural_sha256=np.array(r.safe.digest(nnpath)))
    info=dict(train_case_ids=ids[tr].tolist(),validation_case_ids=ids[va].tolist(),excluded_test_case_ids=ids[split['test']].tolist(),
        embedding_definition='PCA8 across1800 point response curves formed only from42 train signed temporal mean responses, after per-point response std normalization',
        embedding_singular_values=s.tolist(),embedding_rank=8,zero_reference_train_case_id=int(ids[tr[row0[0]]]),
        scale_fit='All condition scaling, signed baseline, point response scales and embedding use train42 only; fixed geometry standardized over original1800 sampled points',
        source_sha256=r.safe.digest(__file__),mean_helper_sha256=r.safe.digest(r.__file__),cache_sha256=r.safe.digest(V1/'cache_v1/rotor.npz'),
        dynamic_model_sha256=r.safe.digest(nnpath),test_labels_materialized=False,full60_fitting_used=False,
        inputs_note='Native I_RMS/Thet mathematical polynomial/polar features, no claimed dq angle',
        model_scope='Fixed original1800 points and supplied60 frames, no claim to unsampled mesh/time; point embeddings carry only train labels')
    return arrays,info,c,tr,va,mean,cf,truth,valmean,dynamic


def train(out,prepared,config):
    import torch
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    arrays,identity,c,tr,va,mean,cf,truth,valmean,dynamic=prepared
    out=Path(out);out.mkdir(exist_ok=False)
    t0=time.perf_counter();target=torch.tensor(mean.astype(np.float32));scale=torch.tensor(arrays['point_response_scale']);base=torch.tensor(arrays['zero_current_baseline'])
    ct=torch.tensor(cf[tr]);cv=torch.tensor(cf[va]);attributes=torch.tensor(arrays['point_attributes']);embedding=torch.nn.Embedding(1800,8)
    embedding.weight.data.copy_(torch.tensor(arrays['point_embedding']));embedding.weight.requires_grad_(config['trainable_embedding'])
    net=torch.nn.Sequential(torch.nn.Linear(21,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,2))
    for layer in net:
        if isinstance(layer,torch.nn.Linear):torch.nn.init.xavier_normal_(layer.weight);torch.nn.init.zeros_(layer.bias)
    net[-1].weight.data.zero_();net[-1].bias.data.zero_()
    parameters=list(net.parameters())+(list(embedding.parameters()) if config['trainable_embedding'] else [])
    optimizer=torch.optim.Adam(parameters,lr=config['learning_rate'])
    rng=np.random.default_rng(config['seed']);history=[];best=np.inf;best_step=0;best_state=None
    nparam=sum(p.numel() for p in parameters if p.requires_grad)
    def full_mean(condition):
        pp=torch.arange(1800);attr=torch.cat((embedding(pp),attributes),dim=1)
        inp=torch.cat((condition[:,None,:].expand(-1,1800,-1),attr[None,:,:].expand(len(condition),-1,-1)),dim=-1)
        return base+net(inp)*scale
    def checkpoint(step,loss):
        nonlocal best,best_step,best_state
        with torch.no_grad():
            trainpred=full_mean(ct).cpu().numpy().astype(np.float64);predmean=full_mean(cv).cpu().numpy().astype(np.float64)
        field=predmean.reshape(9,1,3600)+dynamic;pred=np.hypot(field[...,0::2],field[...,1::2])
        vm=r.metrics(pred,truth);tm=r.metrics(trainpred,mean);vsm=r.metrics(predmean,valmean)
        row=dict(step=step,loss=float(loss),train_signed_mean_rmse_T=tm['rmse_T'],
            val_signed_mean_rmse_T=vsm['rmse_T'],validation_rmse_T=vm['rmse_T'],validation_relative_L2_pct=vm['relative_L2_pct'],
            validation_mae_T=vm['mae_T'],validation_max_abs_T=vm['max_abs_T'],elapsed_seconds=time.perf_counter()-t0)
        history.append(row)
        if vm['rmse_T']<best:
            best=vm['rmse_T'];best_step=step;best_state=({k:v.detach().clone() for k,v in net.state_dict().items()},embedding.weight.detach().clone())
            print('BEST',out.name,json.dumps(row),flush=True)
        return row
    r.dump(out/'config.json',config);checkpoint(0,0)
    denominator=float(np.mean(mean*mean))
    for step in range(1,config['steps']+1):
        j=torch.tensor(rng.integers(0,42,size=config['batch_size']),dtype=torch.long);p=torch.tensor(rng.integers(0,1800,size=config['batch_size']),dtype=torch.long)
        inp=torch.cat((ct[j],embedding(p),attributes[p]),dim=1);pred=base[p]+net(inp)*scale[p]
        loss=torch.mean((pred-target[j,p])**2)/denominator
        # Embedding penalty prevents trainable point codes from escaping initial labels.
        if config['trainable_embedding']:loss=loss+config['embedding_regularization']*torch.mean((embedding.weight-torch.tensor(arrays['point_embedding']))**2)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,10.);optimizer.step()
        if step%100==0:
            checkpoint(step,loss.detach());r.dump(out/'history.json',history)
            if step>=config['min_steps'] and step-best_step>=config['patience']:break
    net.load_state_dict(best_state[0]);embedding.weight.data.copy_(best_state[1])
    saved={k:np.array(v,copy=True) for k,v in arrays.items()};saved['point_embedding']=embedding.weight.detach().numpy().copy();linear=[z for z in net if isinstance(z,torch.nn.Linear)]
    saved['layer_count']=np.array(len(linear));saved['config_json']=np.array(json.dumps(config));saved['training_identity_json']=np.array(json.dumps(identity))
    for i,l in enumerate(linear):saved[f'w{i}']=l.weight.detach().numpy().T.copy();saved[f'b{i}']=l.bias.detach().numpy().copy()
    np.savez_compressed(out/'model.npz',**saved);(out/'model.npz.sha256').write_text(r.safe.digest(out/'model.npz')+'\n',encoding='ascii')
    deployed=predict(out/'model.npz',c[va]);met=r.metrics(deployed,truth)
    with torch.no_grad():torchmean=full_mean(cv).numpy().astype(np.float64)
    npmean=signed_mean(saved,c[va]).reshape(9,1800,2);parity=float(np.max(np.abs(torchmean-npmean)))
    if parity>5e-5:raise AssertionError('Float32 replay mismatch '+str(parity))
    met.update(per_case={str(cid):r.metrics(deployed[i],truth[i]) for i,cid in enumerate(identity['validation_case_ids'])})
    met['worst_case_rmse_T']=max(v['rmse_T'] for v in met['per_case'].values())
    np.savez_compressed(out/'validation_predictions.npz',prediction=deployed,mean_signed_prediction=npmean,case_ids=np.array(identity['validation_case_ids']),point_id=saved['point_id'])
    r.dump(out/'history.json',history);r.dump(out/'summary.json',dict(config=config,selected_step=best_step,executed_steps=step,
        validation=met,trainable_parameters=nparam,float32_torch_numpy_max_signed_mean_difference_T=parity,
        model_sha256=r.safe.digest(out/'model.npz'),elapsed_seconds=time.perf_counter()-t0,test_labels_materialized=False,full60_fitting_used=False))
    return r.safe.read_json(out/'summary.json')


def run():
    out=ROOT/'point_operator_nn'
    if out.exists():raise FileExistsError(out)
    out.mkdir();prepared=prepare();r.dump(out/'preparation_identity.json',prepared[1])
    configs=[dict(seed=431,trainable_embedding=flag,embedding_regularization=.0001,hidden=[128,128,128],activation='tanh',
        learning_rate=.0015,batch_size=4096,steps=5000,min_steps=1200,patience=1000,embedding_rank=8) for flag in (False,True)]
    r.dump(out/'protocol.json',dict(configs=configs,selection='Original-T complete val9 x60x1800 planar magnitude RMSE; no test/full60fit labels used',
        training_objective='Original-T train42 signed temporal mean MSE through shared point operator',
        quantity='Shared point NN signed mean + frozen pureNN signed zero-time-mean dynamics, then hypot',
        trainable_embedding_note='When enabled1800x8 point codes are tuned only by train signed-mean loss, all points share MLP weights',
        cache_hash=prepared[1]['cache_sha256'],source_sha256=r.safe.digest(__file__)))
    results=[]
    for i,conf in enumerate(configs):
        results.append(train(out/f'pilot_{i+1:02d}',prepared,conf));r.dump(out/'running_results.json',results)
    r.dump(out/'results.json',dict(completed_training_runs=len(results),results=results,best_index=int(np.argmin([x['validation']['rmse_T'] for x in results])),
        test_labels_materialized=False,full60_fitting_used=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
