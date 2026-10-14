"""Train-only LOO-selected output-adaptive kernels with a POD neural control.

Every input-dependent coefficient can have a different smooth interpolant.
Cache field/coeff test rows are skipped by the V1 audited stream reader.
Model choices are based only on the original nine validation conditions.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

V1=Path(__file__).resolve().parent.parent/'new60_20261014'
sys.path.insert(0,str(V1/'code'))
import hybrid_cycle_rbf as safe
import experiment_core as core

def dump(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def coordinates(x,kind):
    x=np.asarray(x,dtype=np.float64)
    if kind=='native': return x
    a=np.deg2rad(x[:,1]); cur=x[:,0]
    if kind.startswith('saturated_'):
        cap=float(kind.split('_')[1]); cur=cap*np.tanh(cur/cap)
        return np.stack((cur*np.cos(a),cur*np.sin(a)),axis=1)
    if kind=='cartesian': return np.stack((cur*np.cos(a),cur*np.sin(a)),axis=1)
    if kind=='half_cartesian': return np.stack((cur*np.cos(a/2),cur*np.sin(a/2)),axis=1)
    raise ValueError(kind)

def polynomial(x,degree):
    return np.stack([x[:,0]**i*x[:,1]**(n-i) for n in range(degree+1) for i in range(n+1)],axis=1)

def kernel(x,c,spec):
    l,ratio,ridge,kind,degree=spec
    d=((x[:,None,:]-c[None,:,:])/np.array([l*np.sqrt(ratio),l/np.sqrt(ratio)]))
    rr=np.einsum('ijk,ijk->ij',d,d)
    if kind=='gaussian': return np.exp(-.5*rr)
    r=np.sqrt(np.maximum(rr,1e-24))
    if kind=='matern52':
        q=np.sqrt(5)*r; return (1+q+q*q/3)*np.exp(-q)
    if kind=='matern32':
        q=np.sqrt(3)*r; return (1+q)*np.exp(-q)
    raise ValueError(kind)

def predict(model,x,base=None):
    x=coordinates(np.atleast_2d(x),str(model['coordinate']))
    x=(x-model['input_mean'])/model['input_scale']
    pred=np.zeros((len(x),model['weights'].shape[1]),dtype=np.float64)
    for k,ss in enumerate(model['specs']):
        spec=json.loads(str(ss)); cols=np.flatnonzero(model['choice']==k)
        if not len(cols): continue
        phi=np.concatenate((kernel(x,model['centers'],spec),polynomial(x,int(spec[-1]))),axis=1)
        pred[:,cols]=phi@model['weights'][:phi.shape[1],cols]
    pred=pred.reshape(len(x),int(model['time_count']),model['basis'].shape[1])
    if base is not None: pred+=base
    return model['mean']+pred@model['basis'].T

def candidate_specs():
    for kind in ('gaussian','matern52','matern32'):
      for degree in (0,1,2):
       for l in (.3,.5,.8,1.2,1.8,2.7,4.,6.,10.,16.):
        for ratio in (.5,1.,2.):
         for ridge in (1e-8,1e-6,1e-4,1e-2):
          yield [l,ratio,ridge,kind,degree]

def run(region,out,limited=False,interior=False,selected_coordinates=None):
    out=Path(out)/region
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.perf_counter()
    a,s,meta=safe._load_cache(V1/'cache_v1'/f'{region}.npz',V1/'cache_v1/split.json')
    tr,va=np.array(s['train']),np.array(s['validation'])
    y=np.asarray(a['field'][va],dtype=np.float64)
    trc=np.asarray(a['coeff'][tr],dtype=np.float64).reshape(42,-1)
    vac=np.asarray(a['coeff'][va],dtype=np.float64).reshape(9,-1)
    r=a['basis'].shape[1]; t=len(a['theta']); output_dim=t*r
    specs=list(candidate_specs())
    if limited: specs=specs[::8]
    curmedian=float(np.median(a['conditions'][tr,0])); coords=selected_coordinates or ['cartesian','native','half_cartesian']
    coords=[f'saturated_{curmedian*float(c.split(":")[1]):.12g}' if c.startswith('saturated:') else c for c in coords]
    protocol=dict(region=region,split=s,coordinates=coords,training_current_median=curmedian,
      kernel_spec_count=len(specs),selection_data='train LOO coefficient losses; route choices validation full-field RMSE',
      groups=['spatial_mode','time_mode'],base=['none','V1_train_only_uniform_NN'],test_labels_accessed=False,
      conditional_CV_basis='POD remains the fixed train42-only basis; LOO is conditional diagnostic, not unbiased CV',
      loo_interior_only=interior,source_cache_sha256=safe.digest(V1/'cache_v1'/f'{region}.npz'),source_code_sha256=safe.digest(__file__))
    dump(out/'protocol.json',protocol)
    mlpfile=V1/'experiments'/region/'stage2/uniform_field_control_seed42/best_model.npz'
    if not mlpfile.exists():
        raise FileNotFoundError(mlpfile)
    ntr=core.predict(mlpfile,a['features'][tr],return_coeff=True).reshape(42,-1)
    nva=core.predict(mlpfile,a['features'][va],return_coeff=True).reshape(9,-1)
    rows=[]; best=float('inf'); best_model=None; best_prediction=None
    for coordinate in protocol['coordinates']:
      raw=coordinates(a['conditions'],coordinate)
      mu=raw[tr].mean(0); scale=raw[tr].std(0); scale=np.maximum(scale,1e-9)
      x=(raw[tr]-mu)/scale; v=(raw[va]-mu)/scale
      loo_mask=np.ones(42,dtype=bool)
      if interior:
        from scipy.spatial import ConvexHull
        loo_mask[ConvexHull(x).vertices]=False
      dump(out/f'loo_mask_{coordinate}.json',dict(included_train_case_ids=a['case_ids'][tr[loo_mask]].tolist(),excluded_input_hull_train_case_ids=a['case_ids'][tr[~loo_mask]].tolist()))
      bases={'none':(np.zeros_like(trc),np.zeros_like(vac)),'V1_train_only_uniform_NN':(ntr,nva)}
      states={}
      max_tail=polynomial(x,2).shape[1]
      for b,(base_tr,base_va) in bases.items():
       for group in protocol['groups']:
        size=r if group=='spatial_mode' else output_dim
        states[b,group]=dict(score=np.full(size,np.inf),choice=np.full(size,-1,dtype=np.int32),
          weights=np.zeros((42+max_tail,output_dim)),prediction=np.zeros((9,output_dim)),residual=trc-base_tr)
      trace=[]
      for ki,spec in enumerate(specs):
        degree=int(spec[-1]); p=polynomial(x,degree); pv=polynomial(v,degree)
        k=kernel(x,x,spec)+np.eye(42)*spec[2]
        aug=np.block([[k,p],[p.T,np.zeros((p.shape[1],p.shape[1]))]])
        try: inv=np.linalg.inv(aug)
        except np.linalg.LinAlgError: continue
        if not np.isfinite(inv).all() or np.min(np.abs(np.diag(inv)[:42]))<1e-15: continue
        phi=np.concatenate((kernel(v,x,spec),pv),axis=1)
        for b,(base_tr,base_va) in bases.items():
         state=states[b,'time_mode']; weights=inv[:,:42]@state['residual']
         loo=(weights[:42]/np.diag(inv)[:42,None])**2
         per_time=loo[loo_mask].mean(0)
         for group in protocol['groups']:
          st=states[b,group]
          score=per_time.reshape(t,r).mean(0) if group=='spatial_mode' else per_time
          improve=score<st['score']
          if not improve.any(): continue
          cols=np.flatnonzero(np.tile(improve,t)) if group=='spatial_mode' else np.flatnonzero(improve)
          st['score'][improve]=score[improve]; st['choice'][improve]=ki
          st['weights'][:,cols]=0; st['weights'][:len(inv),cols]=weights[:,cols]
          st['prediction'][:,cols]=phi@weights[:,cols]
        if (ki+1)%120==0 or ki+1==len(specs):
          info={}
          for (b,g),st in states.items():
            pred=(st['prediction']+bases[b][1]).reshape(9,t,r)
            # Uniform orthogonal-field proxy used for trace, final values replay real fields below.
            diff=pred-vac.reshape(9,t,r)
            info[f'{b}/{g}']=float(np.sqrt(np.mean(np.sum(diff*diff,axis=2))/a['basis'].shape[0]))
          trace.append(dict(candidate=ki+1,coefficient_field_error_proxy_T=info))
          print(region,coordinate,ki+1,info,flush=True)
      dump(out/f'trace_{coordinate}.json',trace)
      for (b,group),st in states.items():
        coeff=(st['prediction']+bases[b][1]).reshape(9,t,r)
        pred=a['mean']+coeff@np.asarray(a['basis'],dtype=np.float64).T
        metrics=safe._metrics(pred,y)
        per={str(cid):safe._metrics(pred[i],y[i]) for i,cid in enumerate(a['case_ids'][va])}
        row=dict(name=f'{coordinate}_{b}_{group}',coordinate=coordinate,base=b,group=group,
          metrics=metrics,per_case=per,unique_kernels=int(len(np.unique(st['choice']))),
          worst_case_rmse_T=max(z['rmse_T'] for z in per.values()),test_labels_accessed=False)
        rows.append(row); dump(out/'validation_ranking.json',sorted(rows,key=lambda z:z['metrics']['rmse_T']))
        print('VALIDATION',region,row['name'],metrics,flush=True)
        if metrics['rmse_T']<best:
          best=metrics['rmse_T']; choice=np.tile(st['choice'],t) if group=='spatial_mode' else st['choice']
          best_model=dict(format=np.array('new60_adaptive_spectral_kernel_v2'),coordinate=np.array(coordinate),
            base=np.array(b),input_mean=mu,input_scale=scale,centers=x,choice=choice,
            specs=np.array([json.dumps(v) for v in specs]),weights=st['weights'].copy(),basis=a['basis'],
            mean=a['mean'],time_count=np.array(t),point_id=a['point_id'],xy=a['xy'],theta=a['theta'],
            metadata_json=np.array(json.dumps(dict(protocol=protocol,best_validation=row,base_model_sha256=safe.digest(mlpfile) if b!='none' else None))))
          best_prediction=pred.copy()
    np.savez_compressed(out/'model.npz',**best_model)
    (out/'model.npz.sha256').write_text(safe.digest(out/'model.npz')+'\n',encoding='ascii')
    np.savez_compressed(out/'validation_predictions.npz',prediction=best_prediction,case_ids=a['case_ids'][va])
    dump(out/'summary.json',dict(best=sorted(rows,key=lambda z:z['metrics']['rmse_T'])[0],rows=rows,
        elapsed_seconds=time.perf_counter()-started,source_code_sha256=safe.digest(__file__),test_labels_accessed=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--region',default='rotor');p.add_argument('--out',default=str(Path(__file__).parent/'adaptive_route'));p.add_argument('--limited',action='store_true');p.add_argument('--loo-interior',action='store_true');p.add_argument('--coordinates',nargs='+');args=p.parse_args()
    with threadpool_limits(limits=2): run(args.region,args.out,args.limited,args.loo_interior,args.coordinates)
if __name__=='__main__': main()
