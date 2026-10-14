"""Train-only joint condition/time spatial kernel controls (not a neural net).

Uses the audited V2 spatial POD, with all output means/scalers/inducing inputs
computed only from training conditions. Test/full60-fit targets never opened.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PREP=HERE.parent/'new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def raw_features(c,cap,rotation,static_weight):
    c=np.atleast_2d(np.asarray(c,dtype=float)); r=c[:,0]/420.
    if cap>0:r=cap/420.*np.tanh(c[:,0]/cap)
    a=np.deg2rad(c[:,1]); phase=np.arange(1,61)/60.*(2*np.pi)
    shifted=a[:,None]+rotation*phase[None,:]
    f=np.stack([r[:,None]*np.cos(shifted),r[:,None]*np.sin(shifted)],axis=-1)
    if static_weight>0:
        s=static_weight*np.stack([r*np.cos(a),r*np.sin(a)],axis=-1)
        f=np.concatenate([f,np.broadcast_to(s[:,None,:],(len(c),60,2))],axis=-1)
    # First recorded cycle, no forced endpoint equality: retain nonperiodic t.
    tf=np.stack([np.cos(phase),np.sin(phase),.5*np.arange(1,61)/60.],axis=-1)
    return f.reshape(-1,f.shape[-1]),np.tile(tf,(len(c),1))

def dist(x,y):
    return np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(y*y,axis=1)[None,:]-2*x@y.T,0.)

def kernel(dc,dt,length_c,length_t,kind):
    r2=dc/(length_c*length_c)+dt/(length_t*length_t)
    if kind=='gaussian':return np.exp(-.5*r2)
    q=np.sqrt(5*r2);return (1+q+q*q/3)*np.exp(-q)

def coeff_metrics(pred,truth,floor,energy,d):
    e=pred-truth; sse=np.sum(e*e,axis=(1,2))+floor
    return dict(rmse_T=float(np.sqrt(sse.sum()/(len(pred)*60*d))),
                relative_L2_pct=float(100*np.sqrt(sse.sum()/energy.sum())),
                worst_case_rmse_T=float(np.sqrt(sse/(60*d)).max()),per_case_rmse_T=np.sqrt(sse/(60*d)).tolist())

def exact_metrics(p,y):
    e=p-y
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(y)),
                mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),
                worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()),
                per_case_rmse_T=np.sqrt(np.mean(e*e,axis=(1,2))).tolist())

def predict(model,conditions,frame_index=None,nonnegative=False):
    if isinstance(model,(str,Path)):
        with np.load(model,allow_pickle=False) as z:model={k:z[k] for k in z.files}
    spec=json.loads(str(model['configuration_json'])); one=np.asarray(conditions).ndim==1
    x,t=raw_features(conditions,spec['cap'],spec['rotation'],spec['static_weight'])
    x=(x-model['input_mean'])/model['input_scale']
    k=kernel(dist(x,model['centers_condition']),dist(t,model['centers_time']),spec['length_c'],spec['length_t'],spec['kernel'])
    c=(k@model['weights']).reshape(-1,60,model['basis'].shape[1])+model['frame_coefficient_mean']
    out=model['spatial_mean']+c@model['basis'].T
    if nonnegative:out=np.maximum(out,0)
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<60:raise ValueError('Expected fixed frame_index 0..59')
        out=out[:,frame_index]
    return out[0] if one else out

def run(output,mode):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    with np.load(PREP/'train_validation_data.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(PREP/'representation.npz',allow_pickle=False) as z:rep={k:z[k] for k in z.files}
    with np.load(PREP/'validation_truth.npz',allow_pickle=False) as z:truth=z['truth']
    q=512;coeff=a['coeff'][...,:q];base=coeff[:42].mean(0);target=(coeff[:42]-base).reshape(-1,q)
    floor=np.maximum(a['centered_energy'][42:]-np.sum(coeff[42:]**2,axis=(1,2)),0)
    frames=np.unique(np.round(np.linspace(0,59,10)).astype(int))
    landmarks=(np.arange(42)[:,None]*60+frames[None,:]).ravel()
    settings=([(0.,0.,0.),(150.,0.,0.),(0.,1.,0.),(0.,-1.,0.)] if mode=='pilot' else
              [(cap,rotation,sw) for cap in (0.,150.,300.) for rotation,sw in [(0.,0.),(1.,0.),(-1.,0.),(1.,.3),(-1.,.3)]])
    lengths=[(.6,.25),(1.2,.5),(2.4,1.),(1.2,1.)] if mode=='pilot' else [(lc,lt) for lc in (.4,.8,1.6,3.2) for lt in (.15,.3,.6,1.2)]
    kinds=['gaussian'] if mode=='pilot' else ['gaussian','matern52']
    ridges=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
    dump(out/'protocol.json',dict(mode=mode,settings=settings,lengths=lengths,kinds=kinds,ridges=ridges,
        source_sha256=digest(__file__),input_sha256={p.name:digest(p) for p in [PREP/'identity.json',PREP/'representation.npz',PREP/'train_validation_data.npz',PREP/'validation_truth.npz']},
        training_samples=2520,training_cases=a['case_ids'][:42].tolist(),validation_cases=a['case_ids'][42:].tolist(),
        train_only_inducing_indices=landmarks.tolist(),test_labels_opened=False,rank=q,
        quantity='planar magnitude hypot(Br,Bt)',selection='raw original-T validation field RMSE, no reused-test labels',
        method='Nyström spatial kernel control; this is not a neural network'))
    best=np.inf;rows=[]
    for setting_index,(cap,rotation,sw) in enumerate(settings):
        x,t=raw_features(a['conditions'],cap,rotation,sw)
        # Do not independently standardize the static multiplier away.
        mu=x[:2520].mean(0);std=np.maximum(x[:2520,:2].std(0),1e-8)
        scale=np.tile(std,x.shape[1]//2);x=(x-mu)/scale
        xc=x[:2520][landmarks];tc=t[:2520][landmarks]
        dc=dist(x[:2520],xc);dt=dist(t[:2520],tc);dcv=dist(x[2520:],xc);dtv=dist(t[2520:],tc)
        dcc=dist(xc,xc);dtt=dist(tc,tc)
        for kind in kinds:
          for lc,lt in lengths:
            spec=dict(cap=cap,rotation=rotation,static_weight=sw,kernel=kind,length_c=lc,length_t=lt)
            kmm=kernel(dcc,dtt,lc,lt,kind)
            km=kernel(dc,dt,lc,lt,kind);kv=kernel(dcv,dtv,lc,lt,kind)
            chol=cholesky(kmm+np.eye(len(kmm))*1e-8,lower=True,check_finite=False)
            phi=solve_triangular(chol,km.T,lower=True,check_finite=False).T
            phiv=solve_triangular(chol,kv.T,lower=True,check_finite=False).T
            eig,vec=eigh(phi.T@phi,check_finite=False);eig=np.maximum(eig,0)
            rhs=vec.T@(phi.T@target);pv=phiv@vec
            for ridge in ridges:
                w=rhs/(eig[:,None]+ridge*eig[-1])
                val=(pv@w).reshape(9,60,q)+base
                metrics=coeff_metrics(val,coeff[42:],floor,a['truth_energy'][42:],truth.shape[-1])
                row=dict(candidate=len(rows),configuration=dict(spec,ridge=ridge),validation=metrics)
                rows.append(row)
                if metrics['rmse_T']<best:
                    raw=rep['spatial_mean']+val@rep['spatial_basis'][:,:q].T
                    exact=exact_metrics(raw,truth)
                    if not np.isclose(exact['rmse_T'],metrics['rmse_T'],rtol=1e-8,atol=1e-9):raise AssertionError('Coefficient/field mismatch')
                    row['validation']=exact;row['validation_nonnegative']=exact_metrics(np.maximum(raw,0),truth)
                    row['raw_negative_count']=int(np.sum(raw<0))
                    best=exact['rmse_T']; row['elapsed_seconds']=time.perf_counter()-start
                    weights=solve_triangular(chol.T,vec@w,lower=False,check_finite=False)
                    model=dict(configuration_json=np.array(json.dumps(dict(spec,ridge=ridge))),centers_condition=xc,centers_time=tc,
                        input_mean=mu,input_scale=scale,weights=weights,spatial_mean=rep['spatial_mean'],basis=rep['spatial_basis'][:,:q],
                        frame_coefficient_mean=base,point_id=rep['point_id'],xy=rep['xy'],theta=rep['theta'],time_s=rep['time_s'])
                    replay=predict(model,a['conditions'][42:])
                    if not np.allclose(replay,raw,rtol=2e-9,atol=2e-9):raise AssertionError('Model replay differs')
                    np.savez_compressed(out/'best_model.npz',**model)
                    np.savez_compressed(out/'best_validation_predictions.npz',prediction=raw,case_ids=a['case_ids'][42:],quantity=np.array('|B|'))
                    dump(out/'best.json',row)
                    print(json.dumps(dict(best_candidate=row['candidate'],configuration=row['configuration'],validation=row['validation'])),flush=True)
            dump(out/'ranking.json',sorted(rows,key=lambda z:z['validation']['rmse_T']))
        print(json.dumps(dict(setting_done=setting_index,completed=len(rows),best_rmse_T=best,elapsed=time.perf_counter()-start)),flush=True)
    summary=dict(completed=len(rows),best=min(rows,key=lambda z:z['validation']['rmse_T']),elapsed_seconds=time.perf_counter()-start,
        model_sha256=digest(out/'best_model.npz'),source_sha256=digest(__file__),test_labels_opened=False)
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['pilot','full'],default='pilot');a=p.parse_args()
    with threadpool_limits(limits=2):run(a.output,a.mode)
