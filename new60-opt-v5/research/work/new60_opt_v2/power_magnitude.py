"""Nonlinear output-coordinate controls: interpolate B^p, reconstruct planar B.

Only training/validation signed-field rows are materialized. Nonnegative
reconstruction is part of the actual model; every real sampled point is scored.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;V1=ROOT.parent/'new60_20261014'
sys.path.insert(0,str(V1/'code'));sys.path.insert(0,str(ROOT/'kernel_route'))
import hybrid_cycle_rbf as safe
import kernel_model as km

def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def predict(model,conditions,frame_index=None):
    if isinstance(model,(str,Path)):
        with np.load(model,allow_pickle=False) as z:model={k:z[k] for k in z.files}
    power=float(model['output_power']);raw=km.predict(model,conditions,frame_index)
    return np.maximum(raw,0)**(1/power)

def run(region,out):
    out=Path(out)/region
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.perf_counter()
    a,s,meta=safe._load_cache(V1/'cache_v1'/f'{region}.npz',V1/'cache_v1/split.json')
    tr,va=np.asarray(s['train']),np.asarray(s['validation']);traincond=a['conditions'][tr];valcond=a['conditions'][va]
    yt=np.linalg.norm(np.asarray(a['field'][tr],dtype=np.float64).reshape(42,60,-1,2),axis=-1)
    yv=np.linalg.norm(np.asarray(a['field'][va],dtype=np.float64).reshape(9,60,-1,2),axis=-1)
    current_scale=float(np.median(traincond[:,0][traincond[:,0]>0]));rows=[];best=np.inf;best_model=None
    coordinates=('polar_xy','saturated_1','saturated_2')
    configs=[dict(coordinate=coord,anisotropy=anis,kernel=kind,lengthscale=ell,ridge=ridge,degree=degree,current_scale=current_scale)
      for coord in coordinates for anis in (1.,2.) for kind in ('gaussian','matern52') for ell in (.6,1.2,2.4,4.8)
      for ridge in (1e-6,.001,.01) for degree in (0,1,2)]
    dump(out/'protocol.json',dict(powers=[.5,1.,2.,3.],candidates_per_power=len(configs),configs=configs,
      quantity='planar_magnitude',split=s,nonnegative_policy='max(predicted_B_power,0)**(1/power) is deployed and evaluated, no points removed',
      train_only_basis=True,test_labels_accessed=False,source_code_sha256=safe.digest(__file__),kernel_source_sha256=safe.digest(ROOT/'kernel_route/kernel_model.py')))
    for power in (.5,1.,2.,3.):
      transformed=yt.reshape(42,-1)**power;mu=transformed.mean(0);centered=transformed-mu
      eig,u=np.linalg.eigh(centered@centered.T);order=np.argsort(eig)[::-1];eig=eig[order][:41];u=u[:,order][:,:41]
      good=eig>max(eig[0]*1e-14,1e-18);eig=eig[good];u=u[:,good]
      basis=centered.T@u/np.sqrt(eig)[None,:];target=u*np.sqrt(eig)[None,:]
      for index,spec in enumerate(configs):
        trainraw=km.coordinates(traincond,spec['coordinate'],current_scale);c=trainraw.mean(0);scale=np.maximum(trainraw.std(0),1e-8)
        x=km.transformed(traincond,spec['coordinate'],current_scale,c,scale,spec['anisotropy'])
        xv=km.transformed(valcond,spec['coordinate'],current_scale,c,scale,spec['anisotropy'])
        try: weights,trend=km.solve_map(x,target,spec['kernel'],spec['lengthscale'],spec['ridge'],spec['degree'])
        except np.linalg.LinAlgError:continue
        coeff=km.apply_map(xv,x,weights,trend,spec['kernel'],spec['lengthscale'],spec['degree'])
        raw=(mu+coeff@basis.T).reshape(yv.shape);pred=np.maximum(raw,0)**(1/power)
        metrics=safe._metrics(pred,yv); per={str(cid):safe._metrics(pred[i],yv[i]) for i,cid in enumerate(a['case_ids'][va])}
        row=dict(power=power,spec=spec,validation=metrics,worst_case_rmse_T=max(z['rmse_T'] for z in per.values()),negative_transformed_count=int(np.count_nonzero(raw<0)))
        rows.append(row)
        if metrics['rmse_T']<best:
            best=metrics['rmse_T'];bestrow=row
            best_model=dict(b0_center=c,b0_input_scale=scale,b0_centers=x,b0_weights=weights,b0_trend=trend,b0_mean=mu,b0_basis=basis,b0_indices=np.arange(mu.size),output_power=np.array(power),metadata_json=np.array(json.dumps(dict(format='new60_condition_kernel_v2',region=region,quantity='planar_magnitude',time_count=60,field_dim=yt.shape[-1],blocks=[spec],training_original_ids=a['case_ids'][tr].tolist(),validation_original_ids=a['case_ids'][va].tolist(),source_code_sha256=safe.digest(__file__),cache_sha256=safe.digest(V1/'cache_v1'/f'{region}.npz'),test_labels_accessed=False,nonnegative_output=True,output_transform='B_power'))))
            print('NEW BEST',region,power,index,metrics,flush=True)
        if index%100==0:dump(out/'running_ranking.json',sorted(rows,key=lambda z:z['validation']['rmse_T'])[:40])
      print('POWER DONE',region,power,'best',bestrow,flush=True)
    np.savez_compressed(out/'model.npz',**best_model);(out/'model.npz.sha256').write_text(safe.digest(out/'model.npz')+'\n',encoding='ascii')
    pred=predict(best_model,valcond);np.savez_compressed(out/'validation_predictions.npz',prediction=pred,case_ids=a['case_ids'][va])
    dump(out/'ranking.json',sorted(rows,key=lambda z:z['validation']['rmse_T']))
    dump(out/'summary.json',dict(best=bestrow,model_sha256=safe.digest(out/'model.npz'),validation_replay=safe._metrics(pred,yv),elapsed_seconds=time.perf_counter()-start,test_labels_accessed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--region',default='rotor');p.add_argument('--out',default=str(ROOT/'power_route'));q=p.parse_args()
    with threadpool_limits(limits=2):run(q.region,q.out)
