"""48 local condition RBF signed-mean controls, train-only label access.

Signed Br/Bt temporal means + frozen pureNN zero-mean signed fluctuations,
then magnitude. No oracle validation time means enter deployed predictions.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.interpolate import RBFInterpolator
from threadpoolctl import threadpool_limits
import rational_control as r

ROOT=r.ROOT;V1=r.V1;V2=r.V2


def coordinates(c,kind):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));cur=c[:,0];a=np.deg2rad(c[:,1])
    if kind=='native':return c
    if kind=='sat150':cur=150*np.tanh(cur/150)
    return np.column_stack((cur*np.cos(a),cur*np.sin(a)))


def radial(x,y,kind,epsilon):
    rr=np.sqrt(np.maximum(np.sum((x[:,None,:]-y[None,:,:])**2,axis=-1),0))*epsilon
    if kind=='cubic':return rr**3
    if kind=='thin_plate_spline':return rr*rr*np.log(np.maximum(rr,1e-300))
    if kind=='gaussian':return np.exp(-rr*rr)
    if kind=='inverse_multiquadric':return 1/np.sqrt(1+rr*rr)
    raise ValueError(kind)


def numpy_means(model,conditions):
    spec=json.loads(str(model['spec_json']));x=(coordinates(conditions,spec['coordinate'])-model['input_center'])/model['input_scale']
    centers=model['centers'];neighbors=np.argsort(np.sum((x[:,None,:]-centers[None,:,:])**2,axis=-1),axis=1)[:,:spec['neighbors']]
    out=[]
    for xi,nb in zip(x,neighbors):
        zz=centers[nb];p=np.column_stack((np.ones(len(nb)),zz));q=np.r_[1.,xi]
        lhs=np.block([[radial(zz,zz,spec['kernel'],spec['epsilon'])+spec['smoothing']*np.eye(len(nb)),p],
                      [p.T,np.zeros((3,3))]])
        coef=np.linalg.solve(lhs,np.concatenate((model['signed_mean_targets'][nb],np.zeros((3,3600))),axis=0))
        out.append(radial(xi[None,:],zz,spec['kernel'],spec['epsilon'])@coef[:len(nb)]+q@coef[len(nb):])
    return np.asarray(out).reshape(len(x),3600)


def predict(model,conditions,frame_index=None):
    if isinstance(model,(str,Path)):
        path=Path(model)
        with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
        nnpath=(path.parent/str(m['neural_relative_path'])).resolve()
        if r.safe.digest(nnpath)!=str(m['neural_sha256']):raise ValueError('Neural model identity mismatch')
    else:m=model;nnpath=V2/'neural_route/rotor/selected_model.npz'
    c=np.asarray(conditions);single=c.ndim==1;c=np.atleast_2d(c)
    raw=r.safe.predict(nnpath,c);field=numpy_means(m,c)[:,None,:]+raw-raw.mean(1,keepdims=True)
    out=np.hypot(field[...,0::2],field[...,1::2])
    if frame_index is not None:out=out[:,frame_index]
    return out[0] if single else out


def run():
    out=ROOT/'local_mean_rbf_control'
    if out.exists():raise FileExistsError(out)
    out.mkdir();start=time.perf_counter();s=r.safe.read_json(V1/'cache_v1/split.json');tr=s['train'];va=s['validation']
    with np.load(V1/'cache_v1/rotor.npz',allow_pickle=False) as z:c=z['conditions'];ids=z['case_ids'];pid=z['point_id'];xy=z['xy']
    _,yt=r.mean_rows(V1/'cache_v1/rotor.npz',tr)
    tv=r.safe.selected_rows(V1/'cache_v1/rotor.npz','field',va).astype(np.float64);truth=np.hypot(tv[...,0::2],tv[...,1::2])
    true_signed_mean=tv.mean(1);del tv
    nnpath=V2/'neural_route/rotor/selected_model.npz';raw=r.safe.predict(nnpath,c[va]);dynamic=raw-raw.mean(1,keepdims=True)
    configs=[dict(coordinate=coord,neighbors=n,kernel=kernel,smoothing=(0.,.001)[(i+j+k)%2],epsilon=1.,degree=1)
        for i,coord in enumerate(('cart','sat150','native')) for j,n in enumerate((8,14,22,42))
        for k,kernel in enumerate(('cubic','thin_plate_spline','gaussian','inverse_multiquadric'))]
    r.dump(out/'protocol.json',dict(configs=configs,candidate_count=len(configs),train_case_ids=ids[tr].tolist(),validation_case_ids=ids[va].tolist(),
        excluded_test_case_ids=ids[s['test']].tolist(),test_labels_materialized=False,full60_fitting_used=False,
        normalization='train42 coordinate mean/std only; current+nativeThet are mathematical coordinates, no dq interpretation',
        quantity='hypot(localRBF mean signed Br/Bt + frozen pureNN zero temporal mean signed Br/Bt)',
        selection='Complete validation original-T magnitude RMSE, all cases/frames/points retained',
        smoothing_note='Predeclared balanced parity assignment of 0 and .001 across48 recipes, no per-point validation tuning',
        cache_sha256=r.safe.digest(V1/'cache_v1/rotor.npz'),source_sha256=r.safe.digest(__file__),frozen_neural_sha256=r.safe.digest(nnpath)))
    rows=[];best=np.inf;bestm=None;bestrow=None;bestpred=None;bestmeans=None
    for index,spec in enumerate(configs):
        rr=coordinates(c[tr],spec['coordinate']);mu=rr.mean(0);scale=np.maximum(rr.std(0),1e-8);x=(rr-mu)/scale;xv=(coordinates(c[va],spec['coordinate'])-mu)/scale
        row=dict(index=index,spec=spec)
        try:
            inter=RBFInterpolator(x,yt,neighbors=spec['neighbors'],kernel=spec['kernel'],epsilon=spec['epsilon'],smoothing=spec['smoothing'],degree=1)
            mean=inter(xv);field=mean[:,None,:]+dynamic;pred=np.hypot(field[...,0::2],field[...,1::2])
            pc={str(cid):r.metrics(pred[j],truth[j]) for j,cid in enumerate(ids[va])}
            met=r.metrics(pred,truth);met.update(per_case=pc,worst_case_rmse_T=max(v['rmse_T'] for v in pc.values()))
            row.update(validation=met,validation_signed_mean=r.metrics(mean,true_signed_mean),
                       magnitude_of_mean_vs_true_mean_magnitude=r.metrics(np.hypot(mean[:,0::2],mean[:,1::2]),truth.mean(1)))
            if met['finite'] and met['rmse_T']<best:
                best=met['rmse_T'];bestrow=row;bestpred=pred.copy();bestmeans=mean.copy()
                bestm=dict(input_center=mu,input_scale=scale,centers=x,signed_mean_targets=yt,spec_json=np.array(json.dumps(spec)),
                    neural_relative_path=np.array('../../new60_opt_v2/neural_route/rotor/selected_model.npz'),neural_sha256=np.array(r.safe.digest(nnpath)))
                print('BEST',index,met,flush=True)
        except (ValueError,np.linalg.LinAlgError) as err:row['failure']=str(err)
        rows.append(row)
    np.savez_compressed(out/'model.npz',**bestm);(out/'model.npz.sha256').write_text(r.safe.digest(out/'model.npz')+'\n',encoding='ascii')
    replay=predict(out/'model.npz',c[va]);parity=float(np.max(np.abs(replay-bestpred)))
    if parity>1e-9:raise AssertionError('NumPy/SciPy parity mismatch '+str(parity))
    np.savez_compressed(out/'validation_predictions.npz',prediction=bestpred,mean_signed_prediction=bestmeans,case_ids=ids[va],point_id=pid,xy=xy)
    r.dump(out/'ranking.json',sorted(rows,key=lambda z:z.get('validation',{}).get('rmse_T',np.inf)))
    r.dump(out/'summary.json',dict(best=bestrow,elapsed_seconds=time.perf_counter()-start,candidate_count=len(rows),
        source_sha256=r.safe.digest(__file__),model_sha256=r.safe.digest(out/'model.npz'),numpy_scipy_max_difference_T=parity,
        test_labels_materialized=False,output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
