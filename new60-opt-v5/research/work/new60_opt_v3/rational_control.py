"""Finite train-only mean-response controls, original-T full-field scoring.

Prediction uses one common recipe across all points. No clipping of poles,
negative magnitudes, or failed points. Frozen V2 pureNN temporal residual is
optional and explicitly identified; all test field rows are skipped.
"""
import argparse, hashlib, json, sys, time, zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
V1=ROOT.parent/'new60_20261014'; V2=ROOT.parent/'new60_opt_v2'
sys.path.insert(0,str(V2/'neural_route'))
import cycle_neural as safe


def dump(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def mean_rows(path,indices):
    """Mean magnitudes and signed vectors, materialize one permitted row only."""
    index_to_output={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as a,a.open('field.npy') as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or shape!=(60,60,3600):raise ValueError('Invalid rotor field')
        out=np.empty((len(indices),1800));signed=np.empty((len(indices),3600))
        block=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(shape[0]):
            if i in index_to_output:
                raw=f.read(block); row=np.frombuffer(raw,dtype=dtype).reshape(shape[1:]).astype(np.float64)
                out[index_to_output[i]]=np.hypot(row[:,0::2],row[:,1::2]).mean(0)
                signed[index_to_output[i]]=row.mean(0)
                del row,raw
            else:f.seek(block,1)
    return out,signed


def coords(c,kind,current_scale):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));r=c[:,0]/current_scale;a=np.deg2rad(c[:,1])
    if kind=='native':return np.column_stack((r,c[:,1]/90))
    if kind=='polar':return np.column_stack((r*np.cos(a),r*np.sin(a)))
    if kind=='saturated':
        r=np.tanh(r);return np.column_stack((r*np.cos(a),r*np.sin(a)))
    raise ValueError(kind)


def powers(degree,constant=True):
    return [(i,total-i) for total in range(0 if constant else 1,degree+1) for i in range(total+1)]


def poly(x,p):return np.column_stack([x[:,0]**i*x[:,1]**j for i,j in p]) if len(p) else np.empty((len(x),0))


def target_forward(y,transform,mu,scale):
    if transform=='identity':return (y-mu)/scale
    if transform=='asinh':return np.arcsinh((y-mu)/scale)
    if transform=='log':return np.log(y+scale)
    raise ValueError(transform)


def target_inverse(z,transform,mu,scale):
    with np.errstate(over='ignore',invalid='ignore'):
        if transform=='identity':return mu+scale*z
        if transform=='asinh':return mu+scale*np.sinh(z)
        if transform=='log':return np.exp(z)-scale
    raise ValueError(transform)


def fit(c,y,spec):
    cur=max(float(np.max(c[:,0])),1.);raw=coords(c,spec['coordinate'],cur)
    center=raw.mean(0);scale=np.maximum(raw.std(0),1e-8);x=(raw-center)/scale
    pn=powers(spec['numerator_degree']);pd=powers(spec['denominator_degree'],False)
    num=poly(x,pn);den=poly(x,pd)
    mu=y.mean(0);ys=np.maximum(y.std(0),.005);target=target_forward(y,spec['transform'],mu,ys)
    # Point-major small linear systems; columns are numerator and denominator.
    d=np.concatenate((np.broadcast_to(num.T[None,:,:],(y.shape[1],num.shape[1],len(c))),
                      -target.T[:,None,:]*den.T[None,:,:]),axis=1)
    lhs=np.einsum('pik,pjk->pij',d,d)/len(c)
    rhs=np.einsum('pik,pk->pi',d,target.T)/len(c)
    reg=np.full(lhs.shape[-1],spec['ridge']);reg[0]=spec['ridge']*.001
    lhs[:,np.arange(lhs.shape[-1]),np.arange(lhs.shape[-1])]+=reg
    weights=np.linalg.solve(lhs,rhs[...,None])[...,0]
    return dict(input_center=center,input_scale=scale,current_scale=np.array(cur),target_mean=mu,target_scale=ys,
                numerator_powers=np.asarray(pn,dtype=np.int64),denominator_powers=np.asarray(pd,dtype=np.int64).reshape(-1,2),
                numerator_weights=weights[:,:len(pn)].T,denominator_weights=weights[:,len(pn):].T,
                spec_json=np.array(json.dumps(spec)))


def predict_mean(m,c):
    if isinstance(m,(str,Path)):
        with np.load(m,allow_pickle=False) as a:m={k:a[k] for k in a.files}
    spec=json.loads(str(m['spec_json']));c=np.atleast_2d(c)
    x=(coords(c,spec['coordinate'],float(m['current_scale']))-m['input_center'])/m['input_scale']
    den=1+poly(x,m['denominator_powers'])@m['denominator_weights'];num=poly(x,m['numerator_powers'])@m['numerator_weights']
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):pred=target_inverse(num/den,spec['transform'],m['target_mean'],m['target_scale'])
    return pred,den


def metrics(pred,truth):
    if not np.isfinite(pred).all():return {'finite':False,'nonfinite_count':int(np.count_nonzero(~np.isfinite(pred)))}
    e=pred-truth
    if np.max(np.abs(e))>1e100:return dict(finite=False,numerically_explosive=True,max_abs_error_exponent=float(np.log10(np.max(np.abs(e)))))
    en=float(np.sum(e*e));return dict(finite=True,rmse_T=float(np.sqrt(en/e.size)),
        relative_L2_pct=float(100*np.sqrt(en/np.sum(truth*truth))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        negative_count=int(np.count_nonzero(pred<0)))


def full_metrics(mean,dynamic,truth,ids):
    # Score the raw, complete 9 x 60 x 1800 predictions. No projection.
    pred=mean[:,None,:]+dynamic
    m=metrics(pred,truth)
    if m['finite']:
        pc={str(int(cid)):metrics(pred[i],truth[i]) for i,cid in enumerate(ids)}
        m.update(per_case=pc,worst_case_rmse_T=max(v['rmse_T'] for v in pc.values()))
    return m


def main(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);t0=time.perf_counter()
    split=safe.read_json(V1/'cache_v1/split.json');tr=split['train'];va=split['validation']
    with np.load(V1/'cache_v1/rotor.npz',allow_pickle=False) as a:
        conditions=a['conditions'];case_ids=a['case_ids'];point_id=a['point_id'];xy=a['xy']
    yt,ys=mean_rows(V1/'cache_v1/rotor.npz',tr)
    fieldval=safe.selected_rows(V1/'cache_v1/rotor.npz','field',va).astype(np.float64)
    truth=np.hypot(fieldval[...,0::2],fieldval[...,1::2]);ym=truth.mean(1)
    signed_mag=np.hypot(fieldval.mean(1)[:,0::2],fieldval.mean(1)[:,1::2])
    del fieldval
    nnpath=V2/'neural_route/rotor/selected_model.npz'
    nn=safe.predict(nnpath,conditions[va]);base=np.hypot(nn[...,0::2],nn[...,1::2]);del nn
    dynamic=base-base.mean(1,keepdims=True)
    coords_kinds=['native','polar']
    configs=[dict(coordinate=c,numerator_degree=n,denominator_degree=d,ridge=r,transform=t)
        for c in coords_kinds for n in (3,4) for d in (1,2) for r in (.00001,.001,.1) for t in ('identity','asinh')]
    configs += [dict(coordinate=c,numerator_degree=n,denominator_degree=0,ridge=r,transform='identity')
        for c in coords_kinds for n in (2,3,4,5,6) for r in (.00001,.001,.1)]
    configs += [dict(coordinate=c,numerator_degree=n,denominator_degree=d,ridge=r,transform='log')
        for c in coords_kinds for n in (3,4) for d in (0,1) for r in (.001,.1)]
    # 94 configurations. Whole-field RMSE common hyperparameter selection only.
    protocol=dict(configs=configs,candidate_count=len(configs),training_case_ids=case_ids[tr].tolist(),validation_case_ids=case_ids[va].tolist(),
        excluded_test_case_ids=case_ids[split['test']].tolist(),test_labels_materialized=False,full60_fit_used=False,
        normalization='train42 per-point mean/std floor0.005T; train-only mathematical native/polar coordinates, no dq interpretation',
        selection='Full validation original-T planar magnitude RMSE; all points/frames retained. Raw rational output without clipping.',
        dynamic='Frozen V2 pureNN zero temporal mean magnitude fluctuation; not trained or reselected here',
        dynamic_model_sha256=safe.digest(nnpath),cache_sha256=safe.digest(V1/'cache_v1/rotor.npz'),split_sha256=safe.digest(V1/'cache_v1/split.json'),
        source_sha256=safe.digest(__file__))
    dump(out/'protocol.json',protocol)
    baseline=dict(frozen_nn=metrics(base,truth),frozen_nn_mean_error=metrics(base.mean(1),ym),
        mean_truth_plus_nn_dynamic=full_metrics(ym,dynamic,truth,case_ids[va]),
        norm_of_signed_mean_vs_mean_norm=metrics(signed_mag,ym),train_mean_norm_vs_norm_of_mean=metrics(np.hypot(ys[:,0::2],ys[:,1::2]),yt))
    dump(out/'baseline.json',baseline)
    rows=[];best=float('inf');bestmodel=None;bestrow=None
    for index,spec in enumerate(configs):
        row=dict(index=index,spec=spec)
        try:
            m=fit(conditions[tr],yt,spec);mean,den=predict_mean(m,conditions[va]);trainmean,trainden=predict_mean(m,conditions[tr])
            row.update(validation=full_metrics(mean,dynamic,truth,case_ids[va]),validation_mean=metrics(mean,ym),
                train_mean=metrics(trainmean,yt),denominator_min_abs_validation=float(np.min(np.abs(den))),
                denominator_min_abs_train=float(np.min(np.abs(trainden))),denominator_nonpositive_validation=int(np.count_nonzero(den<=0)),
                denominator_near_pole_validation=int(np.count_nonzero(np.abs(den)<.05)),
                invalid_policy='No clipping; all finite raw outputs scored including negative magnitudes and poles')
            # A denominator sign/pole is reported and does not silently discard points.
            if row['validation']['finite'] and row['validation']['rmse_T']<best:
                best=row['validation']['rmse_T'];bestmodel=m;bestrow=row
                print('BEST',index,json.dumps(row,allow_nan=False),flush=True)
        except (np.linalg.LinAlgError,FloatingPointError) as err:row['failure']=str(err)
        rows.append(row)
        if index%10==0:dump(out/'running.json',dict(completed=len(rows),rows=rows,best=bestrow))
    if bestmodel is None:raise RuntimeError('No finite recipe')
    np.savez_compressed(out/'model.npz',**bestmodel);(out/'model.npz.sha256').write_text(safe.digest(out/'model.npz')+'\n',encoding='ascii')
    mean,den=predict_mean(bestmodel,conditions[va]);pred=mean[:,None,:]+dynamic
    np.savez_compressed(out/'validation_predictions.npz',prediction=pred,mean_prediction=mean,truth_mean=ym,denominator=den,case_ids=case_ids[va],point_id=point_id,xy=xy)
    dump(out/'ranking.json',sorted(rows,key=lambda r:r.get('validation',{}).get('rmse_T',float('inf'))))
    result=dict(best=bestrow,baseline=baseline,model_sha256=safe.digest(out/'model.npz'),elapsed_seconds=time.perf_counter()-t0,
        successful_configs=sum('validation' in r for r in rows),candidate_count=len(configs),test_labels_materialized=False,
        total_output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        assessment='Finite nonlinear rational control; do not advertise validation improvement as fresh-test performance')
    dump(out/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'rational_control'));args=p.parse_args()
    with threadpool_limits(limits=1):main(args.out)
