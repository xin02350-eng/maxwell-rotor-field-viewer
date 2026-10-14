"""Six train-only nearest-neighbour response coordinates; raw original-T QA.

Neighbour IDs use geometry only. Ratio/difference target fits a common recipe
at every original point. The neighbor reference and time residual used during
deployment come from frozen pureNN predictions, never validation truth.
"""
from pathlib import Path
import json,time
import numpy as np
from threadpoolctl import threadpool_limits
import rational_control as r

ROOT=r.ROOT;V1=r.V1;V2=r.V2


def predict(model,conditions,frame_index=None):
    """Pure NumPy mean control + frozen pureNN time residual."""
    if isinstance(model,(str,Path)):
        path=Path(model)
        with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
        nnpath=(path.parent/str(m['neural_relative_path'])).resolve()
        if r.safe.digest(nnpath)!=str(m['neural_sha256']):raise ValueError('Frozen neural model mismatch')
    else:m=model;nnpath=V2/'neural_route/rotor/selected_model.npz'
    c=np.asarray(conditions);one=c.ndim==1;c=np.atleast_2d(c)
    signed=r.safe.predict(nnpath,c);nn=np.hypot(signed[...,0::2],signed[...,1::2]);base=nn.mean(1)
    neighbor=base[:,m['neighbor_indices']].mean(-1)
    response,_=r.predict_mean(m,c)
    if str(m['neighbor_transform'])=='ratio':mean=(neighbor+float(m['ratio_epsilon']))*response-float(m['ratio_epsilon'])
    else:mean=neighbor+response
    pred=mean[:,None,:]+nn-base[:,None,:]
    if frame_index is not None:pred=pred[:,frame_index]
    return pred[0] if one else pred


def run():
    out=ROOT/'rational_neighbor_control'
    if out.exists():raise FileExistsError(out)
    out.mkdir();start=time.perf_counter()
    split=r.safe.read_json(V1/'cache_v1/split.json');tr=split['train'];va=split['validation']
    with np.load(V1/'cache_v1/rotor.npz',allow_pickle=False) as z:
        condition=z['conditions'];xy=z['xy'];pid=z['point_id'];ids=z['case_ids']
    yt,_=r.mean_rows(V1/'cache_v1/rotor.npz',tr)
    fv=r.safe.selected_rows(V1/'cache_v1/rotor.npz','field',va).astype(np.float64)
    truth=np.hypot(fv[...,0::2],fv[...,1::2]);del fv
    # Geometry only, no target correlations or validation response enter neighbour choice.
    dist=np.sum((xy[:,None,:]-xy[None,:,:])**2,axis=-1);np.fill_diagonal(dist,np.inf)
    order=np.argsort(dist,axis=1)[:,:8];del dist
    neuralpath=V2/'neural_route/rotor/selected_model.npz'
    signed=r.safe.predict(neuralpath,condition[va]);nn=np.hypot(signed[...,0::2],signed[...,1::2]);del signed
    nmean=nn.mean(1);dynamic=nn-nmean[:,None,:]
    epsilon=max(float(np.median(yt))*.01,.005)
    # Follow-up uses the recorded best polynomial family, one global ridge/degree.
    spec=dict(coordinate='polar',numerator_degree=6,denominator_degree=0,ridge=.001,transform='identity')
    configs=[dict(neighbor_count=k,neighbor_transform=t) for k in (1,3,8) for t in ('ratio','difference')]
    r.dump(out/'protocol.json',dict(candidate_count=6,configs=configs,spec=spec,
        train_case_ids=ids[tr].tolist(),validation_case_ids=ids[va].tolist(),excluded_test_case_ids=ids[split['test']].tolist(),
        neighbor_selection='Nearest spatial coordinates only, excludes own point; no label-based selection',
        epsilon=epsilon,reference='Train target uses train true neighbour mean; inference uses frozen pureNN neighbour mean',
        selection='Complete raw original-T validation RMSE, all points/frames; no clipping',
        training_labels_materialized=tr,validation_labels_materialized=va,test_labels_materialized=False,
        full60_fitting_used=False,cache_sha256=r.safe.digest(V1/'cache_v1/rotor.npz'),
        source_sha256=r.safe.digest(__file__),mean_module_sha256=r.safe.digest(r.__file__),
        frozen_neural_sha256=r.safe.digest(neuralpath)))
    rows=[];best=np.inf;bestm=None;bestrow=None
    for index,conf in enumerate(configs):
        neighbors=order[:,:conf['neighbor_count']];nb=yt[:,neighbors].mean(-1)
        target=(yt+epsilon)/(nb+epsilon) if conf['neighbor_transform']=='ratio' else yt-nb
        m=r.fit(condition[tr],target,spec)
        m.update(neighbor_indices=neighbors,neighbor_point_ids=pid[neighbors],neighbor_transform=np.array(conf['neighbor_transform']),
            ratio_epsilon=np.array(epsilon),neural_relative_path=np.array('../../new60_opt_v2/neural_route/rotor/selected_model.npz'),neural_sha256=np.array(r.safe.digest(neuralpath)))
        pred=predict(m,condition[va]);mean=pred.mean(1)
        row=dict(index=index,config=conf,validation=r.full_metrics(mean,dynamic,truth,ids[va]),
            validation_mean=r.metrics(mean,truth.mean(1)),target_train_min=float(target.min()),target_train_max=float(target.max()))
        rows.append(row)
        if row['validation']['rmse_T']<best:best=row['validation']['rmse_T'];bestm=m;bestrow=row
        print(index,row['validation'],flush=True)
    np.savez_compressed(out/'model.npz',**bestm);(out/'model.npz.sha256').write_text(r.safe.digest(out/'model.npz')+'\n',encoding='ascii')
    deployed=predict(out/'model.npz',condition[va]);check=r.metrics(deployed,truth)
    if not np.isclose(check['rmse_T'],best):raise AssertionError('Saved model replay mismatch')
    np.savez_compressed(out/'validation_predictions.npz',prediction=deployed,case_ids=ids[va],point_id=pid,xy=xy)
    r.dump(out/'ranking.json',sorted(rows,key=lambda x:x['validation']['rmse_T']))
    r.dump(out/'summary.json',dict(best=bestrow,replay=check,elapsed_seconds=time.perf_counter()-start,
        candidate_count=len(rows),test_labels_materialized=False,output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
