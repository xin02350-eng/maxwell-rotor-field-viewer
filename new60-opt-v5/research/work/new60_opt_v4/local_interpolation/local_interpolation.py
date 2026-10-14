"""112 finite local signed-mean interpolation / regime controls, no test access.

Saved deployment model contains train-derived local response parameters and
the full frozen V2 signed neural cycle weights. Inference requires only NumPy
and SciPy; no original cache, labels, or external model are accessed.
"""
from pathlib import Path
import argparse,hashlib,json,time,zipfile,warnings
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator,RBFInterpolator

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[1]
V1=WORK/'new60_20261014';V2=WORK/'new60_opt_v2'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def coordinates(c,kind):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));cur=c[:,0];a=np.deg2rad(c[:,1])
    if kind=='native':return c.copy()
    if kind.startswith('sat'):
        cap=float(kind[3:]);cur=cap*np.tanh(cur/cap)
    return np.column_stack((cur*np.cos(a),cur*np.sin(a)))


def features(c,kind,scale):
    rr=c[:,0]/scale;a=np.deg2rad(c[:,1]);x=rr*np.cos(a);y=rr*np.sin(a)
    if kind=='mixed':return np.column_stack((rr,c[:,1]/90,x,y,x*x,x*y,y*y))
    if kind=='native':return np.column_stack((rr,c[:,1]/90))
    if kind=='cartesian':return np.column_stack((x,y))
    return np.column_stack((x,y,x*x,x*y,y*y))


def neural_cycle(m,c):
    scale=float(m['current_scale']);x=(features(c,str(m['feature_kind']),scale)-m['feature_mean'])/m['feature_scale']
    for i in range(int(m['layer_count'])):
        x=x@m[f'w{i}']+m[f'b{i}']
        if i+1<int(m['layer_count']):x=np.tanh(x) if str(m['activation'])=='tanh' else x/(1+np.exp(-np.clip(x,-700,700)))
    cart=features(c,'cartesian',scale);degree=int(m['base_degree']);p=np.column_stack([cart[:,0]**i*cart[:,1]**(total-i) for total in range(degree+1) for i in range(total+1)])
    latent=x*m['output_scale']+p@m['base_weights'];coef=(m['cycle_mean']+latent@m['cycle_basis']).reshape(len(c),int(m['frame_count']),-1)
    return m['spatial_mean']+coef@m['spatial_basis'].T


def affine(x):return np.column_stack((np.ones(len(x)),x))


def mean_prediction(m,c,return_diagnostics=False):
    spec=json.loads(str(m['spec_json']));x=(coordinates(c,spec['coordinate'])-m['input_center'])/m['input_scale'];train=m['centers'];target=m['signed_mean_targets'];tri=Delaunay(train)
    simplex=tri.find_simplex(x);outside=simplex<0;out=np.empty((len(x),target.shape[-1]));inside=~outside
    if np.any(inside):
        q=x[inside];method=spec['method']
        if method=='delaunay_linear':
            s=simplex[inside];b=np.einsum('nij,nj->ni',tri.transform[s,:2,:],q-tri.transform[s,2,:]);weights=np.column_stack((b,1-b.sum(1)))
            out[inside]=np.einsum('ni,nid->nd',weights,target[tri.simplices[s]])
        elif method=='clough_tocher':
            fn=CloughTocher2DInterpolator(tri,target,tol=spec['tol'],maxiter=spec['maxiter']);out[inside]=fn(q)
        elif method=='local_rbf':
            fn=RBFInterpolator(train,target,neighbors=spec['neighbors'],kernel=spec['kernel'],epsilon=spec['epsilon'],degree=1,smoothing=spec['smoothing']);out[inside]=fn(q)
        elif method=='local_quadratic_mls':
            for j,xx in zip(np.flatnonzero(inside),q):
                distance=np.linalg.norm(train-xx,axis=1);nb=np.argsort(distance,kind='stable')[:spec['neighbors']];radius=max(float(distance[nb[-1]]),1e-8);dx=(train[nb]-xx)/radius
                d=np.column_stack((np.ones(len(nb)),dx,dx[:,0]**2,dx[:,0]*dx[:,1],dx[:,1]**2));w=np.exp(-.5*(distance[nb]/(spec['bandwidth']*radius))**2)
                lhs=d.T@(w[:,None]*d);pen=np.full(6,spec['ridge']);pen[0]*=.001;lhs+=np.diag(pen)
                coeff=np.linalg.solve(lhs,d.T@(w[:,None]*target[nb]));out[j]=coeff[0]
        elif method=='regime_affine':
            dist=np.sum((q[:,None,:]-train[None,:,:])**2,axis=-1);nb=np.argsort(dist,axis=1,kind='stable')[:,:6];cluster=m['train_regime_id'];k=int(m['regime_count']);gate=np.zeros((len(q),k))
            for i,n in enumerate(nb):
                rad=max(float(np.sqrt(dist[i,n[-1]])),1e-8);ww=np.exp(-.5*dist[i,n]/(rad*rad))
                for group in range(k):gate[i,group]=ww[cluster[n]==group].sum()
            gate/=np.maximum(gate.sum(1,keepdims=True),1e-100)
            if spec['gate']=='hard':gate=np.eye(k)[np.argmax(gate,axis=1)]
            preds=np.einsum('nf,kfd->nkd',affine(q),m['regime_affine_weights']);out[inside]=np.einsum('nk,nkd->nd',gate,preds)
        else:raise ValueError('Unknown local method')
    if np.any(outside):out[outside]=affine(x[outside])@m['exterior_affine_weights']
    if return_diagnostics:return out,dict(query_count=len(x),outside_train_convex_hull_count=int(outside.sum()),outside_query_indices=np.flatnonzero(outside).tolist())
    return out


def load_model(path):
    if isinstance(path,dict):return path
    path=Path(path)
    if digest(path)!=path.with_suffix('.npz.sha256').read_text().strip():raise ValueError('Model SHA mismatch')
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def predict(model,conditions,frame_index=None,return_signed=False):
    m=load_model(model);c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    if c.ndim!=2 or c.shape[-1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Finite native RMS>=0/Thet pairs required')
    baseline={k[9:]:v for k,v in m.items() if k.startswith('baseline_')};raw=neural_cycle(baseline,c);dyn=raw-raw.mean(1,keepdims=True)
    signed=mean_prediction(m,c)[:,None,:]+dyn;out=signed if return_signed else np.hypot(signed[...,0::2],signed[...,1::2])
    if not np.isfinite(out).all():raise ValueError('Nonfinite raw prediction')
    if frame_index is not None:
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<60:raise ValueError('Saved frame0..59 required')
        out=out[:,frame_index]
    return out[0] if single else out


def mean_rows(path,indices):
    map_to={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as z,z.open('field.npy') as f:
        ver=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,ver)
        if shape!=(60,60,3600) or fort:raise ValueError('Rotor shape guard')
        out=np.empty((len(indices),3600));block=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(60):
            if i in map_to:out[map_to[i]]=np.frombuffer(f.read(block),dtype=dtype).reshape(shape[1:]).astype(np.float64).mean(0)
            else:f.seek(block,1)
    return out


def val_rows(path,indices):
    map_to={int(i):j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as z,z.open('field.npy') as f:
        ver=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,ver)
        if shape!=(60,60,3600) or fort:raise ValueError('Rotor shape guard')
        out=np.empty((len(indices),*shape[1:]),dtype=np.float64);block=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(60):
            if i in map_to:out[map_to[i]]=np.frombuffer(f.read(block),dtype=dtype).reshape(shape[1:])
            else:f.seek(block,1)
    return out


def metrics(p,t,ids):
    if not np.isfinite(p).all():return dict(finite=False,nonfinite_count=int(np.count_nonzero(~np.isfinite(p))))
    e=p-t
    if np.max(np.abs(e))>1e100:return dict(finite=False,numerically_explosive=True,max_error_exponent=float(np.log10(np.max(np.abs(e)))))
    sse=np.sum(e*e,axis=(1,2));en=np.sum(t*t,axis=(1,2));rmse=np.sqrt(sse/np.prod(p.shape[1:]));wi=int(np.argmax(rmse));loc=np.unravel_index(np.argmax(np.abs(e)),e.shape)
    return dict(finite=True,rmse_T=float(np.sqrt(sse.sum()/e.size)),relative_L2_pct=float(100*np.sqrt(sse.sum()/en.sum())),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        worst_case_id=int(ids[wi]),worst_case_rmse_T=float(rmse[wi]),negative_magnitude_count=int(np.count_nonzero(p<0)),
        max_location=dict(case_id=int(ids[loc[0]]),frame_index=int(loc[1]),point_index=int(loc[2])),
        per_case={str(cid):dict(rmse_T=float(rmse[i]),relative_L2_pct=float(100*np.sqrt(sse[i]/en[i]))) for i,cid in enumerate(ids)})


def kmeans(data,k):
    rng=np.random.default_rng(731+k);first=int(rng.integers(len(data)));centers=[data[first]]
    for _ in range(1,k):
        dist=np.min(np.sum((data[:,None,:]-np.asarray(centers)[None,:,:])**2,axis=-1),axis=1);p=dist/dist.sum();centers.append(data[int(rng.choice(len(data),p=p))])
    centers=np.asarray(centers);old=None
    for _ in range(100):
        labels=np.argmin(np.sum((data[:,None,:]-centers[None,:,:])**2,axis=-1),axis=1)
        if old is not None and np.array_equal(old,labels):break
        for j in range(k):
            if not np.any(labels==j):raise ValueError('Empty response regime')
            centers[j]=data[labels==j].mean(0)
        old=labels.copy()
    return labels,centers


def run():
    out=ROOT
    if (out/'protocol.json').exists():raise FileExistsError(out/'protocol.json')
    t0=time.perf_counter();cache=V1/'cache_v1/rotor.npz';split=json.loads((V1/'cache_v1/split.json').read_text('utf8'));tr=split['train'];va=split['validation']
    kinds=('native','cartesian','sat150','sat300')
    configs=[dict(method='delaunay_linear',coordinate=c) for c in kinds]
    configs += [dict(method='clough_tocher',coordinate=c,tol=t,maxiter=it) for c in kinds for t,it in ((1e-6,10),(1e-6,100),(1e-4,400))]
    configs += [dict(method='local_quadratic_mls',coordinate=c,neighbors=k,bandwidth=h,ridge=r) for c in kinds for k in (12,18,24) for h in (1.,2.) for r in (1e-5,.001)]
    configs += [dict(method='local_rbf',coordinate=c,neighbors=k,kernel=kind,epsilon=1.,smoothing=.001) for c in kinds for k in (12,18,24) for kind in ('cubic','inverse_multiquadric')]
    configs += [dict(method='regime_affine',coordinate=c,regimes=k,gate=gate,ridge=.001) for c in kinds for k in (2,3,4) for gate in ('hard','soft')]
    baselinepath=V2/'neural_route/rotor/selected_model.npz'
    with np.load(baselinepath,allow_pickle=False) as z:base={k:z[k] for k in z.files}
    deps={str(baselinepath.relative_to(WORK)):digest(baselinepath)}
    if 'representation_relative_path' in base:
        rp=(baselinepath.parent/str(base['representation_relative_path'])).resolve();deps[str(rp.relative_to(WORK))]=digest(rp)
        with np.load(rp,allow_pickle=False) as z:base.update({k:z[k] for k in z.files if k not in base})
    dump(out/'protocol.json',dict(version='V4-local-signed-mean-controls',candidate_count=len(configs),configs=configs,training_case_ids=tr,validation_case_ids=va,excluded_test_case_ids=split['test'],
        train_only='All signed mean targets, regime PCA/clusters, condition standardization and affine coefficients use train42 only',
        objective='Complete original-T val9x60x1800 planar magnitude RMSE, signed mean response + frozen V2 signed zero-time-mean pureNN dynamics then hypot',
        selection='Finite common recipe chosen by raw whole validation field RMSE; no point-specific validation choice, no test/full60fit scores',
        exterior_policy='Any outside-training-convex-hull query uses the predeclared global affine response trained42 with ridge.001 (intercept weakly penalized). Exterior counts are reported, never delete a point/case.',
        regime_policy='Train signed mean fullfield response PCA4 then deterministic kmeans2/3/4. Local nearest6 coordinate votes route affine experts, hard/soft predeclared.',
        input_note='Native Thet and I*cos(Thet)/I*sin(Thet) are mathematical coordinates, not a claim to physical dq currents',
        test_labels_materialized=False,full60_fitting_used=False,neural_training_runs=0,
        source_sha256=digest(__file__),cache_sha256=digest(cache),split_sha256=digest(V1/'cache_v1/split.json'),dependencies=deps))
    with np.load(cache,allow_pickle=False) as z:c=z['conditions'];ids=z['case_ids'];pid=z['point_id'];xy=z['xy']
    target=mean_rows(cache,tr);val=val_rows(cache,va);truth=np.hypot(val[...,0::2],val[...,1::2]);vmean=val.mean(1);del val
    old=neural_cycle(base,c[va]);dynamic=old-old.mean(1,keepdims=True)
    centered=target-target.mean(0);u,s,vt=np.linalg.svd(centered,full_matrices=False);raw=centered@vt[:4].T;pc_scale=np.maximum(raw.std(0),1e-6);response_pc=raw/pc_scale
    clusters={k:kmeans(response_pc,k) for k in (2,3,4)}
    dump(out/'regimes_train_only.json',dict(train_case_ids=tr,pca_rank=4,singular_values=s.tolist(),cluster_ids={str(k):v[0].tolist() for k,v in clusters.items()},pca_training_only=True))
    common={}
    for kind in kinds:
        xx=coordinates(c[tr],kind);center=xx.mean(0);sc=np.maximum(xx.std(0),1e-8);train=(xx-center)/sc;d=affine(train);pen=np.array([1e-6,.001,.001]);external=np.linalg.solve(d.T@d+np.diag(pen),d.T@target)
        common[kind]=dict(input_center=center,input_scale=sc,centers=train,signed_mean_targets=target,exterior_affine_weights=external)
    rows=[];best=np.inf;bm=None;br=None;bp=None
    for index,spec in enumerate(configs):
        model=dict(common[spec['coordinate']],spec_json=np.array(json.dumps(spec)),point_id=pid,xy=xy,**{'baseline_'+k:v for k,v in base.items()})
        if spec['method']=='regime_affine':
            cluster=clusters[spec['regimes']][0];weights=[];d=affine(model['centers'])
            for j in range(spec['regimes']):
                choose=cluster==j;pen=np.array([1e-6,spec['ridge'],spec['ridge']]);weights.append(np.linalg.solve(d[choose].T@d[choose]+np.diag(pen),d[choose].T@target[choose]))
            model.update(train_regime_id=cluster,regime_count=np.array(spec['regimes']),regime_affine_weights=np.stack(weights))
        row=dict(index=index,configuration=spec)
        try:
            with warnings.catch_warnings(record=True) as warn:
                warnings.simplefilter('always');mean,diag=mean_prediction(model,c[va],True)
            signed=mean[:,None,:]+dynamic;pred=np.hypot(signed[...,0::2],signed[...,1::2]);met=metrics(pred,truth,ids[va])
            row.update(validation=met,query_geometry=diag,warnings=[str(w.message) for w in warn],
                       validation_signed_mean_rmse_T=float(np.sqrt(np.mean((mean-vmean)**2))))
            if met['finite'] and met['rmse_T']<best:best=met['rmse_T'];bm=model;br=row;bp=pred.copy();print('BEST',index,spec,met,flush=True)
        except (ValueError,np.linalg.LinAlgError) as e:row['failure']=str(e)
        rows.append(row)
    if bm is None:raise RuntimeError('No successful candidate')
    bm['metadata_json']=np.array(json.dumps(dict(format='V4-local-signed-mean-control',selected_configuration=br['configuration'],training_case_ids=tr,validation_case_ids=va,test_labels_materialized=False,source_sha256=digest(__file__),dependencies=deps)))
    np.savez_compressed(out/'selected_model.npz',**bm);(out/'selected_model.npz.sha256').write_text(digest(out/'selected_model.npz')+'\n',encoding='ascii')
    replay=predict(out/'selected_model.npz',c[va]);err=float(np.max(np.abs(replay-bp)))
    if err>1e-9:raise AssertionError('Portable model replay disagreement '+str(err))
    np.savez_compressed(out/'validation_best.npz',prediction=bp,case_ids=ids[va],point_id=pid,xy=xy,quantity=np.array('planar_magnitude'))
    dump(out/'candidate_results.json',dict(rows=rows,selected=br,baseline_frozen_V2_pureNN=metrics(np.hypot(old[...,0::2],old[...,1::2]),truth,ids[va]),
        successful_candidate_count=sum('validation' in r for r in rows),failed_candidate_count=sum('failure' in r for r in rows),
        completed_configurations=len(rows),neural_training_runs=0,test_labels_materialized=False,full60_fitting_used=False,
        model_sha256=digest(out/'selected_model.npz'),inference_source_sha256=digest(__file__),portable_replay_max_difference_T=err,
        elapsed_seconds=time.perf_counter()-t0))
    dump(out/'file_manifest.json',{p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(selected=br,completed=len(rows),elapsed_seconds=time.perf_counter()-t0)),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):run()
