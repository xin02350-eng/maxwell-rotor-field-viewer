"""V6 pointwise sensitivity-oriented RBF. NumPy training and inference.

Each fixed probe gets a train-only sensitivity direction. Its RBF interpolates
the full 60-frame response, not a validation mean. Hyperparameter routing uses
training-only conditional leave-one-out; global recipes use old validation9.
This is analytic fixed-center RBF regression, not gradient-trained NN.
"""
from pathlib import Path
import argparse, hashlib, json, time, zipfile, sys
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def rows(path,indices,key='field'):
    """Decode requested rows only. No excluded test-row numerical access."""
    indices=list(map(int,indices));mapping={i:j for j,i in enumerate(indices)}
    with zipfile.ZipFile(path) as z,z.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,version)
        if fort or shape[0]!=60 or shape[1]!=60 or dtype.kind!='f':raise ValueError('Cache layout')
        out=np.empty((len(indices),*shape[1:]),dtype=np.float64);size=int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(60):
            if i in mapping:
                raw=f.read(size)
                if len(raw)!=size:raise ValueError('Short source row')
                out[mapping[i]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(size,1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite labels')
    return out

def metrics(p,y):
    e=p-y;v=np.unravel_index(np.abs(e).argmax(),e.shape)
    return dict(relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(y)),rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.abs(e).mean()),max_abs_T=float(np.abs(e).max()),
                max_abs_indices=list(map(int,v)),worst_case_rmse_T=float(np.sqrt(np.mean(e*e,axis=(1,2))).max()))

def coords(c,current_scale,cap):
    c=np.atleast_2d(np.asarray(c,dtype=np.float64));r=c[:,0]/current_scale
    if cap:r=cap*np.tanh(r/cap)
    a=np.deg2rad(c[:,1]);return np.column_stack((r*np.cos(a),r*np.sin(a)))

def polynomial(x,degree):
    return np.stack([x[...,0]**i*x[...,1]**(t-i) for t in range(degree+1) for i in range(t+1)],axis=-1)

def kernel(x,y,kind,length):
    r2=np.sum((x[..., :,None,:]-y[...,None,:,:])**2,axis=-1)/length**2
    if kind=='gaussian':return np.exp(-.5*r2)
    if kind=='inverse_multiquadric':return 1/np.sqrt(1+r2)
    if kind=='multiquadric':return -np.sqrt(1+r2)
    if kind=='matern52':
        q=np.sqrt(5*r2);return (1+q+q*q/3)*np.exp(-q)
    raise ValueError(kind)

def prepare_geometry(c,y,cap):
    current_scale=float(np.median(c[:,0]));raw=coords(c,current_scale,cap);center=raw.mean(0);scale=np.maximum(raw.std(0),1e-12);x=(raw-center)/scale
    target=y.transpose(2,0,1,3).reshape(y.shape[2],len(c),-1)
    # Dominant condition sensitivity from all saved training frames/components.
    coef=np.linalg.lstsq(np.column_stack((np.ones(len(c)),x)),target.transpose(1,0,2).reshape(len(c),-1),rcond=1e-12)[0][1:]
    coef=coef.reshape(2,y.shape[2],-1).transpose(1,0,2)
    sens=coef@coef.transpose(0,2,1);_,eigenvectors=np.linalg.eigh(sens);direction=eigenvectors[:,:,-1]
    rotation=np.stack((direction,np.column_stack((-direction[:,1],direction[:,0]))),axis=2)
    return dict(current_scale=np.array(current_scale),cap=np.array(cap),center=center,scale=scale,rotation=rotation),target

def mapped(m,c,ratio):
    x=(coords(c,float(m['current_scale']),float(m['cap']))-m['center'])/m['scale']
    return np.einsum('ni,pij->pnj',x,m['rotation'])*np.array([np.sqrt(ratio),1/np.sqrt(ratio)])

def system(x,spec):
    pol=polynomial(x,spec['degree']);n=x.shape[1];d=pol.shape[-1]
    k=kernel(x,x,spec['kernel'],spec['lengthscale'])+spec['ridge']*np.eye(n)
    return np.concatenate((np.concatenate((k,pol),axis=2),np.concatenate((pol.transpose(0,2,1),np.zeros((len(x),d,d))),axis=2)),axis=1)

def solve_candidate(m,c,target,cv,spec):
    x=mapped(m,c,spec['anisotropy']);xv=mapped(m,cv,spec['anisotropy']);n=len(c);pol=polynomial(x,spec['degree']);d=pol.shape[-1]
    inv=np.linalg.inv(system(x,spec));rhs=np.concatenate((target,np.zeros((len(x),d,target.shape[-1]))),axis=1);sol=inv@rhs
    diag=np.diagonal(inv,axis1=1,axis2=2)[:,:n]
    if np.any(np.abs(diag)<1e-14):raise ValueError('Unstable LOO diagonal')
    loo=sol[:,:n,:]/diag[...,None];loo_mse=np.mean(loo*loo,axis=(1,2))
    pred=kernel(xv,x,spec['kernel'],spec['lengthscale'])@sol[:,:n]+polynomial(xv,spec['degree'])@sol[:,n:]
    if not np.isfinite(pred).all() or not np.isfinite(loo_mse).all():raise ValueError('Nonfinite fit')
    return pred,loo_mse,sol

def quantity(p,route):
    # Internal point,case,frame*components -> case,frame,point output.
    if route=='signed':
        v=p.reshape(p.shape[0],p.shape[1],60,2);return np.hypot(v[...,0],v[...,1]).transpose(1,2,0)
    return p.reshape(p.shape[0],p.shape[1],60).transpose(1,2,0)



def baseline(region,c):
    sys.path.insert(0,str(WORK/'new60_opt_v5/code'))
    import inference_numpy_v5 as safe
    root=WORK/'new60_opt_v5';freeze=safe.read_freeze(root)
    id='prediction_rotor_magnitude' if region=='rotor' else 'prediction_'+region+('_Br' if region=='airgap' else '_magnitude')
    return safe.predict(root,next(v for v in freeze['models'] if v['id']==id),c)

def run(region,route):
    out=ROOT/(region+'_'+route);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    split=json.loads((WORK/'new60_20261014/cache_v1/split.json').read_text());tr=split['train'];va=split['validation'];source=WORK/'new60_20261014/cache_v1'/f'{region}.npz'
    with np.load(source,allow_pickle=False) as z:conditions=z['conditions'];geometry={k:z[k] for k in ['point_id','xy','theta']}
    specs=[dict(kernel=k,lengthscale=l,ridge=r,anisotropy=a,degree=2,cap=0.) for k in ['gaussian','inverse_multiquadric','multiquadric','matern52'] for l in [.6,1.8,5.4] for r in [1e-6,.001] for a in [1.,3.,9.]]
    # Saturated-plane controls: new orientation still comes exclusively from train.
    specs += [dict(kernel=k,lengthscale=1.8,ridge=.001,anisotropy=a,degree=2,cap=.7) for k in ['gaussian','inverse_multiquadric','multiquadric','matern52'] for a in [1.,3.,9.]]
    protocol=dict(region=region,route=route,specs=specs,source_sha256=sha(__file__),cache_sha256=sha(source),train_case_ids=tr,validation_case_ids=va,excluded_test_case_ids=split['test'],
        selector_variants=['best_shared_validation','pointwise_trainLOO','pointwise_trainLOO_shrunk50'],baseline_blend_weights=[0.,.25,.5,.75,1.],
        point_selector='All60 frame/component training conditional LOO MSE; no validation point routing',global_selector='Full9 validation magnitude/Br RMSE; report maxabs separately',
        test_labels_read=False,full60_fit_used=False,gradient_NN_training_count=0,previous_validation_reused=True,
        method='Full-frame output-specific sensitivity-oriented RBF, all spatial points retained; native angle not claimed dq',reference='https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html')
    dump(out/'protocol.json',protocol)
    train=rows(source,tr);valid=rows(source,va);components=1 if region=='airgap' else 2
    train=train.reshape(len(tr),60,-1,components);valid=valid.reshape(len(va),60,-1,components)
    truth=valid[...,0] if components==1 else np.hypot(valid[...,0],valid[...,1])
    if route=='magnitude':train=np.hypot(train[...,0],train[...,1])[...,None]
    m,target=prepare_geometry(conditions[tr],train,0.);sat,_=prepare_geometry(conditions[tr],train,.7)
    base=baseline(region,conditions[va]);base_metrics=metrics(base,truth)
    selector_error=np.full(target.shape[0],np.inf);selector_pred=np.zeros((target.shape[0],9,target.shape[-1]));choice=np.zeros(target.shape[0],dtype=int)
    sharedbest=np.inf;sharedpred=None;sharedindex=None;records=[]
    for i,spec in enumerate(specs):
        try:
            pred,loo,_=solve_candidate(sat if spec['cap'] else m,conditions[tr],target,conditions[va],spec)
            p=quantity(pred,route);met=metrics(p,truth);row=dict(index=i,spec=spec,validation=met,train_conditional_LOO_mse_T2=float(loo.mean()),status='PASS')
            mask=loo<selector_error;selector_error[mask]=loo[mask];selector_pred[mask]=pred[mask];choice[mask]=i
            if met['rmse_T']<sharedbest:sharedbest=met['rmse_T'];sharedpred=pred.copy();sharedindex=i
        except (ValueError,np.linalg.LinAlgError) as exc:row=dict(index=i,spec=spec,status='FAILED',reason=str(exc))
        records.append(row)
        if (i+1)%12==0:
            dump(out/'progress.json',dict(completed=i+1,total=len(specs),best_shared_rmse_T=sharedbest,elapsed_seconds=time.perf_counter()-start))
            print(region,route,i+1,len(specs),'val',sharedbest,'seconds',time.perf_counter()-start,flush=True)
    if sharedpred is None:raise ValueError('No successful RBF')
    # For selector, each point may need a different cap/scaler. Save groups with
    # group-specific geometries in a dedicated helper below.
    routes=[('shared',sharedpred,np.full(target.shape[0],sharedindex)),('pointLOO',selector_pred,choice),('pointLOO_shrunk50',.5*selector_pred+.5*sharedpred,None)]
    recipes=[];best=base_metrics['rmse_T'];selected=dict(route='baseline',alpha=0.,metrics=base_metrics)
    for name,raw,selection in routes:
        value=quantity(raw,route)
        for alpha in protocol['baseline_blend_weights']:
            p=alpha*value+(1-alpha)*base;met=metrics(p,truth);rec=dict(route=name,alpha=alpha,metrics=met);recipes.append(rec)
            if met['rmse_T']<best:best=met['rmse_T'];selected=rec;bestpred=p.copy()
    # Store two primitive models; selector rotation/scales differ only with cap,
    # so each group's complete geometry is included rather than changing source.
    def write(name,selection):
        arrays={k:geometry[k] for k in ['point_id','xy','theta']};groups=[]
        for g,index in enumerate(sorted(set(selection.tolist()))):
            spec=specs[index];points=np.flatnonzero(selection==index);geom=dict(sat if spec['cap'] else m);geom['rotation']=geom['rotation'][points]
            _,_,sol=solve_candidate(geom,conditions[tr],target[points],conditions[va][:1],spec);pre=f'g{g}_'
            arrays.update({pre+k:v for k,v in geom.items()});arrays.update({pre+'points':points,pre+'centers':mapped(geom,conditions[tr],spec['anisotropy']),pre+'weights':sol[:,:len(tr)],pre+'trend':sol[:,len(tr):]});groups.append(spec)
        arrays['metadata_json']=np.array(json.dumps(dict(format='v6-pointwise-rbf-grouped',groups=groups,region=region,route=route,train_case_ids=tr,source_sha256=sha(__file__),gradient_neural_training=False)))
        path=out/(name+'.npz');np.savez_compressed(path,**arrays);path.with_suffix('.npz.sha256').write_text(sha(path)+'\n')
        return sha(path)
    hashes={name:write(name,selection) for name,_,selection in routes[:2]}
    np.savez_compressed(out/'validation_predictions.npz',shared=quantity(sharedpred,route),pointLOO=quantity(selector_pred,route),baseline=base,truth=truth,case_ids=np.array(va))
    np.savez_compressed(out/'train_LOO_routing.npz',candidate_index=choice,conditional_LOO_mse_T2=selector_error,train_case_ids=np.array(tr))
    report=dict(region=region,quantity=route,candidates=len(specs),actual_gradient_NN_runs=0,records=records,baseline=base_metrics,shared_index=sharedindex,recipes=recipes,selected=selected,
        model_hashes=hashes,protocol_sha256=sha(out/'protocol.json'),source_sha256=sha(__file__),elapsed_seconds=time.perf_counter()-start,test_labels_read=False)
    dump(out/'results.json',report);print(json.dumps({k:report[k] for k in ['region','quantity','baseline','selected','elapsed_seconds']}),flush=True)

def predict_grouped(model,conditions,frame_index=None):
    path=Path(model)
    if sha(path)!=path.with_suffix('.npz.sha256').read_text().strip():raise ValueError('RBF model SHA mismatch')
    with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    meta=json.loads(str(m['metadata_json']));c=np.asarray(conditions,dtype=float);single=c.ndim==1;c=np.atleast_2d(c)
    if c.shape[1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Expected finite native RMS/current-angle')
    out=np.empty((len(m['point_id']),len(c),120 if meta['route']=='signed' else 60))
    for i,spec in enumerate(meta['groups']):
        pre=f'g{i}_';geom={k[len(pre):]:v for k,v in m.items() if k.startswith(pre)};x=mapped(geom,c,spec['anisotropy'])
        out[geom['points']]=kernel(x,geom['centers'],spec['kernel'],spec['lengthscale'])@geom['weights']+polynomial(x,spec['degree'])@geom['trend']
    result=quantity(out,meta['route'])
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<60:raise ValueError('Saved frame index 0..59 only')
        result=result[:,frame_index]
    if not np.isfinite(result).all():raise ValueError('Nonfinite RBF output')
    return result[0] if single else result

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--region',choices=['rotor','stator','airgap'],required=True);p.add_argument('--route',choices=['signed','magnitude','Br'],required=True);args=p.parse_args()
    if (args.region=='airgap')!=(args.route=='Br'):raise ValueError('Airgap uses signed Br; vector regions use signed or magnitude')
    with threadpool_limits(limits=2):run(args.region,args.route)
