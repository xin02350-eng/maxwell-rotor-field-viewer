"""Export V5 inference from native inputs and SHA-verified fixed geometry."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from inference_numpy_v5 import digest,read_freeze,predict
from scope_utils_v5 import normalize_scope,case_role,select_choice,original_split
ROOT=Path(__file__).resolve().parent.parent
INPUT_KEYS=('conditions','case_ids','point_id','xy','theta','time_s','solver_index','theta_m_relative_deg')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default=str(ROOT));p.add_argument('--branch',choices=['prediction','fitting_full60'])
    p.add_argument('--region',choices=['rotor','stator','airgap']);p.add_argument('--model-id')
    p.add_argument('--case-id',type=int);p.add_argument('--current-rms',type=float);p.add_argument('--thet-deg',type=float)
    p.add_argument('--frame',type=int);p.add_argument('--coordinates',choices=['reference','world'],default='world');p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.study).resolve();registry=read_freeze(root)
    r=select_choice(registry['models'],args.model_id,args.branch,args.region)
    region=r['region'];branch=r['branch']
    cache=(root/r.get('cache_relative_path',f'../new60_20261014/cache_v1/{region}.npz')).resolve()
    if digest(cache)!=r['cache_sha256']:raise ValueError('Input/geometry cache SHA mismatch')
    # Numeric label/field/coeff/target entries are deliberately not accessed.
    with np.load(cache,allow_pickle=False) as z:a={k:z[k] for k in INPUT_KEYS}
    if (args.current_rms is None)!=(args.thet_deg is None):raise ValueError('Both native inputs required')
    if args.current_rms is not None:
        if args.case_id is not None:raise ValueError('Use case-id or input pair')
        c=np.array([args.current_rms,args.thet_deg]);match=np.flatnonzero(np.all(np.isclose(a['conditions'],c,rtol=0,atol=1e-10),axis=1));cid=int(a['case_ids'][match[0]]) if len(match) else None
    else:
        cid=2 if args.case_id is None else args.case_id;match=np.flatnonzero(a['case_ids']==cid)
        if len(match)!=1:raise ValueError('Unknown original case ID')
        c=a['conditions'][match[0]]
    value=predict(root,r,c,args.frame)
    frames=np.arange(60) if args.frame is None else np.array([args.frame]);value=value if args.frame is None else value[None,:]
    ref=np.asarray(a['xy'],dtype=float);xy=np.broadcast_to(ref,(len(frames),*ref.shape)).copy();theta=a['theta'][frames]
    if region=='rotor' and args.coordinates=='world':
        ang=np.deg2rad(theta);cs=np.cos(ang)[:,None];sn=np.sin(ang)[:,None]
        xy[...,0]=cs*ref[None,:,0]-sn*ref[None,:,1];xy[...,1]=sn*ref[None,:,0]+cs*ref[None,:,1]
    split_path=(root/registry['split_relative_path']).resolve()
    if digest(split_path)!=registry['split_sha256']:raise ValueError('Original split SHA mismatch')
    split=json.loads(split_path.read_text(encoding='utf-8-sig'));old=original_split(split);scope=normalize_scope(r,split)
    role=next((k for k in ['train','validation','test'] if cid in old[k]),'unseen_input');effective=case_role(scope,cid)
    score_scope={'training_fit':'all60 training-fit inherited from V3; not unseen-condition accuracy','train':'training-in-sample; not unseen-condition accuracy',
                 'validation':'original42 validation; used for model selection','reused_test':'original nine reused-test conditions; not fresh blind evaluation','unseen_input':'new input; accuracy not established'}[effective]
    meta=dict(model_id=r['id'],branch=branch,region=region,method=r.get('method',r['family']),quantity=r['quantity'],units='T',xy_units='mm',original_case_id=cid,
        native_condition=c.tolist(),current_convention='RMS, no extra divide by branches',coordinates=args.coordinates,mechanical_angle='absolute degrees',
        frame_indices=frames.tolist(),score_scope=score_scope,old_prediction_split_role=role,effective_role=effective,training_scope=scope,model_sha256=r['model_sha256'],
        choices_sha256=digest(root/'both_versions_choices_frozen.json'),field_labels_read=False,missing_Bz=True)
    destination=Path(args.output)
    if destination.exists() or destination.with_suffix(destination.suffix+'.json').exists():raise FileExistsError(destination)
    if destination.suffix.lower() not in ('.csv','.npz'):raise ValueError('Use .csv or .npz')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.suffix.lower()=='.npz':
        np.savez_compressed(destination,prediction=value,xy=xy,xy_reference=ref,point_id=a['point_id'],theta=theta,time_s=a['time_s'][frames],solver_index=a['solver_index'][frames],metadata_json=np.array(json.dumps(meta,ensure_ascii=False)))
    else:
        vector=r['quantity'] in ['vector','Br,Bt']
        columns=['Br_T','Bt_T','B_planar_magnitude_T'] if vector else ['Br_T' if r['quantity']=='Br' else 'B_planar_magnitude_T']
        with destination.open('x',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(['frame_index','solver_index','time_s','theta_mechanical_absolute_deg','point_id','x_mm','y_mm',*columns])
            for j,frame in enumerate(frames):
                for i,pid in enumerate(a['point_id']):
                    vals=[float(value[j,2*i]),float(value[j,2*i+1]),float(np.hypot(value[j,2*i],value[j,2*i+1]))] if vector else [float(value[j,i])]
                    w.writerow([int(frame),int(a['solver_index'][frame]),float(a['time_s'][frame]),float(theta[j]),int(pid),*xy[j,i].tolist(),*vals])
    destination.with_suffix(destination.suffix+'.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(destination),shape=list(value.shape),model_id=r['id'],field_labels_read=False),ensure_ascii=False))

if __name__=='__main__':main()
