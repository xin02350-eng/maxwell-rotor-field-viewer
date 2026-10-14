"""Export predictions from verified two-version choices, without field labels."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from inference_numpy_v2 import digest,load_model,predict

ROOT=Path(__file__).resolve().parent.parent
INPUT_KEYS=('conditions','case_ids','point_id','xy','theta','time_s','solver_index','theta_m_relative_deg')

def verified_choices(root):
    root=Path(root).resolve();path=root/'both_versions_choices_frozen.json'
    if digest(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Choices SHA mismatch')
    choices=json.loads(path.read_text(encoding='utf8'))
    if digest(root/'protocol.json')!=choices['protocol_sha256']:raise ValueError('Protocol SHA mismatch')
    pp=root/'prediction_choices_frozen.json'
    if digest(pp)!=choices['prediction_freeze_sha256']:raise ValueError('Prediction freeze mismatch')
    return choices

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default=str(ROOT));p.add_argument('--branch',choices=['prediction','fitting_full60'],default='fitting_full60');p.add_argument('--region',choices=['rotor','stator','airgap'],default='rotor');p.add_argument('--model-id');p.add_argument('--quantity',choices=['planar_magnitude','Br,Bt','Br']);p.add_argument('--case-id',type=int);p.add_argument('--current-rms',type=float);p.add_argument('--thet-deg',type=float);p.add_argument('--frame',type=int);p.add_argument('--coordinates',choices=['reference','world'],default='world');p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.study).resolve();choices=verified_choices(root)
    qty=args.quantity or ('Br' if args.region=='airgap' else 'planar_magnitude')
    rows=[m for m in choices['models'] if m['branch']==args.branch and m['region']==args.region and m['quantity']==qty]
    if args.model_id:rows=[m for m in rows if m['id']==args.model_id]
    else:rows=[m for m in rows if m['role']=='primary']
    if len(rows)!=1:raise ValueError('Choose an existing unique model ID/branch/quantity')
    choice=rows[0]
    for key,hashkey in [('inference_module_relative_path','inference_module_sha256'),('model_source_module_relative_path','model_source_module_sha256')]:
        if key in choice and digest(root/choice[key])!=choice[hashkey]:raise ValueError('Frozen original inference source mismatch')
    cache=(root/'../new60_20261014/cache_v1'/f'{args.region}.npz').resolve()
    if digest(cache)!=choice['cache_sha256']:raise ValueError('Input/geometry cache mismatch')
    # Explicit whitelist; field, coeff, target labels are never materialized.
    with np.load(cache,allow_pickle=False) as z:data={k:z[k] for k in INPUT_KEYS}
    if (args.current_rms is None)!=(args.thet_deg is None):raise ValueError('Both current-rms and thet-deg required')
    if args.current_rms is not None:
        if args.case_id is not None:raise ValueError('Choose case ID or native inputs')
        condition=np.array([args.current_rms,args.thet_deg]);matched=np.flatnonzero(np.all(np.isclose(data['conditions'],condition,rtol=0,atol=1e-10),axis=1));cid=int(data['case_ids'][matched[0]]) if len(matched) else None
    else:
        cid=2 if args.case_id is None else args.case_id;index=np.flatnonzero(data['case_ids']==cid)
        if len(index)!=1:raise ValueError('Unknown original case ID')
        condition=data['conditions'][index[0]]
    model=load_model(root/choice['model_relative_path'],choice['model_sha256']);out=predict(model,condition,choice,args.frame)
    frames=np.arange(60) if args.frame is None else np.array([args.frame]);out=out if args.frame is None else out[None,:]
    reference=np.asarray(data['xy'],dtype=np.float64);theta=data['theta'][frames];xy=np.broadcast_to(reference,(len(frames),*reference.shape)).copy()
    if args.region=='rotor' and args.coordinates=='world':
        ang=np.deg2rad(theta);cos=np.cos(ang)[:,None];sin=np.sin(ang)[:,None]
        xy[...,0]=cos*reference[None,:,0]-sin*reference[None,:,1];xy[...,1]=sin*reference[None,:,0]+cos*reference[None,:,1]
    split=json.loads((cache.parent/'split.json').read_text(encoding='utf8'));oldrole=next((name for name in ('train','validation','test') if cid in split['original_ids'][name]),'unseen_input')
    meta=dict(model_id=choice['id'],branch=args.branch,quantity=qty,units='T',xy_units='mm',original_case_id=cid,condition=condition.tolist(),coordinates=args.coordinates,
      score_scope=('training_fit, not heldout accuracy' if cid is not None else 'new input; full60 fitting accuracy is not established here') if args.branch=='fitting_full60' else ('reused_test, not fresh blind test' if oldrole=='test' else oldrole),provided_condition_seen=cid is not None,
      old_prediction_split_role=oldrole,model_sha256=choice['model_sha256'],choices_sha256=digest(root/'both_versions_choices_frozen.json'),frame_indices=frames.tolist(),mechanical_angle='absolute, degree',cloud_magnitude='sqrt(Br^2+Bt^2), Bz unavailable',field_labels_read=False)
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    if destination.suffix.lower()=='.npz':np.savez_compressed(destination,prediction=out,xy=xy,xy_reference=reference,point_id=data['point_id'],theta=theta,time_s=data['time_s'][frames],solver_index=data['solver_index'][frames],metadata_json=np.array(json.dumps(meta,ensure_ascii=False)))
    elif destination.suffix.lower()=='.csv':
        columns=['frame_index','solver_index','time_s','theta_mechanical_absolute_deg','point_id','x_mm','y_mm']+(['Br_T','Bt_T','B_planar_magnitude_T'] if qty=='Br,Bt' else ['Br_T' if qty=='Br' else 'B_planar_magnitude_T'])
        with destination.open('x',encoding='utf8',newline='') as f:
            writer=csv.writer(f);writer.writerow(columns)
            for j,frame in enumerate(frames):
                for i,pid in enumerate(data['point_id']):
                    values=out[j,2*i:2*i+2].tolist() if qty=='Br,Bt' else [float(out[j,i])]
                    if qty=='Br,Bt':values.append(float(np.hypot(*values)))
                    writer.writerow([int(frame),int(data['solver_index'][frame]),float(data['time_s'][frame]),float(theta[j]),int(pid),*xy[j,i].tolist(),*values])
    else:raise ValueError('Output suffix .npz or .csv required')
    destination.with_suffix(destination.suffix+'.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(destination),shape=list(out.shape),model_id=choice['id'],field_labels_read=False),ensure_ascii=False))
if __name__=='__main__':main()
