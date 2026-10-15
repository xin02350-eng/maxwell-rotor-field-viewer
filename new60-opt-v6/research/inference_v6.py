"""RBF experiment inference. Requires the V5 delivery as adjacent folders.

No truth/cached field arrays are opened. Native RMS current and angle select
all 60 stored positions; frame_index selects one stored frame, not a new angle.
"""
from pathlib import Path
import json, numpy as np
import rbf_pointwise as r

ROOT=Path(__file__).resolve().parent

def raw(model,c):
    path=Path(model)
    if r.sha(path)!=path.with_suffix('.npz.sha256').read_text().strip():raise ValueError('RBF SHA mismatch')
    with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    meta=json.loads(str(m['metadata_json']));c=np.atleast_2d(c);out=np.empty((len(m['point_id']),len(c),120 if meta['route']=='signed' else 60))
    for i,spec in enumerate(meta['groups']):
        pre=f'g{i}_';g={k[len(pre):]:v for k,v in m.items() if k.startswith(pre)};x=r.mapped(g,c,spec['anisotropy'])
        out[g['points']]=r.kernel(x,g['centers'],spec['kernel'],spec['lengthscale'])@g['weights']+r.polynomial(x,spec['degree'])@g['trend']
    return out,meta

def old_baseline(region,c,scope):
    if scope=='original42' or region!='rotor':return r.baseline(region,c)
    import sys
    sys.path.insert(0,str(ROOT.parent/'new60_opt_v5/code'));import inference_numpy_v5 as safe
    v5=ROOT.parent/'new60_opt_v5';freeze=safe.read_freeze(v5)
    return safe.predict(v5,next(v for v in freeze['models'] if v['id']=='prediction_rotor_final51_magnitude'),c)

def predict(region,conditions,scope='original42',frame_index=None):
    if region not in ['rotor','stator','airgap'] or scope not in ['original42','final51']:raise ValueError('Unknown region/scope')
    c=np.asarray(conditions,dtype=float);single=c.ndim==1;c=np.atleast_2d(c)
    if c.shape[1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Finite native RMS>=0 and Thet required')
    if frame_index is not None and (type(frame_index) is not int or not 0<=frame_index<60):raise ValueError('frame_index0..59 required')
    pp=ROOT/'rbf_choices_pretest.json'
    if r.sha(pp)!=pp.with_suffix('.json.sha256').read_text().strip():raise ValueError('Recipe SHA mismatch')
    choice=json.loads(pp.read_text())['regions'][region];base=old_baseline(region,c,scope);recipe=choice['recipe'];alpha=recipe['alpha']
    if alpha==0.:out=base
    else:
        folder=ROOT/choice['study'] if scope=='original42' else ROOT/'final51'/region
        if recipe['route']=='pointLOO_shrunk50':
            a,meta=raw(folder/'pointLOO.npz',c);b,_=raw(folder/'shared.npz',c);q=r.quantity(.5*a+.5*b,meta['route'])
        else:
            a,meta=raw(folder/(recipe['route']+'.npz'),c);q=r.quantity(a,meta['route'])
        out=(1-alpha)*base+alpha*q
    if frame_index is not None:out=out[:,frame_index]
    if not np.isfinite(out).all():raise ValueError('Nonfinite output')
    return out[0] if single else out

if __name__=='__main__':
    import argparse
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--region',choices=['rotor','stator','airgap'],required=True);p.add_argument('--current',type=float,required=True);p.add_argument('--angle',type=float,required=True);p.add_argument('--scope',choices=['original42','final51'],default='original42');p.add_argument('--frame',type=int);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=2):v=predict(a.region,[a.current,a.angle],a.scope,a.frame)
    np.savez_compressed(a.output,prediction=v,current_RMS_A=np.array(a.current),native_Thet_deg=np.array(a.angle),frame_index=np.array(-1 if a.frame is None else a.frame),quantity=np.array('Br' if a.region=='airgap' else 'planar_magnitude'),units=np.array('T'))
