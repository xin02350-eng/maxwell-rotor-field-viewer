"""Read the original compact ZIP without extracting its 1 GB of redundant fields."""
import argparse, csv, hashlib, io, json, zipfile
from collections import Counter
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('archive', help='Path to twelve_conditions_8500_compact.zip')
    ap.add_argument('--output', default=str(Path(__file__).resolve().parent.parent/'work/rotor_dataset.npz'))
    args=ap.parse_args(); out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        pts=json.loads(z.read('points.json'));manifest=json.loads(z.read('manifest.json'))
        cases=sorted(manifest['cases'], key=lambda c:c['dataset_case_id'])
        mask=np.array([p['region']=='rotor' for p in pts]);rp=[p for p in pts if p['region']=='rotor']
        ids=np.array([p['point_id'] for p in rp]);xy=np.array([[p['x_ref_mm'],p['y_ref_mm']] for p in rp])
        assert len(rp)==2500
        ys=[];bs=[];cond=[];checks=[]
        for c in cases:
            raw=z.read(c['npz_path']);sha=hashlib.sha256(raw).hexdigest()
            if sha!=c['npz_sha256']:raise ValueError('SHA256 mismatch: '+c['npz_path'])
            with np.load(io.BytesIO(raw),allow_pickle=False) as d:
                assert np.array_equal(d['point_id'],[p['point_id'] for p in pts])
                assert d['columns'].tolist()==['Bx','By','Bz','Br','Bt']
                b=d['B'][:,mask,:];assert np.isfinite(b).all()
                mag=np.linalg.norm(b[:,:,:3],axis=-1)
                ys.append(mag[:300].astype('float32'));bs.append(b[:300,:,3:5].astype('float32'))
                cond.append([c['Irms_A'],c['phase_angle_deg']])
                th=d['theta_relative_deg'][:300];ab=d['theta_deg'][:300]
                assert np.allclose(th,np.arange(300)*1.2)
                checks.append({'case_id':c['dataset_case_id'],'verified_sha256':sha,
                               'closure_max_abs_T':float(np.max(abs(mag[-1]-mag[0])))})
                print(f'Verified case {c["dataset_case_id"]}: {c["Irms_A"]} A RMS, {c["phase_angle_deg"]} deg')
        np.savez_compressed(out,Bmag=np.stack(ys),BrBt=np.stack(bs),conditions=np.array(cond),
                            xy=xy,point_id=ids,theta=th,theta_absolute=ab)
        audit={'archive':str(Path(args.archive).resolve()),'counts':dict(Counter(p['region'] for p in pts)),
               'retained_shape':[12,300,2500],'unit_B':'T','unit_xy':'mm','current':'A RMS','cases':checks}
        out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Saved',out)

if __name__=='__main__':main()
