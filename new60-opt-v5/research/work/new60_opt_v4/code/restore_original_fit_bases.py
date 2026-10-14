"""Restore original V2 NN baselines from calibrated models plus tiny old heads.

The final all60 model retains every original hidden layer and representation.
Only its last linear head and metadata changed. This lossless restoration
avoids packaging another121MB of duplicated hidden arrays. SHA verification
is mandatory; restored models are for provenance/retraining, not inference.
"""
from pathlib import Path
import argparse, hashlib, io, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
def sha_data(data):return hashlib.sha256(data).hexdigest()
def reconstruct(region):
    payload=ROOT/'provenance_restore'/region/'base_head_restore.npz'
    with np.load(payload,allow_pickle=False) as z:
        head={k:z[k] for k in z.files};meta=json.loads(str(head.pop('restore_metadata_json')))
    with np.load(WORK/'new60_opt_v3/fitting_full60'/region/'model.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
    for k,v in head.items():model[k]=v
    buffer=io.BytesIO();np.savez_compressed(buffer,**model);data=buffer.getvalue()
    if sha_data(data)!=meta['original_model_sha256'] or len(data)!=meta['original_model_bytes']:raise AssertionError('Lossless restoration SHA mismatch')
    return data,meta
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args();rows=[]
    for region in ['rotor','stator','airgap']:
        data,m=reconstruct(region);dest=WORK/m['original_model_relative_path']
        if not a.verify_only:
            if dest.exists():
                if sha_data(dest.read_bytes())!=m['original_model_sha256']:raise FileExistsError(dest)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(data)
        rows.append(dict(region=region,bytes=len(data),sha256=m['original_model_sha256'],lossless_verified=True))
    print(json.dumps(dict(all_checks_pass=True,verify_only=a.verify_only,models=rows)))
if __name__=='__main__':main()
