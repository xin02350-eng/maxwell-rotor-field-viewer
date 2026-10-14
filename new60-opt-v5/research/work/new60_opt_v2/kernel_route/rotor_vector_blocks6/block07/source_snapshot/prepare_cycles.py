"""Read TRAIN/VALIDATION field rows only; fit cycle POD on TRAIN only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/"new60_20261014/cache_v1"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def read_rows(path,key,indices):
    indices=list(indices);locations={v:i for i,v in enumerate(indices)}
    with zipfile.ZipFile(path) as z,z.open(key+".npy") as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        if fortran or len(shape)!=3 or shape[0]!=60 or dtype.kind!="f":raise ValueError("Expected contiguous 60-case field")
        size=int(np.prod(shape[1:]))*dtype.itemsize
        out=np.empty((len(indices),*shape[1:]),dtype=np.float64)
        for case in range(shape[0]):
            if case in locations:
                raw=f.read(size)
                if len(raw)!=size:raise ValueError("Truncated row")
                out[locations[case]]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
            else:f.seek(size,1)
    return out

def main(region,magnitude=False):
    output=HERE/("prepared_"+region+("_magnitude" if magnitude else ""))
    if output.exists():raise FileExistsError(output)
    split=json.loads((SOURCE/"split.json").read_text(encoding="utf-8-sig"))
    indices=split["train"]+split["validation"]
    if len(indices)!=51 or set(indices)&set(split["test"]):raise ValueError("Invalid selection")
    cache=SOURCE/(region+".npz")
    with np.load(cache,allow_pickle=False) as z:
        arrays={k:z[k] for k in ("conditions","case_ids","point_id","xy","theta","metadata_json")}
    meta=json.loads(str(arrays["metadata_json"]))
    if arrays["case_ids"].tolist()!=split["case_ids"]:raise ValueError("Case order mismatch")
    selected=read_rows(cache,"field",indices)
    if magnitude:
        if meta["component_count"]!=2:raise ValueError("Magnitude requires both signed Br and Bt")
        selected=np.hypot(selected[...,0::2],selected[...,1::2])
        meta.update(quantity="|B|",components=["|B|"],component_count=1,layout="scalar",
                    magnitude_definition="hypot(Br,Bt) from selected TRAIN/VALIDATION labels only")
        arrays["metadata_json"]=np.asarray(json.dumps(meta,ensure_ascii=False))
    train=selected[:42].reshape(42,-1);validation=selected[42:].reshape(9,-1)
    with threadpool_limits(limits=2):
        mean=train.mean(0);centered=train-mean
        eig,vectors=np.linalg.eigh(centered@centered.T)
        order=np.argsort(eig)[::-1];eig=eig[order];vectors=vectors[:,order]
        keep=eig>eig[0]*1e-12;eig=eig[keep];vectors=vectors[:,keep]
        basis=centered.T@(vectors/np.sqrt(eig))
        basis,_=np.linalg.qr(basis,mode="reduced")
        train_coeff=centered@basis;val_coeff=(validation-mean)@basis
        norm=np.sum(validation*validation,axis=1)
        residual=(validation-mean)-val_coeff@basis.T
        floors=[]
        for rank in sorted(set([r for r in (8,16,24,32,len(eig)) if r<=len(eig)])):
            sse=np.sum(residual*residual,axis=1)+np.sum(val_coeff[:,rank:]**2,axis=1)
            floors.append(dict(rank=rank,rmse_T=float(np.sqrt(sse.sum()/validation.size)),
                               relative_L2_pct=float(100*np.sqrt(sse.sum()/norm.sum())),
                               worst_case_rmse_T=float(np.sqrt(sse/validation.shape[1]).max())))
    output.mkdir();shutil.copy2(__file__,output/"prepare_cycles.py")
    np.savez_compressed(output/"cycles.npz",mean=mean,basis=basis,train_coeff=train_coeff,val_coeff=val_coeff,
        validation=validation,validation_residual=residual,train_conditions=arrays["conditions"][split["train"]],
        validation_conditions=arrays["conditions"][split["validation"]],train_case_ids=arrays["case_ids"][split["train"]],
        validation_case_ids=arrays["case_ids"][split["validation"]],point_id=arrays["point_id"],xy=arrays["xy"],theta=arrays["theta"],
        eig=eig,metadata_json=arrays["metadata_json"])
    info=dict(region=region,time_count=selected.shape[1],field_dim=selected.shape[2],train_count=42,validation_count=9,
        rank=len(eig),validation_projection_floors=floors,basis_orthogonality_max=float(np.max(np.abs(basis.T@basis-np.eye(len(eig))))),
        cache_path=str(cache),cache_sha256=digest(cache),split_path=str(SOURCE/"split.json"),split_sha256=digest(SOURCE/"split.json"),
        train_original_case_ids=arrays["case_ids"][split["train"]].tolist(),validation_original_case_ids=arrays["case_ids"][split["validation"]].tolist(),
        excluded_test_original_case_ids=split["original_ids"]["test"],test_labels_loaded=False,old_test_metrics_read=False,
        basis_fit="training cases only; complete 60-frame fields",source_sha256=digest(__file__),
        prepared_sha256=digest(output/"cycles.npz"),prepared_utc=datetime.now(timezone.utc).isoformat())
    (output/"preparation.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--region",choices=("rotor","stator","airgap"),required=True);p.add_argument("--magnitude",action="store_true")
    args=p.parse_args();main(args.region,args.magnitude)
