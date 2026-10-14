"""Time/component blocks with separate train-only POD and condition maps."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np
from threadpoolctl import threadpool_limits
import kernel_model as km
from prepare_cycles import HERE,SOURCE,digest,read_rows
from search_cycles import run,full_metrics


def main(region,output,coarse,time_blocks=6,magnitude=False):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir();shutil.copy2(__file__,output/"search_blocks.py")
    split=json.loads((SOURCE/"split.json").read_text(encoding="utf-8-sig"))
    cache=SOURCE/(region+".npz")
    with np.load(cache,allow_pickle=False) as z:
        a={k:z[k] for k in ("conditions","case_ids","point_id","xy","theta","metadata_json")}
    meta=json.loads(str(a["metadata_json"]))
    if set(split["train"]+split["validation"])&set(split["test"]):raise ValueError("Split overlap")
    fields=read_rows(cache,"field",split["train"]+split["validation"])
    if magnitude:
        if meta["component_count"]!=2:raise ValueError("No complete vector magnitude available")
        fields=np.hypot(fields[...,0::2],fields[...,1::2])
        meta.update(quantity="|B|",components=["|B|"],component_count=1,layout="scalar")
    t,d=fields.shape[1:];flat=fields.reshape(51,-1);k=meta["component_count"]
    prior=json.loads(Path(coarse).read_text(encoding="utf-8"));candidates=[];seen=set()
    for row in sorted(prior["fully_decoded_shortlist"],key=lambda q:q["validation"]["rmse_T"]):
        s=row["spec"];key=(s["coordinate"],s["kernel"])
        if key in seen:continue
        seen.add(key);candidates.append(s)
        if s["kernel"]=="polynomial_ridge":continue
        ridges=sorted(set([0.,1e-5,.001,.01,.1,s["ridge"]]))
        for lengthfactor,ridge,degree in itertools.product((.7,1.,1.4),ridges,(0,1,2)):
            candidates.append(dict(s,lengthscale=s["lengthscale"]*lengthfactor,ridge=ridge,degree=degree))
    candidates=list({json.dumps(row,sort_keys=True):row for row in candidates}.values())
    pool=output/"candidate_pool.json";pool.write_text(json.dumps(candidates,indent=2),encoding="utf-8")
    plan=dict(region=region,quantity=meta["quantity"],time_blocks=time_blocks,separate_components=True,
        condition_candidates=len(candidates),blocks=time_blocks*k,source_coarse=str(coarse),source_coarse_sha256=digest(coarse),
        cache_sha256=digest(cache),split_sha256=digest(SOURCE/"split.json"),
        selection="Each block independently minimizes validation RMSE; within 0.2% prioritize worst-case/maxabs. Composite is then evaluated, no test access.",
        predeclared_utc=datetime.now(timezone.utc).isoformat(),test_labels_loaded=False)
    (output/"block_protocol.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    block_indices=[]
    for frames in np.array_split(np.arange(t),time_blocks):
        for component in range(k):
            block_indices.append(np.concatenate([frame*d+np.arange(component,d,k) for frame in frames]))
    floors=np.zeros(9);start=time.perf_counter();parts=[]
    for i,idx in enumerate(block_indices):
        prepared=output/f"prepared_block{i:02d}";prepared.mkdir()
        with threadpool_limits(limits=2):
            train=flat[:42,idx];validation=flat[42:,idx];mean=train.mean(0);centered=train-mean
            eigen,vectors=np.linalg.eigh(centered@centered.T);order=np.argsort(eigen)[::-1];eigen=eigen[order];vectors=vectors[:,order]
            keep=eigen>eigen[0]*1e-12;eigen=eigen[keep];vectors=vectors[:,keep]
            basis,_=np.linalg.qr(centered.T@(vectors/np.sqrt(eigen)),mode="reduced")
            tr=centered@basis;va=(validation-mean)@basis;residual=validation-mean-va@basis.T
            floors+=np.sum(residual*residual,axis=1)
        blockmeta=dict(meta,components=[meta["components"][i%k]],component_count=1,layout="scalar",quantity=meta["components"][i%k])
        np.savez_compressed(prepared/"cycles.npz",mean=mean,basis=basis,train_coeff=tr,val_coeff=va,validation=validation,
            validation_residual=residual,train_conditions=a["conditions"][split["train"]],validation_conditions=a["conditions"][split["validation"]],
            train_case_ids=a["case_ids"][split["train"]],validation_case_ids=a["case_ids"][split["validation"]],point_id=a["point_id"],
            xy=a["xy"],theta=a["theta"],eig=eigen,metadata_json=np.asarray(json.dumps(blockmeta)))
        info=dict(region=region,time_count=1,field_dim=len(idx),rank=len(eigen),train_count=42,validation_count=9,
            cache_path=str(cache),cache_sha256=plan["cache_sha256"],split_path=str(SOURCE/"split.json"),split_sha256=plan["split_sha256"],
            train_original_case_ids=split["original_ids"]["train"],validation_original_case_ids=split["original_ids"]["validation"],
            excluded_test_original_case_ids=split["original_ids"]["test"],validation_projection_floors=[dict(rank=len(eigen),
                rmse_T=float(np.sqrt(np.mean(residual**2))),relative_L2_pct=float(100*np.linalg.norm(residual)/np.linalg.norm(validation)))],
            test_labels_loaded=False,prepared_sha256=digest(prepared/"cycles.npz"),global_flat_indices=idx.tolist())
        (prepared/"preparation.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
        parts.append(prepared)
    projection=dict(rmse_T=float(np.sqrt(floors.sum()/flat[42:].size)),
                    relative_L2_pct=float(100*np.sqrt(floors.sum()/np.sum(flat[42:]**2))),
                    worst_case_rmse_T=float(np.sqrt(floors/(t*d)).max()))
    (output/"block_projection_floor.json").write_text(json.dumps(projection,indent=2),encoding="utf-8")
    print(json.dumps(dict(block_projection_floor=projection,blocks=len(parts),candidates_per_block=len(candidates))),flush=True)
    combined={};blocks=[];chosen=[]
    for i,(prepared,idx) in enumerate(zip(parts,block_indices)):
        trial=output/f"block{i:02d}";run(prepared,trial,pool=pool)
        selection=json.loads((trial/"selection.json").read_text(encoding="utf-8"));chosen.append(selection["selected"])
        with np.load(trial/"model.npz",allow_pickle=False) as z:model={name:z[name] for name in z.files}
        blockmeta=json.loads(str(model["metadata_json"]));blocks.append(blockmeta["blocks"][0])
        for name,value in model.items():
            if name.startswith("b0_"):combined[f"b{i}_"+name[3:]]=value
        combined[f"b{i}_indices"]=idx
    modelmeta=dict(format="new60_condition_kernel_v2",region=region,quantity=meta["quantity"],time_count=t,field_dim=d,blocks=blocks,
        train_original_ids=split["original_ids"]["train"],validation_original_ids=split["original_ids"]["validation"],
        cache_sha256=plan["cache_sha256"],split_sha256=plan["split_sha256"],test_labels_loaded=False,
        representation="Per-time/component block train-only POD; independent condition kernel per block",coordinate_caution="No dq interpretation")
    combined["metadata_json"]=np.asarray(json.dumps(modelmeta,ensure_ascii=False))
    np.savez_compressed(output/"model.npz",**combined)
    with threadpool_limits(limits=2):pred=km.predict(output/"model.npz",a["conditions"][split["validation"]])
    validation=full_metrics(pred.reshape(9,-1),flat[42:],a["case_ids"][split["validation"]])
    np.savez_compressed(output/"validation_predictions.npz",prediction=pred,truth=fields[42:],case_ids=a["case_ids"][split["validation"]],
                        point_id=a["point_id"],xy=a["xy"],theta=a["theta"],quantity=meta["quantity"])
    report=dict(**plan,selected_blocks=chosen,validation=validation,projection_floor=projection,model_sha256=digest(output/"model.npz"),
        elapsed_seconds=time.perf_counter()-start,selection_frozen=True,source_sha256=digest(__file__))
    (output/"selection.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"model.npz.sha256").write_text(report["model_sha256"]+"\n",encoding="ascii")
    print(json.dumps(dict(region=region,quantity=meta["quantity"],validation=validation,model_sha256=report["model_sha256"]),ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--region",required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--coarse",type=Path,required=True);p.add_argument("--time-blocks",type=int,default=6);p.add_argument("--magnitude",action="store_true")
    args=p.parse_args();main(args.region,args.output,args.coarse,args.time_blocks,args.magnitude)
