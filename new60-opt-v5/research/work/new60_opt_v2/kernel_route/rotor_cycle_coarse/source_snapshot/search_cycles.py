"""Broad condition interpolation search, training/validation only, 2 threads."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np
from threadpoolctl import threadpool_limits
import kernel_model as km
from prepare_cycles import digest

HERE=Path(__file__).resolve().parent
COORDS=("native","polar_xy","sqrt_native","square_native","saturated_0.25","saturated_0.5","saturated_1","saturated_2","saturated_4")

def specs():
    for coord,anis,kind,length,ridge,degree in itertools.product(COORDS,(.5,1.,2.),
            ("gaussian","matern32","matern52","inverse_multiquadric","multiquadric"),
            (.3,.6,1.2,2.4,4.8,9.6),(0.,1e-10,1e-7,1e-4,.01),(0,1,2)):
        yield dict(coordinate=coord,anisotropy=anis,kernel=kind,lengthscale=length,ridge=ridge,degree=degree)
    for coord,anis,kind,ridge,degree in itertools.product(COORDS,(.5,1.,2.),("exponential","thin_plate","cubic"),(0.,1e-10,1e-7,1e-4,.01),(1,2)):
        yield dict(coordinate=coord,anisotropy=anis,kernel=kind,lengthscale=1.,ridge=ridge,degree=degree)
    for coord,degree,ridge in itertools.product(COORDS,range(1,8),(0.,1e-10,1e-7,1e-4,.01,1.)):
        yield dict(coordinate=coord,anisotropy=1.,kernel="polynomial_ridge",lengthscale=1.,ridge=ridge,degree=degree)


def fast_metrics(pred,truth,floor,ynorm,d,rank):
    sse=floor+np.sum((pred[:,:rank]-truth[:,:rank])**2,axis=1)+np.sum(truth[:,rank:]**2,axis=1)
    return dict(rmse_T=float(np.sqrt(sse.sum()/(len(sse)*d))),relative_L2_pct=float(100*np.sqrt(sse.sum()/ynorm.sum())),
                worst_case_rmse_T=float(np.sqrt(sse/d).max()))


def full_metrics(prediction,truth,ids):
    e=prediction-truth;squared=e*e
    case=np.sqrt(squared.mean(1));den=float(np.sum(truth*truth))
    return dict(rmse_T=float(np.sqrt(squared.mean())),relative_L2_pct=float(100*np.sqrt(squared.sum()/den)),
                mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),worst_case_rmse_T=float(case.max()),
                per_case={str(case_id):dict(rmse_T=float(case[i]),relative_L2_pct=float(100*np.linalg.norm(e[i])/np.linalg.norm(truth[i])),
                                          max_abs_T=float(np.max(np.abs(e[i])))) for i,case_id in enumerate(ids)})


def run(prepared,output,refine=None):
    prepared=Path(prepared);output=Path(output)
    if output.exists():raise FileExistsError(output)
    source=json.loads((prepared/"preparation.json").read_text(encoding="utf-8"))
    if digest(prepared/"cycles.npz")!=source["prepared_sha256"]:raise ValueError("Prepared source changed")
    with np.load(prepared/"cycles.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
    output.mkdir();snap=output/"source_snapshot";snap.mkdir()
    for name in ("search_cycles.py","kernel_model.py","prepare_cycles.py","protocol.json"):
        shutil.copy2(HERE/name,snap/name)
    start=time.perf_counter();r=a["train_coeff"].shape[1]
    ranks=sorted(set(v for v in (8,16,24,32,r) if v<=r))
    current_scale=float(np.median(a["train_conditions"][a["train_conditions"][:,0]>0,0]))
    transformations={}
    for coord in COORDS:
        values=km.coordinates(a["train_conditions"],coord,current_scale)
        center=values.mean(0);scale=np.maximum(values.std(0),1e-10)
        transformations[coord]=(center,scale)
    if refine:
        prior=json.loads(Path(refine).read_text(encoding="utf-8"))
        seeds=[];seen=set()
        for row in prior["ranking"]:
            s=row["spec"]
            if (s["coordinate"],s["kernel"]) not in seen:
                seen.add((s["coordinate"],s["kernel"]));seeds.append(s)
            if len(seeds)==12:break
        candidates=[]
        for s in seeds:
            if s["kernel"]=="polynomial_ridge":continue
            for factor,anis_factor,ridge,degree in itertools.product((.65,.8,1.,1.25,1.55),(.7,1.,1.4),(0.,1e-12,1e-10,1e-8,1e-6,1e-4),(0,1,2)):
                candidates.append(dict(s,lengthscale=s["lengthscale"]*factor,anisotropy=s["anisotropy"]*anis_factor,ridge=ridge,degree=degree))
        candidates=list({json.dumps(s,sort_keys=True):s for s in candidates}.values())
    else:candidates=list(specs())
    provenance=dict(prepared_source=source,prepared_path=str(prepared.resolve()),
        protocol_sha256=digest(HERE/"protocol.json"),source_sha256={p.name:digest(p) for p in snap.iterdir()},
        threads=2,test_labels_loaded=False,test_metrics_used=False,trial_count=len(candidates),ranks=ranks,
        source_note="All validation coefficient predictions below can be decoded with the immutable prepared mean/basis; no test array materialized",
        refine_from=str(refine) if refine else None,started_utc=datetime.now(timezone.utc).isoformat())
    (output/"provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"candidate_specs.json").write_text(json.dumps(candidates,ensure_ascii=False,indent=2),encoding="utf-8")
    predictions=np.full((len(candidates),9,r),np.nan,dtype=np.float64)
    floor=np.sum(a["validation_residual"]**2,axis=1);ynorm=np.sum(a["validation"]**2,axis=1)
    rows=[];failures=[];best=float("inf")
    with threadpool_limits(limits=2), (output/"search_log.jsonl").open("x",encoding="utf-8") as log:
        for i,s in enumerate(candidates):
            center,scale=transformations[s["coordinate"]]
            x=km.transformed(a["train_conditions"],s["coordinate"],current_scale,center,scale,s["anisotropy"])
            xv=km.transformed(a["validation_conditions"],s["coordinate"],current_scale,center,scale,s["anisotropy"])
            try:
                weights,trend=km.solve_map(x,a["train_coeff"],s["kernel"],s["lengthscale"],s["ridge"],s["degree"])
                pred=km.apply_map(xv,x,weights,trend,s["kernel"],s["lengthscale"],s["degree"])
                if not np.isfinite(pred).all() or np.max(np.abs(pred))>1e8:raise ValueError("Unstable prediction")
                predictions[i]=pred
                for rank in ranks:
                    metrics=fast_metrics(pred,a["val_coeff"],floor,ynorm,a["validation"].shape[1],rank)
                    row=dict(candidate_index=i,rank=rank,spec=s,validation=metrics)
                    rows.append(row)
                    if metrics["rmse_T"]<best:
                        best=metrics["rmse_T"]
                        log.write(json.dumps(dict(event="new_best",**row,elapsed_seconds=time.perf_counter()-start))+"\n");log.flush()
            except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                failures.append(dict(candidate_index=i,spec=s,reason=str(exc)))
            if (i+1)%1000==0:
                print(json.dumps(dict(done=i+1,total=len(candidates),best_validation_RMSE_T=best,elapsed_seconds=time.perf_counter()-start)),flush=True)
        rows.sort(key=lambda q:(q["validation"]["rmse_T"],q["validation"]["worst_case_rmse_T"],q["rank"]))
        shortlist=list(rows[:32]);seen={(q["candidate_index"],q["rank"]) for q in shortlist};families=set()
        for row in rows:
            family=(row["spec"]["coordinate"],row["spec"]["kernel"])
            if family not in families:
                families.add(family)
                key=(row["candidate_index"],row["rank"])
                if key not in seen:shortlist.append(row);seen.add(key)
        for row in shortlist:
            rank=row["rank"];pred=a["mean"]+predictions[row["candidate_index"],:,:rank]@a["basis"][:,:rank].T
            exact=full_metrics(pred,a["validation"],a["validation_case_ids"])
            if not np.isclose(exact["rmse_T"],row["validation"]["rmse_T"],rtol=1e-8,atol=1e-10):raise AssertionError("Analytic SSE mismatch")
            row["validation"]=exact
        best_score=min(q["validation"]["rmse_T"] for q in shortlist)
        eligible=[q for q in shortlist if q["validation"]["rmse_T"]<=1.002*best_score]
        selected=min(eligible,key=lambda q:(q["validation"]["worst_case_rmse_T"],q["validation"]["max_abs_T"],q["validation"]["rmse_T"]))
        s=selected["spec"];rank=selected["rank"];center,scale=transformations[s["coordinate"]]
        x=km.transformed(a["train_conditions"],s["coordinate"],current_scale,center,scale,s["anisotropy"])
        weights,trend=km.solve_map(x,a["train_coeff"][:,:rank],s["kernel"],s["lengthscale"],s["ridge"],s["degree"])
        modelmeta=dict(format="new60_condition_kernel_v2",region=source["region"],quantity=json.loads(str(a["metadata_json"]))["quantity"],
            time_count=source["time_count"],field_dim=source["field_dim"],blocks=[dict(s,current_scale=current_scale,rank=rank)],
            training_original_ids=source["train_original_case_ids"],validation_original_ids=source["validation_original_case_ids"],
            split_sha256=source["split_sha256"],cache_sha256=source["cache_sha256"],selected_by="validation only",test_labels_loaded=False,
            representation="train-only whole-cycle POD",coordinate_caution="I*cos(Thet),I*sin(Thet) are mathematical coordinates, not a dq angle claim")
        model=dict(b0_center=center,b0_input_scale=scale,b0_centers=x,b0_weights=weights,b0_trend=trend,
                   b0_mean=a["mean"],b0_basis=a["basis"][:,:rank],b0_indices=np.arange(len(a["mean"])),
                   metadata_json=np.asarray(json.dumps(modelmeta,ensure_ascii=False)))
        np.savez_compressed(output/"model.npz",**model)
        replay=km.predict(output/"model.npz",a["validation_conditions"])
        expected=a["mean"]+predictions[selected["candidate_index"],:,:rank]@a["basis"][:,:rank].T
        if not np.allclose(replay.reshape(9,-1),expected,rtol=1e-9,atol=1e-9):raise AssertionError("Saved model replay mismatch")
        np.savez_compressed(output/"validation_predictions.npz",prediction=replay,truth=a["validation"].reshape(replay.shape),
            case_ids=a["validation_case_ids"],point_id=a["point_id"],xy=a["xy"],theta=a["theta"],quantity=modelmeta["quantity"])
    np.savez_compressed(output/"all_validation_coefficient_predictions.npz",prediction=predictions,
                        metadata_json=json.dumps(dict(shared_prepared_sha256=source["prepared_sha256"],basis_path=str(prepared/"cycles.npz"),
                         instruction="candidate_index selects coefficient map; rank truncates columns before mean + coeff @ basis.T; reshape [9,60,D]")))
    with (output/"ranking.csv").open("x",encoding="utf-8-sig",newline="") as f:
        fields=["candidate_index","rank","coordinate","anisotropy","kernel","lengthscale","ridge","degree","rmse_T","relative_L2_pct","worst_case_rmse_T","max_abs_T"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:v for k,v in dict(candidate_index=row["candidate_index"],rank=row["rank"],**row["spec"],**row["validation"]).items() if k in fields})
    result=dict(region=source["region"],selected=selected,ranking=rows[:150],fully_decoded_shortlist=shortlist,
                candidates=len(candidates),ranked_candidates=len(rows),failures=failures,elapsed_seconds=time.perf_counter()-start,
                projection_floors=source["validation_projection_floors"],model_sha256=digest(output/"model.npz"),
                all_validation_prediction_sha256=digest(output/"all_validation_coefficient_predictions.npz"),
                train_only_fit=True,test_labels_loaded=False,test_metrics_used=False,
                artifact_notes="All candidate specifications and coefficient predictions retained, full field outputs reconstructed with training-only shared mean/basis")
    (output/"selection.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (output/"model.npz.sha256").write_text(result["model_sha256"]+"\n",encoding="ascii")
    print(json.dumps({k:result[k] for k in ("region","selected","candidates","ranked_candidates","elapsed_seconds","model_sha256")},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--refine",type=Path)
    args=p.parse_args();run(args.prepared,args.output,args.refine)
