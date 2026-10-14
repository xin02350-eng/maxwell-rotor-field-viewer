from pathlib import Path
import contextlib
import json
from point_mean_lowcap_nn import train,sha

root=Path(__file__).resolve().parent;prepared=root/"point_prepared"
truth=root.parents[1]/"new60_opt_v2"/"neural_route"/"rotor_spatial_magnitude"/"prepared"/"validation_truth.npz"
recipes=[("point_low_quad4_skip","cartesian2",[4],"magnitude"),("point_low_cart4_skip_signed","cartesian",[4],"signed"),("point_low_quad2_skip_signed","cartesian2",[2],"signed")]
configs=[dict(name=n+"_seed42",features=f,hidden=h,target_kind=t,seed=42,regularization=1e-8,adam_steps=8000,lbfgs_blocks=40,learning_rate=.003) for n,f,h,t in recipes]
revision=dict(version="v3-point-stage2-budget-revision",reason="First two independent8x8 mean point MLPs overfit: raw validation3.1699%,3.1877%, last train mean<1%. Root requested lower capacity before remaining high-capacity runs.",
    completed=["point_cart8_seed42","point_quad8_seed42"],
    interrupted_incomplete="point_sat150_8_seed42; interrupted before final metrics/weights, does not count as completed NN training",
    not_started=["point_sat300_16_seed42","point_signed_cart8_seed42"],
    revised_candidates=configs,parameters_per_point=[35,28,30],
    unchanged="same raw train42 targets, train-only preprocessing, original9×60×1800 raw validation field, no test/all60-fit label access",
    source_sha256=sha(root/"point_mean_lowcap_nn.py"),original_source_sha256=sha(root/"point_mean_nn.py"),threads=2)
p=root/"candidate_protocol_stage3_lowcap.json"
if not p.exists():p.write_text(json.dumps(revision,indent=2)+"\n",encoding="utf-8")
aborted=root/"point_runs"/"point_sat150_8_seed42"
if aborted.exists() and not (aborted/"result.json").exists() and not (aborted/"aborted.json").exists():
    (aborted/"aborted.json").write_text(json.dumps(dict(reason=revision["reason"],completed=False,weights_saved=False,test_labels_loaded=False))+"\n",encoding="utf-8")
rows=[]
for config in configs:
    out=root/"point_runs"/config["name"]
    if (out/"result.json").exists():rows.append(json.loads((out/"result.json").read_text()));continue
    if out.exists():raise RuntimeError(f"Incomplete run exists {out}")
    (root/"lowcap_progress.json").write_text(json.dumps(dict(active=config["name"],completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
    with (root/(config["name"]+"_stdout.log")).open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):row=train(prepared,truth,out,config)
    rows.append(row);print(json.dumps(dict(name=row["name"],validation=row["validation"]["relative_L2_pct"],train=row["final_optimizer"]["train_mean_relative_L2_pct"],seconds=row["elapsed_seconds"])),flush=True)
(root/"stage3_results.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
(root/"lowcap_progress.json").write_text(json.dumps(dict(active=None,completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
