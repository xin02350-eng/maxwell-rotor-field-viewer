from pathlib import Path
import contextlib
import json
from point_mean_nn import train,sha

root=Path(__file__).resolve().parent
prepared=root/"point_prepared"
truth=root.parents[1]/"new60_opt_v2"/"neural_route"/"rotor_spatial_magnitude"/"prepared"/"validation_truth.npz"
recipes=[("point_cart8","cartesian",[8,8],"magnitude"),("point_quad8","cartesian2",[8,8],"magnitude"),("point_sat150_8","sat150",[8,8],"magnitude"),("point_sat300_16","sat300",[16,16],"magnitude"),("point_signed_cart8","cartesian",[8,8],"signed")]
configs=[dict(name=n+"_seed42",features=f,hidden=h,target_kind=target,seed=42,regularization=1e-8,adam_steps=8000,lbfgs_blocks=40,learning_rate=.003) for n,f,h,target in recipes]
protocol=dict(version="v3-independent-point-mean-stage2",candidates=configs,
    diagnosis="Most old validation SSE is across-condition mean-field error; train temporal fluctuation has3.287%relativeL2 (0.1081% squared-energy ratio), so preserve old neural zero-mean dynamics",
    mean_targets="raw source train42 magnetic magnitude temporal means, or signed mean components;1800 separate point MLPs",
    dynamic_scope="immutable V2 pure neural signed-field model and its zero-temporal-mean outputs; a scalar is calibrated once on train42 only (result1.0), no validation choice of temporal mixing",
    loss="raw T train mean SSE, no point weighting and no cropped geometry; all1800 points equally weighted",
    checkpoint="minimum raw original T full9×60×1800 validation field RMSE; no test labels and no all60 fitting labels",
    source_sha256=sha(root/"point_mean_nn.py"),prepared_identity_sha256=sha(prepared/"identity.json"),prepared_data_sha256=sha(prepared/"mean_data.npz"),threads=2)
p=root/"candidate_protocol_stage2.json"
if not p.exists():p.write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
rows=[]
for config in configs:
    out=root/"point_runs"/config["name"]
    if (out/"result.json").exists():rows.append(json.loads((out/"result.json").read_text()));continue
    if out.exists():raise RuntimeError(f"Incomplete run exists {out}")
    (root/"point_progress.json").write_text(json.dumps(dict(active=config["name"],completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
    with (root/(config["name"]+"_stdout.log")).open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):row=train(prepared,truth,out,config)
    rows.append(row);print(json.dumps(dict(name=row["name"],validation=row["validation"]["relative_L2_pct"],train=row["final_optimizer"]["train_mean_relative_L2_pct"],seconds=row["elapsed_seconds"])),flush=True)
(root/"stage2_results.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
(root/"point_progress.json").write_text(json.dumps(dict(active=None,completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
