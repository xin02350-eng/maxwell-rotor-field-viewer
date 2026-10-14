from pathlib import Path
import contextlib
import json
from regime_quadratic_skip_nn import train,sha

root=Path(__file__).resolve().parent;workspace=root.parents[2]
prepared=workspace/"work/new60_opt_v3/joint_neural/point_prepared";truth=workspace/"work/new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared/validation_truth.npz"
configs=[dict(name=f"regime_quadratic_sigmoid_affine{g}_principal_seed42",expert_kind="sigmoid_affine",gates=g,initialization="principal",seed=42,sharpness=3.,gate_prior=1e-6,expert_regularization=1e-7,adam_steps=5000,lbfgs_blocks=20,learning_rate=.005,gate_learning_rate=.001) for g in (2,3)]
revision=dict(version="v4-point-regime-budget-revision-within8-completed-NN-runs",reason="Principal3-gate affine NN is a new max-error Pareto candidate (validation2.724921%/max0.666528T), but mean response still has2.6%+error. Smooth quadratic skip may reduce the nonlinear baseline bias without widening/growing hidden experts.",
    authorization="Root explicitly approved last2 planned NN slots for quadraticCartesian skip+2/3 affine experts, replacing seed-confirmation slots after validation-only observations; no test feedback.",
    candidates=configs,parameters_per_point=[30,39],modified_init="train42 quadratic ridgeOLS coefficients; gate direction from train-only zero-current baseline response derivative; all30/39 parameters per point gradient-trained",
    unchanged="Same42 training and9 validation, old signed NN zero-mean dynamics, raw all9×60×1800 validation magnitude RMSE checkpoint, no crops and no validation point weighting",
    source_sha256=sha(root/"regime_quadratic_skip_nn.py"),prepared_identity_sha256=sha(prepared/"identity.json"),prepared_means_sha256=sha(prepared/"mean_data.npz"),test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
pp=root/"candidate_protocol_last2_revision.json"
if not pp.exists():pp.write_text(json.dumps(revision,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(root/"candidate_protocol_last2_revision.sha256").write_text(sha(pp)+"\n")
rows=[]
for config in configs:
    out=root/"runs"/config["name"]
    if (out/"result.json").exists():rows.append(json.loads((out/"result.json").read_text(encoding="utf-8")));continue
    if out.exists():raise RuntimeError(f"Existing incomplete run {out}")
    (root/"quadratic_progress.json").write_text(json.dumps(dict(active=config["name"],completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
    with (root/(config["name"]+"_stdout.log")).open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):row=train(prepared,truth,out,config)
    rows.append(row);print(json.dumps(dict(name=row["name"],validation=row["validation"]["relative_L2_pct"],max_abs=row["validation"]["max_abs_T"],seconds=row["elapsed_seconds"])),flush=True)
(root/"stage2_results.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(root/"quadratic_progress.json").write_text(json.dumps(dict(active=None,completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
