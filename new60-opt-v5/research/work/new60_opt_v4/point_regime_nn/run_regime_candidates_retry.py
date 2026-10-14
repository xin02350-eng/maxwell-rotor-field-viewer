from pathlib import Path
import contextlib
import json
from regime_point_nn import train,sha

root=Path(__file__).resolve().parent;workspace=root.parents[2]
prepared=workspace/"work/new60_opt_v3/joint_neural/point_prepared"
truth=workspace/"work/new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared/validation_truth.npz"
recipes=[
    ("regime_sigmoid_affine2_principal_retry1", "sigmoid_affine",2,"principal",42,3.,1e-6),
    ("regime_sigmoid_affine3_principal", "sigmoid_affine",3,"principal",42,3.,1e-6),
    ("regime_tanh3_principal", "tanh_constant",3,"principal",42,2.,1e-6),
    ("regime_hinge2_principal", "softplus_hinge",2,"principal",42,3.,1e-6),
    ("regime_sigmoid_affine2_random", "sigmoid_affine",2,"random",17,3.,1e-6),
    ("regime_hinge3_random", "softplus_hinge",3,"random",17,3.,1e-6),
]
configs=[dict(name=n+f"_seed{s}",expert_kind=k,gates=g,initialization=i,seed=s,sharpness=sharp,gate_prior=gp,expert_regularization=1e-7,adam_steps=5000,lbfgs_blocks=20,learning_rate=.005,gate_learning_rate=.001) for n,k,g,i,s,sharp,gp in recipes]
protocol=dict(version="v4-train-only-point-regime-saturation-neural-protocol",planned_completed_NN_budget=8,first_stage_candidates=configs,
    confirmation_policy="Up to2 additional different-seed full NN training confirmations of improved structures; discuss with root before budget extension. Bagging/mixtures are reported separately and do not count as NN training.",
    problem="Large errors are mainly condition-dependent static mean responses; strong adjacent-condition response change may indicate saturation/regimes but is not established simulation noise.",
    label_scope="Train42/validation9 only, preserving all60 frames/all1800 points. Original nine test cases have already been seen3 rounds by overall project; this route never loads/scored their labels, and never loads all60-fit labels.",
    input="native [I_RMS, Thet_deg], transformed into mathematical IcosThet/IsinThet features; no claim these are physical dq-axis currents; no extra divide-by2",
    targets="train-only temporal mean signedBr/Bt per fixed spatial point; old immutable signed neural model contributes its zero-temporal-mean dynamic output, train-only scale1.0; output magnetic magnitude throughhypot",
    architectures="trainable signed affine skip +2/3 learned sigmoid affine experts, tanh saturation offsets, or softplus piecewise slope changes;16–33 parameters per point",
    initialization="OLS/ridge baseline fitted to train42; principal gate direction from train-only bias-field response gradient or seeded random direction; thresholds from train-only projected input quantiles",
    optimization="PyTorch real gradient Adam5000 plus200 LBFGS maximum; checkpoints every50Adam steps and10LBFGS iterations. Complete epochs and source/model/provenance hashes retained.",
    checkpoint="minimum raw originalTesla unweighted full9×60×1800 validation magnitude RMSE; maxAbs/worstcase/worstpoint separately retained. No cropped points, validation point weights, or test-selected recipes.",
    resources="CPU2 threads; no copied truth/prepared datasets; modelweights/logs retained, only final best validationpred persisted",
    prepared_identity_sha256=sha(prepared/"identity.json"),prepared_means_sha256=sha(prepared/"mean_data.npz"),validation_truth_sha256=sha(truth),inference_training_source_sha256=sha(root/"regime_point_nn.py"))
pp=root/"candidate_protocol.json"
if not pp.exists():pp.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");(root/"candidate_protocol.sha256").write_text(sha(pp)+"\n")
rows=[]
for config in configs:
    out=root/"runs"/config["name"]
    if (out/"result.json").exists():rows.append(json.loads((out/"result.json").read_text(encoding="utf-8")));continue
    if out.exists():raise RuntimeError(f"Existing incomplete run {out}")
    (root/"progress.json").write_text(json.dumps(dict(active=config["name"],completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
    with (root/(config["name"]+"_stdout.log")).open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):row=train(prepared,truth,out,config)
    rows.append(row);print(json.dumps(dict(name=row["name"],validation=row["validation"]["relative_L2_pct"],max_abs=row["validation"]["max_abs_T"],seconds=row["elapsed_seconds"])),flush=True)
(root/"stage1_results.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(root/"progress.json").write_text(json.dumps(dict(active=None,completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
