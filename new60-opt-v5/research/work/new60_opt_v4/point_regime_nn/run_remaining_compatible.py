from pathlib import Path
import contextlib
import json
from regime_point_nn_softplus_consistent import train,sha

root=Path(__file__).resolve().parent;workspace=root.parents[2]
prepared=workspace/"work/new60_opt_v3/joint_neural/point_prepared";truth=workspace/"work/new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared/validation_truth.npz"
recipes=[("regime_hinge2_principal_retry1","softplus_hinge",2,"principal",42),
         ("regime_sigmoid_affine2_random","sigmoid_affine",2,"random",17),
         ("regime_hinge3_random","softplus_hinge",3,"random",17)]
configs=[dict(name=n+f"_seed{s}",expert_kind=k,gates=g,initialization=i,seed=s,sharpness=3.,gate_prior=1e-6,expert_regularization=1e-7,adam_steps=5000,lbfgs_blocks=20,learning_rate=.005,gate_learning_rate=.001) for n,k,g,i,s in recipes]
repair=dict(reason="PyTorch softplus defaults to linear z when z>20; NumPy logaddexp retains an exponentially small term. NumPy inference is corrected to exactly match the training operation's threshold20. No learned architecture/config/data/math during gradient optimization changes.",
    old_source_sha256=sha(root/"regime_point_nn.py"),new_source_sha256=sha(root/"regime_point_nn_softplus_consistent.py"),failed_attempt="regime_hinge2_principal_seed42",retry="regime_hinge2_principal_retry1_seed42",
    failed_attempt_gradient_budget_completed=True,failed_attempt_verified_candidate=False,failed_attempt_has_full_saved_checkpoint_csv=False,
    label_exclusion="all original9 test/all60-fit labels stay excluded",remaining_candidates=configs)
(root/"runtime_softplus_repair_binding.json").write_text(json.dumps(repair,indent=2)+"\n",encoding="utf-8")
failed=root/"runs/regime_hinge2_principal_seed42"
if not (failed/"failed.json").exists():(failed/"failed.json").write_text(json.dumps(repair,indent=2)+"\n",encoding="utf-8")
rows=[]
for config in configs:
    out=root/"runs"/config["name"]
    if (out/"result.json").exists():rows.append(json.loads((out/"result.json").read_text(encoding="utf-8")));continue
    if out.exists():raise RuntimeError(f"Existing incomplete run {out}")
    (root/"remaining_progress.json").write_text(json.dumps(dict(active=config["name"],completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
    with (root/(config["name"]+"_stdout.log")).open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):row=train(prepared,truth,out,config)
    rows.append(row);print(json.dumps(dict(name=row["name"],validation=row["validation"]["relative_L2_pct"],max_abs=row["validation"]["max_abs_T"],seconds=row["elapsed_seconds"])),flush=True)
(root/"stage1_remaining_results.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(root/"remaining_progress.json").write_text(json.dumps(dict(active=None,completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
