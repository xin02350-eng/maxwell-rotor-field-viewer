from pathlib import Path
import contextlib
import json
import time
from joint_spatial_nn import train, sha

root=Path(__file__).resolve().parent
prepared=root.parents[1]/"new60_opt_v2"/"neural_route"/"rotor_spatial_magnitude"/"prepared"
recipes=[
    ("shared_cart128",384,"cartesian",[128,128,128],4,2,True,False,1e-7,4000,30),
    ("shared_sat150_128",384,"sat150",[128,128,128],4,2,True,False,1e-7,4000,30),
    ("shared_cart256_r512",512,"cartesian",[256,256,256],6,2,True,False,1e-8,4000,30),
    ("shared_sat300_128_skip",384,"sat300",[128,128,128],4,2,True,True,1e-7,4000,30),
    ("shared_mixed128_reg",384,"mixed",[128,128,128],4,2,True,False,1e-5,4000,30),
]
configs=[dict(name=n+"_seed42",rank=q,features=f,hidden=h,harmonics=harm,interaction=interaction,transients=trans,skip=skip,regularization=reg,adam_steps=steps,lbfgs_blocks=lbfgs,learning_rate=.003,seed=42) for n,q,f,h,harm,interaction,trans,skip,reg,steps,lbfgs in recipes]
protocol=dict(version="v3-joint-MLP-candidate-stage1",candidates=configs,
    pilot_completed="shared_native128_pilot_seed42",adaptation="cartesian/saturated alternatives planned after native pilot; all selection remains validation-only",
    label_access="immutable train42 and validation9 prepared coefficients and validation truth only; excluded test9 and all60 fit labels are never numerically loaded",
    checkpoint_selection="raw complete validation field RMSE in T including fixed spatial projection residual; raw and nonnegative output metrics separately recorded",
    train_preprocessing="train-only POD, condition feature scaler, per-saved-frame coefficient means and common coefficient scales",
    time_policy="joint shared NN weights; t, t^2, harmonic and exponential mathematical features permit unequal endpoints; 60 first-period saved-frame scope only",
    inference="NumPy float64 forward validated independently against float64 PyTorch, original field units and all1800 points",
    representation_sha256=sha(prepared/"representation.npz"),source_sha256=sha(root/"joint_spatial_nn.py"),threads=2)
protocol_path=root/"candidate_protocol_stage1.json"
if not protocol_path.exists():protocol_path.write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
rows=[]
for config in configs:
    out=root/"runs"/config["name"]
    if (out/"result.json").exists():
        row=json.loads((out/"result.json").read_text(encoding="utf-8"));rows.append(row);continue
    if out.exists():raise RuntimeError(f"Incomplete run already exists {out}")
    (root/"progress.json").write_text(json.dumps(dict(active=config["name"],completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
    with (root/(config["name"]+"_stdout.log")).open("w",encoding="utf-8") as log,contextlib.redirect_stdout(log):row=train(prepared,out,config)
    rows.append(row)
    print(json.dumps(dict(name=row["name"],validation=row["validation"]["relative_L2_pct"],train=row["train"]["relative_L2_pct"],seconds=row["elapsed_seconds"])),flush=True)
(root/"stage1_results.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
(root/"progress.json").write_text(json.dumps(dict(active=None,completed=[r["name"] for r in rows]))+"\n",encoding="utf-8")
