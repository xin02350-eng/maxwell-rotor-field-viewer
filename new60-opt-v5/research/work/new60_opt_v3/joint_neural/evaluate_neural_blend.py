from pathlib import Path
import json
import numpy as np
from joint_spatial_nn import exact_metrics
from pure_neural_blend_v3 import sha,predict

root=Path(__file__).resolve().parent;out=root/"pure_neural_blend";out.mkdir(exist_ok=False)
point=root/"point_runs"/"point_low_quad2_skip_signed_seed42"/"model.npz"
protocol=dict(version="v3-pure-neural-global-blend-validation-only",new_point_model=str(point.relative_to(root)),new_point_model_sha256=sha(point),
    old_model="embedded immutable V2 direct_mixed_seed42 signed pure neural cycle",
    candidates=[dict(route=r,alpha=a) for r in ("scalar_magnitude","signed_vector") for a in (0.,.25,.5,.75,1.)],
    selection="minimum original T raw full-validation field RMSE; all9 cases ×60 frames ×1800 points, no per-point alpha",
    reasoning="New mean experts and old pure NN have residual cosine0.7793 and improve5/9 validation cases; compare limited global blending only",
    no_new_training=True,test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,
    inference_source_sha256=sha(root/"pure_neural_blend_v3.py"),point_source_sha256=sha(root/"point_mean_lowcap_nn.py"))
(out/"candidate_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
with np.load(point,allow_pickle=False) as z:base={k:z[k] for k in z.files}
with np.load(root/"point_prepared"/"mean_data.npz",allow_pickle=False) as z:c=z["conditions"][42:];ids=z["case_ids"][42:]
with np.load(root.parents[1]/"new60_opt_v2"/"neural_route"/"rotor_spatial_magnitude"/"prepared"/"validation_truth.npz",allow_pickle=False) as z:y=z["truth"]
rows=[]
for config in protocol["candidates"]:
    m=dict(base,point_module_sha256=np.asarray(protocol["point_source_sha256"]),blend_route=np.asarray(config["route"]),blend_alpha=np.asarray(config["alpha"]))
    p=predict(m,c);metrics=exact_metrics(p,y,ids);rows.append(dict(configuration=config,validation=metrics))
chosen=min(rows,key=lambda r:r["validation"]["rmse_T"])
m=dict(base,point_module_sha256=np.asarray(protocol["point_source_sha256"]),blend_route=np.asarray(chosen["configuration"]["route"]),blend_alpha=np.asarray(chosen["configuration"]["alpha"]))
meta=dict(version=protocol["version"],point_model_sha256=sha(point),candidate_protocol_sha256=sha(out/"candidate_protocol.json"),inference_source_sha256=protocol["inference_source_sha256"],
    point_source_sha256=protocol["point_source_sha256"],configuration=chosen["configuration"],selection="full validation raw field RMSE, never reused test",test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
m["blend_metadata_json"]=np.asarray(json.dumps(meta));np.savez_compressed(out/"selected_model.npz",**m);(out/"selected_model.npz.sha256").write_text(sha(out/"selected_model.npz")+"\n")
p=predict(out/"selected_model.npz",c);np.savez_compressed(out/"validation_predictions.npz",raw_prediction=p,prediction=p,case_ids=ids,quantity=np.asarray("|B|"),metadata_json=np.asarray(json.dumps(meta)))
result=dict(selected=chosen,rows=rows,selected_model_sha256=sha(out/"selected_model.npz"),validation_predictions_sha256=sha(out/"validation_predictions.npz"),
    source_sha256=protocol["inference_source_sha256"],point_source_sha256=protocol["point_source_sha256"],test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,additional_NN_training_count=0)
(out/"selection_validation_only.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");(out/"selection_validation_only.sha256").write_text(sha(out/"selection_validation_only.json")+"\n")
print(json.dumps(dict(selected=chosen["configuration"],validation=chosen["validation"]["relative_L2_pct"],rmse=chosen["validation"]["rmse_T"],max_abs=chosen["validation"]["max_abs_T"],all=[dict(**r["configuration"],val=r["validation"]["relative_L2_pct"]) for r in rows])),flush=True)
