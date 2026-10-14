"""Explicit budget amendment; does not rewrite any completed spatial experiment."""
from pathlib import Path
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
from spatial_magnitude_neural import train,load_model
from magnitude_cycle_neural import read_json,write_new,digest,save_model


def run(root):
    root=Path(root)
    configs=[dict(name=f"{label}_budget_seed42",rank=384,features=feature,hidden=[32,32],seed=42,
                  regularization=1e-7,adam_steps=1500,lbfgs_blocks=30)
             for label,feature in (("sat300_32","sat300"),("native32","native"))]
    write_new(root/"candidate_protocol_v2.json",dict(version="explicit-spatial-budget-amendment-v2",candidates=configs,
        reason="Parent requested finite extension after fully trained cart32 and sat150_32 both failed to improve magnitude whole-cycle validation; avoid low-yield repeated validation tuning",
        completed_original_runs=["cart32_seed42","sat150_32_seed42"],
        abandoned_original_run="sat300_32_seed42 interrupted after its1000-step diagnostic; no model or final score available",
        cancelled_original_runs=["native32_seed42","sat150_64_seed42","sat150_r512_seed42","all confirmation seeds"],
        source_sha256=digest(__file__),training_source_sha256=digest(Path(__file__).with_name("spatial_magnitude_neural.py")),
        selection="minimum raw validation field RMSE among completed spatial runs, fixed seed42; no test labels",
        modification="Original protocol and completed results remain unchanged; new budget suffix distinguishes amended runs"))
    incomplete=root/"runs"/"sat300_32_seed42"
    if incomplete.exists() and not (incomplete/"result.json").exists():
        write_new(incomplete/"aborted.json",dict(reason="Budget amendment; interrupted at train step>=1000 before final checkpoint export",usable_model=False,final_metrics_available=False))
    with threadpool_limits(limits=2):
        rows=[read_json(root/"runs"/name/"result.json") for name in ("cart32_seed42","sat150_32_seed42")]
        for cfg in configs:rows.append(train(root/"prepared",root/"runs"/cfg["name"],cfg))
        best=min(rows,key=lambda r:r["validation"]["rmse_T"]);name=best["name"]
        model=load_model(root/"runs"/name/"model.npz");model.pop("representation_relative_path",None);save_model(root/"selected_model.npz",model)
        frozen=dict(version="spatial-prediction-amended-selection-v2",quantity="|B|",selected=name,rows=rows,
            protocol_original_sha256=digest(root/"candidate_protocol.json"),protocol_amendment_sha256=digest(root/"candidate_protocol_v2.json"),
            source_sha256=digest(__file__),training_source_sha256=digest(Path(__file__).with_name("spatial_magnitude_neural.py")),
            selected_model_sha256=digest(root/"selected_model.npz"),test_labels_loaded=False,test_metrics_computed=False)
        write_new(root/"selection_frozen.json",frozen);(root/"selection_frozen.sha256").write_text(digest(root/"selection_frozen.json")+"\n")
        print(json.dumps(dict(selected=name,validation=best["validation"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();run(a.output)
