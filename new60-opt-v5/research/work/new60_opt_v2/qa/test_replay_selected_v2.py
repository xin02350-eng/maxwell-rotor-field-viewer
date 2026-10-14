"""Synthetic checks; this file never opens the real Maxwell dataset."""
from pathlib import Path
import copy
import json
import tempfile
import numpy as np
import replay_selected_v2 as qa


def write_json(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def main():
    checks = []
    def check(name, passed):
        checks.append({"name": name, "pass": bool(passed)})
        assert passed, name
    with tempfile.TemporaryDirectory(prefix="v2_replay_synthetic_") as temp:
        root = Path(temp)
        p = root / "rows.npz"
        a = np.arange(60*2*3, dtype=np.float32).reshape(60, 2, 3)
        a[59] = np.nan
        np.savez_compressed(p, field=a)
        check("selected rows preserve requested order", np.array_equal(qa.read_rows(p, "field", [5, 0]), a[[5, 0]]))
        masked = qa.read_rows(p, "field", [0, 5], return_masked=True)
        check("excluded NaN row never materialized; all excluded outputs NaN", np.isnan(masked[[i for i in range(60) if i not in [0, 5]]]).all())
        try:
            qa.read_rows(p, "field", [59])
        except ValueError:
            check("nonfinite authorized label rejects", True)
        else:
            check("nonfinite authorized label rejects", False)
        zero = qa.aggregate_metrics(np.ones((2, 3)), np.zeros((2, 3)))
        check("zero truth relative L2 is null, absolute RMSE retained", zero["relative_L2_pct"] is None and zero["rmse_T"] == 1 and zero["near_zero_truth_rms"])
        # Direction reversal is a 200% vector error but zero magnitude error.
        true = np.asarray([[[3., 4.], [3., 4.]]]); pred = -true
        m = qa.metric_report(pred, true, np.array([7]), np.array([11]), "vector")
        check("vector and derived magnitude metrics separated", m["global_metrics"]["relative_L2_pct"] == 200 and m["derived_planar_magnitude_error_separate_scope"]["rmse_T"] == 0)
        check("frame and point aggregation retain identity", m["per_case_frame"]["case_ids"] == [7] and m["per_point"]["point_ids"] == [11])
        choice_path = root / "choices.json"
        write_json(choice_path, {"frozen": True, "previous_test_already_seen": True})
        try:
            qa.label_policy(qa.read_json(choice_path), "prediction", True, choice_path)
        except ValueError:
            check("test rejected without freeze SHA", True)
        else:
            check("test rejected without freeze SHA", False)
        choice_path.with_suffix(".json.sha256").write_text(qa.sha256(choice_path), encoding="ascii")
        check("test gate recognizes valid frozen choice", "reused_test" in qa.label_policy(qa.read_json(choice_path), "prediction", True, choice_path))
        check("default prediction still excludes test despite freeze", qa.label_policy(qa.read_json(choice_path), "prediction", False, choice_path) == ["train", "validation"])
        check("fitting only partition is training_fit", qa.label_policy({"full60_fitting_authorized": True}, "fitting_full60", False, choice_path) == ["training_fit"])
        try:
            qa.label_policy({}, "fitting_full60", False, choice_path)
        except ValueError:
            check("fitting needs explicit authorization record", True)
        else:
            check("fitting needs explicit authorization record", False)

        # End-to-end miniature source/cache/model uses 60 synthetic cases, not real data.
        study = root / "study"; (study / "source/combined").mkdir(parents=True); (study / "cache_v1").mkdir()
        split = {"case_ids": list(range(60)), "train": list(range(42)), "validation": list(range(42, 51)), "test": list(range(51, 60))}
        split["original_ids"] = {k: split[k] for k in ("train", "validation", "test")}
        write_json(study / "cache_v1/split.json", split)
        c = np.column_stack((np.arange(60), np.arange(60)*2)).astype(float)
        time = np.arange(3, dtype=float)
        b = np.arange(6, dtype=float).reshape(3, 2)/8
        field = (c[:, 0, None, None]/16 + b[None]).astype(np.float32)
        source_data = dict(case_id=np.arange(60), I_A=c[:, 0], Thet_deg=c[:, 1], theta_m_absolute_deg=time, solver_index=time+1, time_s=time/100)
        for region in ("stator", "rotor", "airgap"):
            source_data.update({region+"_Br": field, region+"_point_id": np.arange(2), region+"_x_ref_mm": np.arange(2), region+"_y_ref_mm": np.ones(2)})
            if region != "airgap":
                source_data[region+"_Bt"] = field/2
        source = study / "source/combined/flux60_s60_merged.npz"
        np.savez_compressed(source, **source_data)
        cache = study / "cache_v1/airgap.npz"
        np.savez_compressed(cache, field=field, case_ids=np.arange(60), conditions=c,
            point_id=np.arange(2), xy=np.column_stack((np.arange(2), np.ones(2))), theta=time,
            pod_train_case_indices=split["train"], mean=field[:42].reshape(-1, 2).mean(axis=0, dtype=np.float64).astype(np.float32))
        module = root / "inference.py"
        module.write_text('''from pathlib import Path
import numpy as np
def predict(model,conditions,frame_index=None):
    if isinstance(model,(str,Path)):
        with np.load(model,allow_pickle=False) as z:model={k:z[k] for k in z.files}
    c=np.asarray(conditions);single=c.ndim==1;c=np.atleast_2d(c)
    out=c[:,0,None,None]/16+model['template'][None]
    if frame_index is not None:out=out[:,frame_index]
    return out[0] if single else out
''', encoding="utf-8")
        metadata = dict(source_sha256=qa.sha256(module), cache_sha256=qa.sha256(cache), split_sha256=qa.sha256(study / "cache_v1/split.json"),
                        train_case_ids=split["train"], validation_case_ids=split["validation"])
        model = root / "model.npz"
        np.savez_compressed(model, template=b, metadata_json=json.dumps(metadata))
        saved = root / "prediction.npz"
        np.savez_compressed(saved, prediction=field[42:51].astype(float), case_ids=np.arange(42, 51))
        choice = dict(id="tiny", region="airgap", quantity="Br", model_relative_path="model.npz", model_sha256=qa.sha256(model),
            inference_module_relative_path="inference.py", inference_module_sha256=qa.sha256(module),
            saved_predictions=[dict(partition="validation", relative_path="prediction.npz", sha256=qa.sha256(saved))])
        choices = dict(source_study_relative_path="study", source_sha256=qa.sha256(source), previous_test_already_seen=True, models=[choice])
        write_json(choice_path, choices)
        report = qa.run(choice_path, root / "audit", root)
        check("end-to-end synthetic original/cache/model replay PASS", report["all_checks_pass"])
        check("prediction QA reads 51 rows and no test labels", report["decoded_original_label_rows"] == list(range(51)) and report["excluded_label_rows"] == list(range(51, 60)) and not report["test_labels_decoded_for_prediction"])
        check("exact synthetic metric replay zero", report["results"][0]["metrics"]["validation"]["global_metrics"]["rmse_T"] == 0)
        try:
            qa.run(choice_path, root / "audit", root)
        except FileExistsError:
            check("existing report never overwritten", True)
        else:
            check("existing report never overwritten", False)
        # Hash verifies current identity, metadata also independently verifies the training-time identity.
        module.write_text(module.read_text() + "\n# post-training mutation\n", encoding="utf-8")
        changed = copy.deepcopy(choices); changed["models"][0]["inference_module_sha256"] = qa.sha256(module)
        write_json(choice_path, changed)
        drift = qa.run(choice_path, root / "drift_audit", root)
        check("source change detected despite updated choices hash", not drift["all_checks_pass"] and any(c["name"].startswith("training-time source SHA") and not c["pass"] for c in drift["checks"]))
        module.write_text(module.read_text().replace("\n# post-training mutation\n", ""), encoding="utf-8")
        wrong = copy.deepcopy(choices); wrong["models"][0]["model_sha256"] = "0"*64
        write_json(choice_path, wrong)
        failure = qa.run(choice_path, root / "hash_failure", root)
        check("wrong model SHA fails before inference", not failure["all_checks_pass"] and "metrics" not in failure["results"][0])
        # Fit is a separate branch using synthetic full60 only.
        metadata["train_case_ids"] = list(range(60)); metadata.pop("validation_case_ids")
        np.savez_compressed(model, template=b, metadata_json=json.dumps(metadata))
        np.savez_compressed(saved, prediction=field.astype(float), case_ids=np.arange(60))
        fitting = copy.deepcopy(choice); fitting.update(branch="fitting_full60", model_sha256=qa.sha256(model))
        fitting["saved_predictions"] = [dict(partition="training_fit", relative_path="prediction.npz", sha256=qa.sha256(saved))]
        write_json(choice_path, dict(choices, full60_fitting_authorized=True, models=[fitting]))
        fitted = qa.run(choice_path, root / "fit_audit", root, branch="fitting_full60")
        check("separate full60 fitting replay PASS", fitted["all_checks_pass"] and fitted["evaluated_partitions"] == ["training_fit"] and len(fitted["decoded_original_label_rows"]) == 60)
        vector = np.stack((field, field/2), axis=-1).reshape(60, 3, 4)
        vcache = study / "cache_v1/rotor.npz"
        np.savez_compressed(vcache, field=vector, case_ids=np.arange(60), conditions=c,
            point_id=np.arange(2), xy=np.column_stack((np.arange(2), np.ones(2))), theta=time,
            pod_train_case_indices=split["train"], mean=vector[:42].reshape(-1, 4).mean(axis=0, dtype=np.float64).astype(np.float32))
        base = root / "vector_inference.py"
        base.write_text(module.read_text().replace("c[:,0,None,None]/16", "c[:,0,None,None]*np.array([1/16,1/32,1/16,1/32])"), encoding="utf-8")
        wrapper = root / "wrapper.py"
        wrapper.write_text('''import numpy as np
from vector_inference import predict as vector_predict
def predict(model,conditions,frame_index=None):
    v=vector_predict(model,conditions,frame_index)
    return np.hypot(v[...,0::2],v[...,1::2])
''', encoding="utf-8")
        vmodel = root / "vector.npz"
        vm = dict(source_sha256=qa.sha256(base), cache_sha256=qa.sha256(vcache), split_sha256=qa.sha256(study / "cache_v1/split.json"),
                  train_case_ids=split["train"], validation_case_ids=split["validation"], quantity="Br,Bt")
        np.savez_compressed(vmodel, template=np.stack((b, b/2), axis=-1).reshape(3, 4), metadata_json=json.dumps(vm))
        magnitude = np.hypot(vector[..., 0::2], vector[..., 1::2]).astype(float)
        msaved = root / "magnitude.npz"
        # Generate saved array by the float64 inference convention, not float32 hypot.
        exact_magnitude = np.hypot(vector.astype(float)[..., 0::2], vector.astype(float)[..., 1::2])
        np.savez_compressed(msaved, prediction=exact_magnitude[42:51], case_ids=np.arange(42, 51))
        mc = dict(choice, id="derived", region="rotor", quantity="planar_magnitude", source_quantity="Br,Bt", output_transform="hypot(Br,Bt)",
            model_relative_path="vector.npz", model_sha256=qa.sha256(vmodel), inference_module_relative_path="wrapper.py", inference_module_sha256=qa.sha256(wrapper),
            model_source_module_relative_path="vector_inference.py", model_source_module_sha256=qa.sha256(base),
            saved_predictions=[dict(partition="validation", relative_path="magnitude.npz", sha256=qa.sha256(msaved))])
        write_json(choice_path, dict(choices, models=[mc]))
        derived = qa.run(choice_path, root / "derived_audit", root)
        check("derived wrapper uses signed model provenance and scalar metrics", derived["all_checks_pass"])
        check("derived wrapper independently equals base hypot", any(x["name"] == "derived magnitude wrapper equals independent hypot" and x["pass"] for x in derived["checks"]))
        metric_file = root / "old_metric.json"
        write_json(metric_file, {"validation": {"rmse_T": 123.}})
        mc["reported_metrics"] = [dict(partition="validation", relative_path="old_metric.json", sha256=qa.sha256(metric_file), pointer="/validation")]
        write_json(choice_path, dict(choices, models=[mc]))
        wrongmetric = qa.run(choice_path, root / "wrong_metric", root)
        check("inconsistent reported metric is retained as FAIL", not wrongmetric["all_checks_pass"] and any(x["name"] == "reported metric replay: validation/rmse_T" and not x["pass"] for x in wrongmetric["checks"]))
    result = dict(all_pass=all(x["pass"] for x in checks), checks=checks,
                  real_Maxwell_labels_opened=False, checker_sha256=qa.sha256(Path(qa.__file__)), test_source_sha256=qa.sha256(Path(__file__)))
    out = Path(__file__).with_name("replay_selected_v2_synthetic_check.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(all_pass=result["all_pass"], check_count=len(checks), real_Maxwell_labels_opened=False)))


if __name__ == "__main__":
    main()
