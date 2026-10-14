"""Freeze kernel route results using saved validation records only."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from prepare_cycles import digest
from search_cycles import full_metrics
import kernel_model
import kernel_magnitude

HERE=Path(__file__).resolve().parent
DIRECTORIES=("rotor_cycle_coarse","rotor_cycle_refined","rotor_vector_blocks6", "rotor_magnitude_cycle_coarse",
             "stator_cycle_coarse","stator_magnitude_cycle_coarse","airgap_cycle_coarse")


def main():
    path=HERE/"route_choices.json"
    if path.exists():raise FileExistsError(path)
    rows=[]
    for name in DIRECTORIES:
        directory=HERE/name;selection=json.loads((directory/"selection.json").read_text(encoding="utf-8"))
        model=directory/"model.npz"
        with np.load(model,allow_pickle=False) as z:meta=json.loads(str(z["metadata_json"]))
        metrics=selection["validation"] if "validation" in selection else selection["selected"]["validation"]
        rows.append(dict(name=name,region=meta["region"],quantity=meta["quantity"],model_relative_path=model.relative_to(HERE).as_posix(),
                         model_sha256=digest(model),predict_module="kernel_model.py",predict_module_sha256=digest(HERE/"kernel_model.py"),
                         prediction_relative_path=(directory/"validation_predictions.npz").relative_to(HERE).as_posix(),validation=metrics,
                         model_format=meta["format"],source_quantity=meta["quantity"],output_transform="identity"))
        if meta["quantity"]!="Br,Bt":continue
        derived=HERE/"derived_magnitude"/name;derived.mkdir(parents=True,exist_ok=False)
        with np.load(directory/"validation_predictions.npz",allow_pickle=False) as z:
            a={k:z[k] for k in z.files}
        pred=np.hypot(a["prediction"][...,0::2],a["prediction"][...,1::2])
        truth=np.hypot(a["truth"][...,0::2],a["truth"][...,1::2])
        metrics=full_metrics(pred.reshape(9,-1),truth.reshape(9,-1),a["case_ids"])
        np.savez_compressed(derived/"validation_predictions.npz",prediction=pred,truth=truth,case_ids=a["case_ids"],
                            point_id=a["point_id"],xy=a["xy"],theta=a["theta"],quantity="|B|")
        rows.append(dict(name=name+"_derived_magnitude",region=meta["region"],quantity="|B|",source_quantity="Br,Bt",
            model_relative_path=model.relative_to(HERE).as_posix(),model_sha256=digest(model),
            predict_module="kernel_magnitude.py",predict_module_sha256=digest(HERE/"kernel_magnitude.py"),
            vector_predict_module_sha256=digest(HERE/"kernel_model.py"),output_transform="hypot(Br,Bt)",
            prediction_relative_path=(derived/"validation_predictions.npz").relative_to(HERE).as_posix(),validation=metrics,
            model_format=meta["format"],no_extra_training=True))
    selected=[]
    for region,quantity in (("rotor","Br,Bt"),("rotor","|B|"),("stator","Br,Bt"),("stator","|B|"),("airgap","Br")):
        group=[row for row in rows if row["region"]==region and row["quantity"]==quantity]
        smallest=min(row["validation"]["rmse_T"] for row in group)
        eligible=[row for row in group if row["validation"]["rmse_T"]<=1.002*smallest]
        best=min(eligible,key=lambda row:(row["validation"]["worst_case_rmse_T"],row["validation"]["max_abs_T"],row["validation"]["rmse_T"]))
        selected.append(best)
    # Numerical replay of all route choices uses ONLY their saved validation
    # conditions and saved validation outputs, not test arrays or test reports.
    checks=[]
    cache_root=HERE.parents[1]/"new60_20261014/cache_v1"
    split=json.loads((cache_root/"split.json").read_text(encoding="utf-8-sig"))
    with threadpool_limits(limits=2):
        for row in selected:
            with np.load(cache_root/(row["region"]+".npz"),allow_pickle=False) as z:
                conditions=z["conditions"][split["validation"]]
            module=kernel_magnitude if row["output_transform"]!="identity" else kernel_model
            actual=module.predict(HERE/row["model_relative_path"],conditions)
            with np.load(HERE/row["prediction_relative_path"],allow_pickle=False) as z:expected=z["prediction"]
            error=float(np.max(np.abs(actual-expected)))
            if not np.allclose(actual,expected,rtol=1e-9,atol=1e-9):raise AssertionError("Frozen prediction replay mismatch")
            checks.append(dict(name=row["name"],maximum_absolute_replay_difference_T=error,passed=True))
    result=dict(schema_version=1,route="condition_kernel",frozen_utc=datetime.now(timezone.utc).isoformat(),rows=rows,selected=selected,
        validation_only=True,test_labels_loaded=False,test_metrics_read=False,all60_fitting_labels_used=False,
        original_split_sha256=digest(cache_root/"split.json"),train_original_case_ids=split["original_ids"]["train"],
        validation_original_case_ids=split["original_ids"]["validation"],excluded_test_original_case_ids=split["original_ids"]["test"],
        predict_module_sha256=digest(HERE/"kernel_model.py"),magnitude_wrapper_sha256=digest(HERE/"kernel_magnitude.py"),
        selection_rule="Minimum validation RMSE; within 0.2% prefer worst-case RMSE then max-absolute error",
        validation_replay_checks=checks,
        limitation="Large search repeatedly reuses nine validation cases. Old V1 test results were already seen by the overall project; final holdout is reused, not fresh blind. No test or all60 fitting data used in this route.",
        goal_rotor_below_one_percent_achieved=False,
        negatives=["Whole-cycle rotor vector floor1.317%; direct magnitude floor1.075%; block vector floor1.206%.",
                   "Refined common-kernel sweep did not improve the coarse rotor model.",
                   "Independent time/component blocks only slightly reduced vector validation error; no <1% claim."])
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (HERE/"route_choices.sha256").write_text(digest(path)+"\n",encoding="ascii")
    lines=["# V2条件核路线冻结结果","","只使用原42组训练和9组验证。所有结果是验证开发指标，不是最终测试误差或全60训练拟合误差。","",
           "| 区域 | 输出量 | 最佳路线 | RMSE/T | 相对L2/% | 最差工况RMSE/T | 最大误差/T |","|---|---|---|---|---|---|---|"]
    for row in selected:
        v=row["validation"];lines.append(f"| {row['region']} | {row['quantity']} | {row['name']} | {v['rmse_T']:.9g} | {v['relative_L2_pct']:.7g} | {v['worst_case_rmse_T']:.8g} | {v['max_abs_T']:.8g} |")
    lines += ["","|B|均为二维hypot(Br,Bt)。derived_magnitude是已训练Br/Bt模型的确定性输出变换，与直接|B|训练对照同量评价；气隙Br不能这样转换。","",
              "转子低于1%的目标未达成。共同整段权重、低秩投影和分块表示的下限及失败改进均保留，不继续无限网格。大规模扫参与分块分别选型对9组验证集使用很重，因此不能把这些验证改进当作未经偏倚的泛化证明。","",
              "全部候选参数、排名CSV、可重构验证系数、源码快照、POD来源与模型哈希均保留。模型统一NumPy入口见README；最终跨路线选择与重复留出测试由主控冻结后统一执行。",""]
    (HERE/"route_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(selected=[{k:row[k] for k in ("name","region","quantity","model_relative_path","model_sha256","predict_module")}|
                                  {"relative_L2_pct":row["validation"]["relative_L2_pct"],"rmse_T":row["validation"]["rmse_T"]} for row in selected],
                          route_choices_sha256=digest(path),replay=checks),ensure_ascii=False,indent=2))

if __name__=="__main__":
    with threadpool_limits(limits=2):main()
