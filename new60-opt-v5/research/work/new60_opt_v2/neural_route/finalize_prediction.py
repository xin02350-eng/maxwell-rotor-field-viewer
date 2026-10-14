"""Freeze this route before any all60 fitting labels are read."""
from pathlib import Path
import datetime,json
from magnitude_cycle_neural import read_json,write_new,digest


def run(root):
    root=Path(root);candidates=[];all_rows=[]
    families=(("rotor","cycle_neural.py","Br,Bt"),("rotor_magnitude","magnitude_cycle_neural.py","|B|"),
              ("rotor_magnitude_blocks","magnitude_blocks.py","|B|"),("rotor_spatial_magnitude","spatial_magnitude_neural.py","|B|"))
    for folder,module,quantity in families:
        f=read_json(root/folder/"selection_frozen.json")
        if f.get("test_labels_loaded") is not False:raise ValueError("Prediction family has unsafe label provenance")
        row=read_json(root/folder/"runs"/f["selected"]/"result.json")
        candidates.append(dict(family=folder,prediction_module=module,quantity=quantity,selected=f["selected"],
            model_relative_path=folder+"/selected_model.npz",model_sha256=digest(root/folder/"selected_model.npz"),
            family_freeze_sha256=digest(root/folder/"selection_frozen.json"),validation_raw=row["validation"],
            validation_nonnegative=row.get("validation_nonnegative"),selected_checkpoint_train=row["train"],
            raw_negative_count=row.get("raw_negative_count"),source_sha256=digest(root/module)))
        for p in sorted((root/folder/"runs").glob("*/result.json")):
            r=read_json(p);all_rows.append(dict(family=folder,name=r["name"],quantity=quantity,validation=r["validation"],
                train=r["train"],configuration=r["configuration"],elapsed_seconds=r["elapsed_seconds"],model_sha256=r["model_sha256"]))
    selected=min((c for c in candidates if c["quantity"]=="|B|"),key=lambda c:c["validation_raw"]["rmse_T"])
    frozen=dict(version="neural-route-prediction-choice-v2",frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selected=selected["selected"],selected_model_relative_path=selected["model_relative_path"],selected_model_sha256=selected["model_sha256"],
        prediction_module=selected["prediction_module"],quantity="|B|",region="rotor",validation_raw=selected["validation_raw"],
        selection_rule="Compare fixed-seed42 deployment representatives from each completed magnitude family using raw validation field RMSE; family internal choices preserved",
        candidates=candidates,completed_run_count=len(all_rows),test_labels_loaded=False,test_metrics_computed=False,
        target_below_one_percent_achieved=selected["validation_raw"]["relative_L2_pct"]<1,
        limitation="Repeated validation exploration can overfit validation; final independent heldout evaluation is reserved for root after global freeze",
        fitting_full60_branch_started=False,source_sha256=digest(__file__))
    write_new(root/"prediction_choice_frozen.json",frozen)
    (root/"prediction_choice_frozen.sha256").write_text(digest(root/"prediction_choice_frozen.json")+"\n",encoding="ascii")
    write_new(root/"prediction_route_results.json",dict(frozen=frozen,runs=all_rows))
    lines=["# V2 独立神经网络预测路线结果", "", "这里只记录 42 个训练工况、9 个验证工况的结果。未数值读取固定 9 个测试工况标签，也未计算测试误差。全部 60 工况拟合将于此选择冻结后进入独立分支，其误差不能与本表混称。", "",
           "| 目标量与方法 | 固定 seed 42 模型 | 验证 RMSE / T | 验证 relative L2 | 所选检查点训练 relative L2 |", "|---|---|---:|---:|---:|"]
    for c in candidates:
        lines.append(f"| {c['quantity']} / {c['family']} | {c['selected']} | {c['validation_raw']['rmse_T']:.8f} | {c['validation_raw']['relative_L2_pct']:.5f}% | {c['selected_checkpoint_train']['relative_L2_pct']:.5f}% |")
    lines += ["",f"本路线共完成 {len(all_rows)} 个真实神经网络实验；磁密大小预测候选选用 `{selected['model_relative_path']}`，纯 NumPy 入口为 `{selected['prediction_module']}::predict`。固定 42/9 分组下仍未实现小于 1% 的验证误差，不能宣称达到该目标。", "",
        "原始输出用于全部检查点和配方比较；磁密大小的可选非负投影另报，不以投影后得分替代原始误差。磁密大小是 sqrt(Br²+Bt²)，其 relative L2 与 Br/Bt 向量 relative L2 分母和量不同，不能直接混用。", "",
        "整周期 42 工况仿射子空间的磁密大小验证投影下限约 1.075%，每帧独立工况子空间约 0.895%；训练专用空间 POD 的 rank384/512 下限约 0.219%/0.171%。这些使用验证真值投影的诊断是表示能力下限，不是可部署网络的预测得分。", "",
        "更长空间 MLP 训练能继续降低训练误差，却使验证误差上升。原完整 cart32、sat150_32 的 3000 Adam + 800 LBFGS 预算结果保留；其余空间探索按显式第二协议缩减为 sat300/native 两个 1500 Adam + 300 LBFGS 配方。中断候选只记录中断状态，没有伪造最终得分。", "",
        "各实验保存训练日志、原始权重、验证预测、超参数、数据/代码/表示 SHA。整周期及分块路线均以最优两个配方的三个种子均值确认配方，部署种子固定为 42；有限空间扩展只有种子 42，不能声称三种子稳定。", "",
        "数值推理只覆盖保存的 60 个转子位置与固定采样点；未证明任意转子角、转速或独立 Maxwell 盲测的泛化。重复使用同一验证集探索本身存在选择偏差，因此最终测试应在全研究冻结后独立进行。"]
    (root/"PREDICTION_ROUTE_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(frozen,ensure_ascii=False,indent=2))


if __name__=="__main__":run(Path(__file__).parent)
