"""Count completed recorded V2 experiments; no training or candidate selection."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    output = ROOT / "qa/optimization_inventory.json"
    if output.exists():
        raise FileExistsError(output)
    checks = []
    def check(name, yes, details=None):
        checks.append(dict(name=name, passed=bool(yes), details=details))
    nn = []
    for p in sorted((ROOT / "neural_route").glob("*/runs/*/result.json")):
        row = read(p); model = p.with_name("model.npz"); hist = p.with_name("history.csv")
        check("completed NN result/model/history present", model.exists() and hist.exists(), str(p.relative_to(ROOT)))
        check("NN saved model SHA", sha(model) == row["model_sha256"], str(model.relative_to(ROOT)))
        check("NN selection excluded test labels", row.get("test_labels_loaded") is False and row.get("test_metrics_computed") is False, str(p.relative_to(ROOT)))
        with hist.open(newline="", encoding="utf-8-sig") as f:
            history = list(csv.DictReader(f))
        nn.append(dict(family=p.parents[2].name, run=p.parent.name, result_relative_path=str(p.relative_to(ROOT)),
            result_sha256=sha(p), model_sha256=sha(model), history_sha256=sha(hist), checkpoint_rows=len(history),
            seed=row.get("configuration", {}).get("seed"), parameter_count=row.get("parameter_count"),
            recorded_best_step=row.get("best_step"), recorded_adam_final_step=row.get("adam_final_step"),
            recorded_lbfgs_closures=row.get("lbfgs_closures"), elapsed_seconds=row.get("elapsed_seconds")))
    frame = []
    for p in sorted((ROOT / "neural_kernel_route").glob("*/summary.json")):
        s = read(p)
        check("neural frame selection excludes test", s["test_labels_accessed"] is False)
        for row in s["rows"]:
            model = p.parent / row["model"]; history = model.with_name("history.json")
            hist = read(history)
            check("frame neural model and history SHA", sha(model) == row["model_sha256"] and len(hist) > 0, row["name"])
            frame.append(dict(run=str(model.parent.relative_to(ROOT)), model_sha256=sha(model), history_sha256=sha(history),
                              checkpoint_rows=len(hist), last_logged_step=hist[-1]["step"], best_step=row["best_step"], seed=row["spec"]["seed"]))
    fit = []
    for p in sorted((ROOT / "neural_route/fitting_full60").glob("*/result.json")):
        s = read(p); model = p.with_name("model.npz")
        check("full60 NN model SHA", sha(model) == s["model_sha256"], p.parent.name)
        check("full60 fitting does not claim holdout metrics", s["heldout_metrics_computed"] is False and len(s["training_case_ids"]) == 60)
        fit.append(dict(region=p.parent.name, result_relative_path=str(p.relative_to(ROOT)), result_sha256=sha(p),
                        model_sha256=sha(model), history_sha256=sha(p.with_name("history.csv")), parameter_count=s["parameter_count"],
                        best_step=s["best_step"], seed=s["seed"], elapsed_seconds=s["elapsed_seconds"]))
    kernels = []
    for ranking in sorted((ROOT / "kernel_route").rglob("ranking.csv")):
        selection = ranking.with_name("selection.json"); s = read(selection)
        unique, ranks = set(), set(); count = 0; finite = True
        with ranking.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                count += 1; unique.add(int(row["candidate_index"])); ranks.add(int(row["rank"]))
                finite = finite and all(math.isfinite(float(row[k])) for k in ("rmse_T", "relative_L2_pct", "worst_case_rmse_T"))
        check("kernel completed ranking matches summary", count == s["ranked_candidates"] and len(unique)+len(s["failures"]) == s["candidates"] and finite,
              dict(directory=str(ranking.parent.relative_to(ROOT)), csv_rows=count, unique_solved_configs=len(unique)))
        check("kernel final saved model SHA", sha(ranking.with_name("model.npz")) == s["model_sha256"])
        kernels.append(dict(directory=str(ranking.parent.relative_to(ROOT)), csv_sha256=sha(ranking), selection_sha256=sha(selection),
                            attempted_configurations=s["candidates"], completed_unique_configurations=len(unique), ranked_configuration_rank_rows=count,
                            ranks=sorted(ranks), failure_count=len(s["failures"]), fully_decoded_shortlist_count=len(s["fully_decoded_shortlist"]),
                            model_sha256=s["model_sha256"], optimization_epochs=0))
    adaptive = []
    for name in ("adaptive_pilot", "adaptive_pilot2", "adaptive_interior"):
        directory = ROOT / name / "rotor"; summary = directory / "summary.json"
        if not summary.exists():
            adaptive.append(dict(directory=str(directory.relative_to(ROOT)), completed_validation_variants=0,
                                 status="protocol only; no completed summary/model counted", protocol_sha256=sha(directory/"protocol.json")))
            continue
        s = read(summary); ranking = directory / "validation_ranking.json"; rows = read(ranking)
        check("adaptive complete ranking coverage", len(rows) == len(s["rows"]) and {r["name"] for r in rows} == {r["name"] for r in s["rows"]})
        model = directory / "model.npz"
        check("adaptive saved model sidecar", sha(model) == model.with_suffix(".npz.sha256").read_text().strip())
        adaptive.append(dict(directory=str(directory.relative_to(ROOT)), completed_validation_variants=len(rows), status="completed",
                             summary_sha256=sha(summary), ranking_sha256=sha(ranking), model_sha256=sha(model),
                             variant_names=[r["name"] for r in rows], optimization_epochs=0,
                             note="Each row is a completed grouped analytic predictor, not an NN training run. Internal per-mode kernel trials are not added to this variant count."))
    power_dir = ROOT / "power_route/rotor"; ranking = read(power_dir/"ranking.json"); summary = read(power_dir/"summary.json")
    unique = {json.dumps({"power":r["power"], "spec":r["spec"]},sort_keys=True) for r in ranking}
    check("power completed configurations unique and finite", len(unique) == len(ranking) and all(math.isfinite(r["validation"]["rmse_T"]) for r in ranking))
    check("power final model SHA", sha(power_dir/"model.npz") == summary["model_sha256"])
    power = dict(completed_validation_configurations=len(ranking), by_power=dict(Counter(str(r["power"]) for r in ranking)),
                 ranking_sha256=sha(power_dir/"ranking.json"), summary_sha256=sha(power_dir/"summary.json"), model_sha256=summary["model_sha256"], optimization_epochs=0)
    totals = dict(prediction_MLP_training_runs=len(nn), prediction_neural_frame_kernel_training_runs=len(frame),
                  prediction_neural_training_runs=len(nn)+len(frame), full60_fitting_NN_training_runs=len(fit),
                  all_completed_neural_training_runs=len(nn)+len(frame)+len(fit),
                  kernel_search_directories=len(kernels), kernel_completed_unique_configuration_solves=sum(k["completed_unique_configurations"] for k in kernels),
                  kernel_configuration_rank_validation_rows=sum(k["ranked_configuration_rank_rows"] for k in kernels),
                  kernel_failed_configurations=sum(k["failure_count"] for k in kernels),
                  adaptive_completed_validation_variants=sum(a["completed_validation_variants"] for a in adaptive),
                  power_completed_validation_configurations=len(ranking))
    result = dict(all_pass=all(c["passed"] for c in checks), counts=totals, neural_families=dict(Counter(r["family"] for r in nn)),
                  neural_prediction_runs=nn, neural_frame_kernel_runs=frame, full60_fitting_runs=fit,
                  kernel_searches=kernels, adaptive_searches=adaptive, power_search=power, checks=checks,
                  checker_sha256=sha(Path(__file__)), prediction_freeze_sha256=sha(ROOT/"prediction_choices_frozen.json"),
                  combined_freeze_sha256=sha(ROOT/"both_versions_choices_frozen.json"),
                  counting_rules=["Count existing completed result/history/model records, never infer success from a protocol's requested budget.",
                    "Seed replicas count as separate NN optimization runs. Final selected_model copies do not count again.",
                    "Kernel configuration solves and multiple decoder ranks are different counts. Neither is an NN epoch/run.",
                    "Only completed adaptive summary rows count; adaptive_pilot with protocol only contributes zero.",
                    "V1, synthetic tests, fixed diagnostic oracle/LOOCV calculations and deterministic vector-to-magnitude wrappers are excluded.",
                    "This inventory checks artifacts and recorded counts, not another replay of every historical candidate."])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# V2 实际完成的优化实验清单", "", "状态："+("PASS" if result["all_pass"] else "FAIL"), "",
             f"预测分支实际完成 {len(nn)} 次 MLP 训练和 {len(frame)} 次神经坐标核训练，共 {len(nn)+len(frame)} 次；另有独立全 60 工况拟合 NN 训练 {len(fit)} 次，总计 {len(nn)+len(frame)+len(fit)} 次。", "",
             "|MLP 系列|完成训练次数|", "|---|---:|"]
    lines += [f"|{k}|{v}|" for k,v in result["neural_families"].items()]
    lines += ["", f"解析 kernel 搜索共 {len(kernels)} 个目录，完成配置求解 {totals['kernel_completed_unique_configuration_solves']:,} 个，配置×解码秩验证排名行 {totals['kernel_configuration_rank_validation_rows']:,} 行；失败配置 {totals['kernel_failed_configurations']} 个。不同秩共用同一次条件映射求解，不重复计为模型训练。", "",
              f"power 变换路线有 {len(ranking):,} 个完成的验证配置；adaptive_pilot2 有 12 个、adaptive_interior 有 20 个完成的分组预测器，合计 {totals['adaptive_completed_validation_variants']} 个。adaptive_pilot 仅留下 protocol，没有完成模型/summary，计 0。", "",
              "上述解析候选均为 0 个梯度训练 epoch；不能与神经网络训练次数相加后声称训练了同等数量的神经网络。V1、合成 QA、诊断性 oracle/LOOCV、selected_model 副本与确定性取范数 wrapper 均不重复计数。", "",
              f"共 {len(checks)} 项文件与数量检查；详细 CSV/JSON/模型/日志 SHA 见同名 JSON。此清单不再次训练、运行候选或调整选择。"]
    output.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(all_pass=result["all_pass"], checks=len(checks), **totals)))
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
