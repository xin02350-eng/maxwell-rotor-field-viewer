"""Check serialized science artifacts after frozen evaluation; no inference/training.

All 27 files: hashes, schema, partition identity, units/coordinates, scalar
global metrics after NPZ roundtrip, and original-source truth spot checks.
This does not select models or feed results into training.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def source_subset(p, key, wanted):
    result = {}
    with zipfile.ZipFile(p) as z, z.open(key+".npy") as f:
        v = np.lib.format.read_magic(f)
        shape, order, dtype = np.lib.format._read_array_header(f, v)
        if order or shape[0] != 60 or len(shape) != 3:
            raise ValueError("Unexpected original field layout")
        size = int(np.prod(shape[1:]))*dtype.itemsize
        for i in range(60):
            if i in wanted:
                result[i] = np.frombuffer(f.read(size), dtype=dtype).reshape(shape[1:]).astype(float)
            else:
                f.seek(size, 1)
    return result


def main():
    output = ROOT / "qa/scientific_npz_spotchecks.json"
    if output.exists():
        raise FileExistsError(output)
    source = ROOT.parent / "new60_20261014/source/combined/flux60_s60_merged.npz"
    choices = ROOT / "both_versions_choices_frozen.json"
    choice_hash = digest(choices)
    if choice_hash != "c26dcdb4213e5afe09d78cf301e08e34234be2c1bf78a7aa32eab5673c4174c0":
        raise ValueError("Unexpected global frozen choices")
    reports = [ROOT / "qa/prediction_replay/replay_report.json", ROOT / "qa/fitting_replay/replay_report.json"]
    split = read(ROOT.parent / "new60_20261014/cache_v1/split.json")
    checks, artifacts = [], []
    def check(name, yes, details=None):
        checks.append(dict(name=name, passed=bool(yes), details=details))
    records = [r for p in reports for r in read(p)["results"]]
    with np.load(source, allow_pickle=False) as raw:
        conditions = np.column_stack((raw["I_A"], raw["Thet_deg"]))
        geometry = {r: dict(xy=np.column_stack((raw[r+"_x_ref_mm"], raw[r+"_y_ref_mm"])), point_id=raw[r+"_point_id"])
                    for r in ("stator", "rotor", "airgap")}
        timing = {"theta": raw["theta_m_absolute_deg"], "time_s": raw["time_s"], "solver_index": raw["solver_index"],
                  "theta_m_relative_deg": raw["theta_m_relative_deg"], "theta_e_deg": raw["theta_e_deg"]}
    groups = dict(train=split["train"], validation=split["validation"], reused_test=split["test"], training_fit=list(range(60)))
    wanted = sorted({v[i] for v in groups.values() for i in (0, len(v)//2, len(v)-1)})
    signed = {}
    for region in ("stator", "rotor", "airgap"):
        br = source_subset(source, region+"_Br", wanted)
        if region == "airgap":
            signed[region] = br
        else:
            bt = source_subset(source, region+"_Bt", wanted)
            signed[region] = {i: np.stack((br[i], bt[i]), axis=-1).reshape(60, -1) for i in wanted}
    total_bytes = 0
    with threadpool_limits(limits=2):
      for record in records:
        region, quantity = record["region"], record["quantity"]
        quantity = "vector" if quantity == "Br,Bt" else quantity
        for partition, artifact in record["scientific_artifacts"].items():
            p = Path(artifact["path"])
            # Export paths are recorded absolutely; require the expected fixed local tree.
            expected_path = ROOT / "evaluation" / record["id"] / (partition+"_predictions.npz")
            check("artifact location", p.resolve() == expected_path.resolve(), record["id"]+"/"+partition)
            check("saved artifact SHA", digest(p) == artifact["sha256"], p.name)
            total_bytes += p.stat().st_size
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata_json"]))
                ids = z["case_ids"]; true = z["truth"]; pred = z["prediction"]
                check("partition identity", ids.tolist() == groups[partition], record["id"]+"/"+partition)
                check("freeze and model identities", meta["choices_sha256"] == choice_hash and meta["model_sha256"] == record["model_sha256"])
                check("float64 physical arrays", true.dtype == np.float64 and pred.dtype == np.float64 and true.shape == pred.shape == tuple(artifact["shape"]))
                check("finite physical arrays", np.isfinite(true).all() and np.isfinite(pred).all())
                check("physical quantity and units", str(z["quantity"]) == quantity and meta["quantity"] == quantity and meta["units"] == "T" and meta["xy_units"] == "mm")
                check("native condition identity", np.array_equal(z["conditions"], conditions[ids]))
                check("reference spatial point order", all(np.array_equal(z[k], v) for k, v in geometry[region].items()))
                check("saved time and angle identity", all(np.array_equal(z[k], v) for k, v in timing.items()))
                check("fit versus reused-test meaning", (record["branch"] == "fitting_full60") == (partition == "training_fit")
                      and meta["test_is_fresh_blind"] is False)
                # Separate explicit scalar reductions, independent of replay aggregation helpers.
                error = pred-true
                sumsquared = float(np.square(error).sum(dtype=np.float64)); norm2 = float(np.square(true).sum(dtype=np.float64))
                observed = dict(rmse_T=float(np.sqrt(sumsquared/error.size)), mae_T=float(np.abs(error).sum(dtype=np.float64)/error.size),
                                max_abs_T=float(np.abs(error).max()), relative_L2_pct=float(100*np.sqrt(sumsquared/norm2)))
                expected = record["metrics"][partition]["global_metrics"]
                check("post-serialization global metrics", all(np.isclose(v, expected[k], atol=1e-12, rtol=1e-12) for k,v in observed.items()), observed)
                positions = [0, len(ids)//2, len(ids)-1]
                frames = np.asarray([0, 17, 59]); points = np.asarray([0, len(z["point_id"])//2, len(z["point_id"])-1])
                count = 0; max_delta = 0.
                for pos in positions:
                    original = signed[region][int(ids[pos])]
                    if quantity == "planar_magnitude":
                        original = np.hypot(original[..., 0::2], original[..., 1::2])
                        columns = points
                    elif quantity == "vector":
                        columns = np.ravel(np.column_stack((2*points, 2*points+1)))
                    else:
                        columns = points
                    expected_values = original[np.ix_(frames, columns)]
                    written_values = true[pos][np.ix_(frames, columns)]
                    max_delta = max(max_delta, float(np.max(np.abs(expected_values-written_values))))
                    count += expected_values.size
                check("source truth spot checks", max_delta < 2e-14, dict(sample_values=count, max_abs_delta_T=max_delta,
                      case_ids=ids[positions].tolist(), frame_indices=frames.tolist(), point_indices=points.tolist()))
                if partition == "training_fit" and quantity == "planar_magnitude":
                    check("deployed fitting magnitude is nonnegative", np.min(pred) >= 0)
                artifacts.append(dict(id=record["id"], partition=partition, sha256=artifact["sha256"], shape=list(true.shape),
                                      quantity=quantity, independent_metrics=observed, truth_sample_max_abs_delta_T=max_delta))
    check("all eleven frozen models represented", len(records) == 11)
    check("all twenty-seven partition files represented", len(artifacts) == 27)
    result = dict(all_pass=all(c["passed"] for c in checks), checks=checks, artifact_count=len(artifacts), model_count=len(records),
                  scientific_npz_total_bytes=total_bytes, artifacts=artifacts, created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=digest(source), choices_sha256=choice_hash, checker_sha256=digest(Path(__file__)),
                  source_case_rows_numerically_decoded=wanted, labels_authorized_by="Root explicit frozen reused-test and independent full60 fitting evaluation authorization",
                  inference_or_training_performed=False, no_selection_changes=True,
                  limitation="Truth spot checks inspect selected case/frame/point entries; all exported prediction/truth arrays are read for independent global metric and finite-value checks. This is not a second complete model inference.")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 保存后的科学数据核验", "", "状态："+("PASS" if result["all_pass"] else "FAIL"), "",
             f"11 个冻结模型、27 个分区 NPZ，{len(checks)} 项检查；文件合计 {total_bytes:,} 字节。", "",
             "逐文件重新读取 float64 truth/prediction，独立计算 RMSE、MAE、maxAbs、全场 relative L2，与第一次回放的 JSON 全部比较。核验文件 SHA、工况和原空间点顺序、原生电流条件、时间/角度、单位、分支身份。", "",
             "原始 signed Br/Bt 或 Br 的抽样核验使用每分区首/中/末工况、帧 0/17/59、首/中/末空间点；范数由原分量用 hypot 计算。抽样未声称再次逐元素核验全部真值，第一次回放已做原始 signed field→cache 的完整逐点核验。", "",
             "预测的 reused_test 是 V1 已看过的旧留出集；全 60 版仅报告 training_fit。未再次推理、训练、选型、修改模型或以测试结果改选择。"]
    if not result["all_pass"]:
        lines += ["", "失败项："]+["- "+str(c) for c in checks if not c["passed"]]
    output.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(all_pass=result["all_pass"], checks=len(checks), artifacts=len(artifacts), bytes=total_bytes)))
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
