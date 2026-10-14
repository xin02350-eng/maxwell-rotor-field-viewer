"""Audit all60 fitting outputs only; never feeds prediction-model selection."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from fitting_full60 import predict,digest


def metric(p,y):
    e=np.asarray(p,dtype=float)-np.asarray(y,dtype=float);den=np.sum(y*y);sse=np.sum(e*e)
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),
        relative_L2_pct=float(100*np.sqrt(sse/den)) if den else None)


def write_json(p,data):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def write_csv(path,rows):
    with Path(path).open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(cache_dir,root):
    from threadpoolctl import threadpool_limits
    root=Path(root);cache_dir=Path(cache_dir);summary=[];cases=[];frames=[];points=[];worst=[];checks=[]
    with threadpool_limits(limits=2):
        for region in ("rotor","stator","airgap"):
            folder=root/region;result=json.loads((folder/"result.json").read_text(encoding="utf-8"))
            cache=cache_dir/(region+".npz")
            if result["branch"]!="fitting_full60" or digest(cache)!=result["cache_sha256"]:raise AssertionError("Branch/cache mismatch")
            with np.load(cache,allow_pickle=False) as z:
                raw=z["field"].astype(float);conditions=z["conditions"];ids=z["case_ids"];pid=z["point_id"];xy=z["xy"];theta=z["theta"]
            quantity=result["quantity"];truth=np.hypot(raw[...,0::2],raw[...,1::2]) if quantity=="|B|" else raw;del raw
            with np.load(folder/"fitting_predictions.npz",allow_pickle=False) as z:
                saved=z["raw_prediction"];saved_ids=z["case_ids"]
            if not np.array_equal(ids,saved_ids):raise AssertionError("Case mapping mismatch")
            replay=predict(folder/"model.npz",conditions,nonnegative=False)
            if replay.shape!=truth.shape or not np.isfinite(replay).all() or not np.allclose(replay,saved,rtol=1e-12,atol=1e-12):raise AssertionError("NumPy field replay mismatch")
            global_metric=metric(replay,truth)
            if not all(np.isclose(global_metric[k],result["fitting_raw"][k],rtol=1e-12,atol=1e-12) for k in global_metric):raise AssertionError("Fitting metric replay mismatch")
            prefix=dict(branch="all60_fitting",region=region,quantity=quantity,units="T",case_role="fitting")
            summary.append(dict(prefix,**global_metric,raw_negative_count=result["raw_negative_count"],parameter_count=result["parameter_count"],best_step=result["best_step"],training_seconds=result["elapsed_seconds"]))
            for c,cid in enumerate(ids):
                cases.append(dict(prefix,case_id=int(cid),I_RMS_A=float(conditions[c,0]),native_Thet_deg=float(conditions[c,1]),**metric(replay[c],truth[c])))
                for t in range(truth.shape[1]):frames.append(dict(prefix,case_id=int(cid),frame_index=t,theta_m_absolute_deg=float(theta[t]),**metric(replay[c,t],truth[c,t])))
            for p,point_id in enumerate(pid):points.append(dict(prefix,point_id=int(point_id),x_mm=float(xy[p,0]),y_mm=float(xy[p,1]),**metric(replay[...,p],truth[...,p])))
            c,t,p=np.unravel_index(np.argmax(np.abs(replay-truth)),truth.shape)
            worst.append(dict(prefix,case_id=int(ids[c]),frame_index=int(t),theta_m_absolute_deg=float(theta[t]),point_id=int(pid[p]),
                reference_xy_mm=xy[p].tolist(),truth_T=float(truth[c,t,p]),raw_prediction_T=float(replay[c,t,p]),absolute_error_T=float(abs(replay[c,t,p]-truth[c,t,p]))))
            checks.append(dict(region=region,shape=list(truth.shape),model_sha256=digest(folder/"model.npz"),model_hash_matches_result=digest(folder/"model.npz")==result["model_sha256"],
                prediction_sha256=digest(folder/"fitting_predictions.npz"),cache_sha256=digest(cache),numpy_replay_max_abs_difference_T=float(np.max(np.abs(replay-saved))),
                metric_recomputed_matches=True,raw_projection_used_for_metrics=False,all60_training_labels_used=True,heldout_generalization_evaluated=False))
    write_csv(root/"fitting_summary.csv",summary);write_csv(root/"fitting_case_metrics.csv",cases);write_csv(root/"fitting_frame_metrics.csv",frames);write_csv(root/"fitting_point_metrics.csv",points)
    write_json(root/"fitting_summary.json",dict(branch="fitting_full60",rows=summary,scope="All60 existing conditions; fitting error, not heldout prediction",source_sha256=digest(__file__)))
    write_json(root/"fitting_worst_locations.json",worst);write_json(root/"replay_check.json",dict(checks=checks,numpy_only_prediction="torch" not in sys.modules,source_sha256=digest(__file__)))
    lines=["# 全部 60 工况拟合版", "", "这是一条独立拟合分支，明确使用全部 60 个既有工况、每工况 60 个已保存转子位置与固定空间采样点，其中包含预测研究原来的验证和测试工况。下列数值是训练内拟合误差，不能称为未见工况预测误差。预测分支的选择文件已在读取全量标签之前冻结。", "",
        "模型为全 60 工况中心化整周期 POD（rank 59）与 256×256×256 的 tanh MLP。神经网络通过 PyTorch Adam 对原始 T 单位场平方误差训练，没有按工况 ID 检索磁密数组。NumPy 推理以原生电流 RMS 与 Thet 为输入，输出整个已保存 60 帧；新工况输出质量未由这个全量拟合实验验证。", "",
        "| 区域 | 目标量 | 原始拟合 RMSE / T | MAE / T | 最大绝对误差 / T | relative L2 |", "|---|---|---:|---:|---:|---:|"]
    for row in summary:lines.append(f"| {row['region']} | {row['quantity']} | {row['rmse_T']:.8g} | {row['mae_T']:.8g} | {row['max_abs_T']:.8g} | {row['relative_L2_pct']:.6f}% |")
    lines += ["", "以上三者均使用未经非负投影的原始网络输出。转子和定子的目标是 sqrt(Br²+Bt²) 的平面内磁密大小；气隙是带符号 Br，两类 relative L2 不可混称向量磁密误差。可选非负投影仅适用于磁密大小，实际原始负值数量和投影后指标另存于各区域 result.json。", "",
        "每区域均达到事先设定的训练内 relative L2≤0.05% 停止标准。很小的整体误差并不意味着每个测点误差同样小，表中最大误差及 fitting_worst_locations.json 保留了局部最差位置。", "",
        "交付包括各区域 model.npz 与 SHA、fitting_predictions.npz、history.csv、result.json、protocol.json；汇总 CSV 分别按整体、工况、工况×帧和空间点列出 RMSE、MAE、最大绝对误差和 relative L2。replay_check.json 记录纯 NumPy 权重重载与原精度预测/指标核对。", "",
        "坐标与原始数据一致；转子仅为真实采样扇区与随体参考坐标。模型输出不是额外 Maxwell 仿真，也未证明任意新转速、任意新转子角或新网格点的准确性。"]
    (root/"FITTING_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(rows=summary,checks=checks),ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache-dir",type=Path,required=True);p.add_argument("--fit-root",type=Path,required=True)
    a=p.parse_args();run(a.cache_dir,a.fit_root)
