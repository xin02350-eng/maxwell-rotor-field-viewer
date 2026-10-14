"""Eight frozen-result V3 figures. CSV-only replay; no model training or selection."""
from pathlib import Path
import argparse,json,os,shutil
import numpy as np
import matplotlib.pyplot as plt
from v2_plot_primitives import read_json,digest,write_json,write_csv,read_csv,nums,save,draw as standard_draw,COLORS


def metric(y,p):
    e=p-y;den=float(np.sum(y*y));sse=float(np.sum(e*e))
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),relative_L2_pct=float(100*np.sqrt(sse/den)) if den else None)


def draw(spec,rows,out):
    kind=spec["kind"];par=spec["parameters"]
    if kind in ("cloud","height"):return standard_draw(spec,rows,out)
    if kind=="comparison":
        fig,axes=plt.subplots(1,2,figsize=(10,4.7));fig.subplots_adjust(bottom=.25,top=.85,wspace=.28)
        for ax,key,label in zip(axes,("relative_L2_pct","max_abs_T"),("Global planar |B| relative L2 (%)","Maximum absolute magnitude error (T)")):
            values=nums(rows,key);bars=ax.bar(np.arange(len(rows)),values,color=["#999999",COLORS[0],COLORS[1],COLORS[2]])
            ax.set_xticks(np.arange(len(rows)),[r["label"] for r in rows],rotation=12);ax.set_ylabel(label);ax.set_ylim(0,max(values)*1.2)
            for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v,f"{v:.4f}",ha="center",va="bottom",fontsize=9)
            if key=="max_abs_T":ax.axhline(1,ls=":",color="#777",label="1 T");ax.legend()
        fig.suptitle("Rotor: identical nine reused-test conditions and planar magnitude",fontsize=12,fontweight="bold")
    elif kind=="cases":
        fig,axes=plt.subplots(1,2,figsize=(11,4.8));fig.subplots_adjust(bottom=.24,top=.85,wspace=.25)
        for j,model in enumerate(dict.fromkeys(r["label"] for r in rows)):
            data=sorted([r for r in rows if r["label"]==model],key=lambda r:int(r["case_id"]))
            for ax,key in zip(axes,("relative_L2_pct","max_abs_T")):
                ax.plot(nums(data,"case_id"),nums(data,key),"o-",lw=1.2,ms=4,color=["#999999",*COLORS][j],label=model)
        axes[0].set_ylabel("Case planar |B| relative L2 (%)");axes[1].set_ylabel("Case maximum absolute error (T)")
        for ax in axes:ax.set_xlabel("Original case ID");ax.set_xticks(sorted(set(int(r["case_id"]) for r in rows)));ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle("All nine reused-test cases retained; no difficult case omitted",fontsize=12,fontweight="bold")
    elif kind=="fitting":
        fig,axes=plt.subplots(1,2,figsize=(10,4.7));fig.subplots_adjust(bottom=.25,top=.83,wspace=.25)
        for ax,key,label in zip(axes,("relative_L2_pct","max_abs_T"),("All60 training-fit relative L2 (%)","All60 training-fit maximum error (T)")):
            for j,version in enumerate(("V2 fitting NN","V3 calibrated head")):
                data=[r for r in rows if r["label"]==version];x=np.arange(3)+(j-.5)*.32;v=nums(data,key)
                ax.bar(x,v,.3,color=["#999999",COLORS[0]][j],label=version)
                for xx,z in zip(x,v):ax.text(xx,z*1.5,f"{z:.2e}",ha="center",va="bottom",fontsize=7,rotation=0)
            ax.set_yscale("log");ax.set_ylim(1e-16,1);ax.set_xticks(range(3),["rotor |B|","stator |B|","airgap Br"]);ax.set_ylabel(label);ax.legend(fontsize=8);ax.grid(axis="y",alpha=.15)
        fig.suptitle("Known-data fitting only: trained hidden layers + least-squares output head",fontsize=11,fontweight="bold")
    elif kind=="diagnostic":
        fig,axes=plt.subplots(1,2,figsize=(10,4.8));fig.subplots_adjust(bottom=.24,top=.85,wspace=.32)
        temporal=[r for r in rows if r["type"]=="temporal"];shares=nums(temporal,"SSE_pct")
        axes[0].bar([0,1],shares,color=[COLORS[0],COLORS[1]]);axes[0].set_xticks([0,1],["Time-mean\nerror offset","Time-varying\nerror residual"]);axes[0].set_ylabel("Validation error SSE share (%)");axes[0].set_ylim(0,108)
        for x,v in enumerate(shares):axes[0].text(x,v+2,f"{v:.3f}%",ha="center")
        concentration=[r for r in rows if r["type"]=="concentration"];x=nums(concentration,"point_count");v=nums(concentration,"SSE_pct")
        axes[1].semilogx(x,v,"o-",color=COLORS[2]);axes[1].set(xlabel="Highest-error spatial point count (out of 1800)",ylabel="Cumulative validation error SSE (%)",ylim=(0,104));axes[1].grid(alpha=.2)
        ninety=next(r for r in concentration if int(r["point_count"])==90)
        axes[1].annotate(f"90 points (5%): {float(ninety['SSE_pct']):.3f}%",xy=(90,float(ninety["SSE_pct"])),xytext=(10,65),arrowprops=dict(arrowstyle="->",color="#444"),fontsize=9)
        fig.suptitle("Frozen V2 validation residual diagnosis; not a mesh-noise finding",fontsize=12,fontweight="bold")
    else:raise ValueError(kind)
    save(fig,out/spec["name"],spec["caption"])


def load(path,choice,sha):
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    meta=json.loads(str(a["metadata_json"]));y=a["truth"].astype(float);p=a["prediction"].astype(float)
    if meta["choices_sha256"]!=sha or meta["model_sha256"]!=choice["model_sha256"]:raise ValueError("Frozen model/data mismatch")
    if y.shape!=p.shape or y.shape[-1]!=len(a["point_id"]) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError("Scalar original-precision data required")
    a.update(truth=y,prediction=p,choice=choice,metadata=meta)
    return a


def tables(a,out):
    y,p=a["truth"],a["prediction"];choice=a["choice"];prefix=dict(model_id=choice["id"],branch=choice["branch"],region=choice["region"],quantity=choice["quantity"])
    case=[dict(prefix,case_id=int(cid),**metric(y[c],p[c])) for c,cid in enumerate(a["case_ids"])]
    frame=[dict(prefix,case_id=int(cid),frame_index=t,theta_m_absolute_deg=float(a["theta"][t]),**metric(y[c,t],p[c,t])) for c,cid in enumerate(a["case_ids"]) for t in range(y.shape[1])]
    point=[dict(prefix,point_id=int(pid),x_ref_mm=float(a["xy"][j,0]),y_ref_mm=float(a["xy"][j,1]),**metric(y[:,:,j],p[:,:,j])) for j,pid in enumerate(a["point_id"])]
    for name,rows in (("case",case),("frame",frame),("point",point)):write_csv(out/(choice["id"]+"_"+name+"_metrics.csv"),rows)
    c,t,j=np.unravel_index(np.argmax(np.abs(p-y)),y.shape)
    worst=dict(model_id=choice["id"],case_id=int(a["case_ids"][c]),frame_index=int(t),theta_m_absolute_deg=float(a["theta"][t]),point_id=int(a["point_id"][j]),
        truth_T=float(y[c,t,j]),prediction_T=float(p[c,t,j]),absolute_error_T=float(abs(p[c,t,j]-y[c,t,j])))
    return dict(prefix,**metric(y,p)),case,worst


def frame_rows(a,c,t,label):
    xy=a["xy"].copy();theta=np.deg2rad(float(a["theta"][t]))
    if a["choice"]["region"]=="rotor":xy=xy@np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    return [dict(representative=label,component="|B|",case_id=int(a["case_ids"][c]),frame_index=t,point_id=int(pid),theta_m_absolute_deg=float(a["theta"][t]),
        x_ref_mm=float(a["xy"][j,0]),y_ref_mm=float(a["xy"][j,1]),x_world_mm=float(xy[j,0]),y_world_mm=float(xy[j,1]),truth_T=float(a["truth"][c,t,j]),
        prediction_T=float(a["prediction"][c,t,j]),absolute_error_T=float(abs(a["prediction"][c,t,j]-a["truth"][c,t,j]))) for j,pid in enumerate(a["point_id"])]


def build(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);data=out/"data";data.mkdir(exist_ok=True)
    freeze=root/"both_versions_choices_frozen.json";ch=read_json(freeze);sha=digest(freeze)
    if sha!=(root/"both_versions_choices_frozen.sha256").read_text().strip():raise ValueError("Freeze SHA mismatch")
    qa_path=root/"qa"/"scientific_qa.json";qa=read_json(qa_path)
    if not qa.get("all_checks_pass",qa.get("pass",False)):raise ValueError("Scientific QA not completed/passed")
    choices={m["id"]:m for m in ch["models"]};sources=[freeze,qa_path];sets={};globals_=[];allcases={};worsts=[];recipe=[]
    for mid,choice in choices.items():
        part="training_fit" if choice["branch"]=="fitting_full60" else "reused_test";path=root/"evaluation"/mid/(part+"_predictions.npz")
        a=load(path,choice,sha);sources.append(path);sets[mid]=a;g,case,worst=tables(a,data);globals_.append(g);allcases[mid]=case;worsts.append(worst)
    def add(kind,name,rows,parameters,caption):
        path=data/(name+".csv");write_csv(path,rows);s=dict(name=name,kind=kind,input_csv=path.relative_to(out).as_posix(),input_sha256=digest(path),parameters=parameters,caption=caption)
        recipe.append(s);draw(s,rows,out)
    primary=sets["prediction_rotor_magnitude"];byglobal={r["model_id"]:r for r in globals_}
    old_path=root.parent/"new60_20261014/experiments/rotor/final_evaluation/selected_predictions.npz";sources.append(old_path)
    with np.load(old_path,allow_pickle=False) as z:old={k:z[k] for k in ("truth","prediction","case_ids","point_id")}
    order=[old["case_ids"].tolist().index(int(cid)) for cid in primary["case_ids"]];y=old["truth"][order].astype(float);p=old["prediction"][order].astype(float)
    y=np.hypot(y[...,0::2],y[...,1::2]);p=np.hypot(p[...,0::2],p[...,1::2])
    if not np.array_equal(old["point_id"],primary["point_id"]) or not np.allclose(y,primary["truth"],atol=1e-7,rtol=1e-7):raise ValueError("V1/V3 truth identity mismatch")
    comparison=[dict(label="V1",**metric(y,p))];casecomparison=[dict(label="V1",case_id=int(cid),**metric(y[c],p[c])) for c,cid in enumerate(primary["case_ids"])]
    for mid,label in (("prediction_rotor_v2_kernel","V2 kernel"),("prediction_rotor_magnitude","V3 mixed"),("prediction_rotor_gated_nn","V3 pure NN")):
        comparison.append(dict(label=label,**{k:byglobal[mid][k] for k in ("rmse_T","mae_T","max_abs_T","relative_L2_pct")}));casecomparison.extend(dict(row,label=label) for row in allcases[mid])
    scope="Same nine previously exposed test cases; reused-test comparison, not fresh blind. All values are planar sqrt(Br^2+Bt^2), not signed vector error. Frozen models were not reselected after this evaluation."
    add("comparison","01_rotor_same_quantity_v1_v2_v3",comparison,{},scope+" V3 mixed=75% neural ensemble+25% V2 kernel; pure NN reported independently.")
    add("cases","02_complete_reused_test_case_errors",casecomparison,{},scope+" Every case retained; average improvement does not remove large local errors.")
    common_field=[0,max(float(primary["truth"].max()),float(primary["prediction"].max()))];error=[0,float(np.max(np.abs(primary["prediction"]-primary["truth"])))]
    c2=primary["case_ids"].tolist().index(2);rows=frame_rows(primary,c2,0,"case02_frame0")
    def cloud(name,rows,label,field,error,title,caption,highlight=None):
        add("cloud",name,rows,dict(region="rotor",panels=[dict(representative=label,component="|B|",label=f"case {rows[0]['case_id']:02d}, frame {rows[0]['frame_index']}\nabsolute theta={rows[0]['theta_m_absolute_deg']:g} deg")],
            scales={"|B|":dict(field=field,error=error)},title=title,**({"highlight_point_id":highlight} if highlight is not None else {})),caption)
    geometry="Actual sampled45-degree sector only; no mirroring or inferred surfaces. World xy=reference points rotated by absolute mechanical theta; units mm. Truth and model share field color limits."
    cloud("03_reused_case02_frame0_2d",rows,"case02_frame0",common_field,error,"V3 mixed | reused case02, frame0 | planar magnitude",scope+" "+geometry)
    w=next(r for r in worsts if r["model_id"]=="prediction_rotor_magnitude");ci=primary["case_ids"].tolist().index(w["case_id"]);worstrows=frame_rows(primary,ci,w["frame_index"],"worst_single_error_frame")
    cloud("04_maximum_local_error_frame_2d",worstrows,"worst_single_error_frame",common_field,error,f"V3 mixed | maximum single-point error={w['absolute_error_T']:.6f} T",scope+f" Maximum point ID={w['point_id']}; this is the frame containing the largest individual error, not necessarily largest frame RMSE. "+geometry,highlight=w["point_id"])
    add("height","05_reused_case02_frame0_3d",rows,dict(representative="case02_frame0",field=common_field,error=error,title="V3 mixed | reused case02, frame0, absolute theta9deg"),scope+" "+geometry+" Height represents planar magnetic magnitude, not a3D volume field. Error height is mT.")
    fitrows=[];v2fit=root.parent/"new60_opt_v2/neural_route/fitting_full60/fitting_summary.json";sources.append(v2fit);oldfit=read_json(v2fit)["rows"]
    for mid in ("fitting_rotor_magnitude","fitting_stator_magnitude","fitting_airgap_Br"):
        r=byglobal[mid];prior=next(x for x in oldfit if x["region"]==r["region"])
        fitrows.append(dict(label="V2 fitting NN",region=r["region"],**{k:prior[k] for k in ("rmse_T","mae_T","max_abs_T","relative_L2_pct")}))
        fitrows.append(dict(label="V3 calibrated head",region=r["region"],**{k:r[k] for k in ("rmse_T","mae_T","max_abs_T","relative_L2_pct")}))
    add("fitting","06_all60_fitting_roundoff_scale",fitrows,{},"All60 existing-condition training fit only, including former validation/test cases. V3 keeps trained nonlinear hidden layers and solves the linear output head by least squares. Near-roundoff fitting does not establish unseen-condition prediction or physical exactness. Log axes retain nonzero scientific-notation errors; no rounded zeros.")
    fit=sets["fitting_rotor_magnitude"];fc=fit["case_ids"].tolist().index(2);frows=frame_rows(fit,fc,0,"known_case02_frame0");ferror=[0,max(float(np.max(np.abs(fit["prediction"]-fit["truth"]))),1e-20)]
    cloud("07_all60_fitted_case02_frame0_2d",frows,"known_case02_frame0",common_field,ferror,"V3 fitting | known case02, frame0 | training reconstruction only","All60 training fit, not heldout prediction. Residual color scale is in T and scientific notation at floating-point roundoff. "+geometry)
    diag_path=root/"diagnostics/diagnosis_v2_rotor.json";temporal_path=root/"diagnostics/temporal_static_error_decomposition.json";sources.extend([diag_path,temporal_path])
    dd=read_json(diag_path)["models"]["prediction_rotor_magnitude"]["validation"];tt=read_json(temporal_path)["models"]["prediction_rotor_magnitude"]["validation"]
    diagrows=[dict(type="temporal",label="time_mean_offset",SSE_pct=tt["error_temporal_mean_SSE_pct"]),dict(type="temporal",label="time_varying",SSE_pct=tt["error_temporal_fluctuation_SSE_pct"])]
    diagrows.extend(dict(type="concentration",point_count=int(k),SSE_pct=v) for k,v in dd["point_SSE_share_pct"].items())
    add("diagnostic","08_validation_bias_and_spatial_concentration",diagrows,{},"Frozen V2 validation9 diagnosis only, used to explain the optimization direction. Time-mean error offset accounts for94.798% of SSE;90 of1800 spatial points account for91.792%. These are descriptive residual statistics, not proof of mesh noise or permission to delete points.")
    write_json(out/"same_quantity_comparison.json",dict(scope=scope,rows=comparison));write_json(out/"worst_locations.json",worsts);write_json(out/"global_metrics.json",globals_)
    for source in (Path(__file__),Path(__file__).with_name("v2_plot_primitives.py")):shutil.copyfile(source,out/source.name)
    rec=dict(version="V3_FIGURES",choices_sha256=sha,figures=recipe,scientific_inputs=[dict(path=Path(os.path.relpath(p,out)).as_posix(),sha256=digest(p)) for p in dict.fromkeys(sources)],
        plotting_sources=[dict(path=p.name,sha256=digest(p)) for p in (Path(__file__),Path(__file__).with_name("v2_plot_primitives.py"))],
        reproduction="python build_figures_v3.py --replay-recipe figure_recipe.json --output regenerated --only 05_reused_case02_frame0_3d",evaluation_scope="reused test and all60 fitting are separate")
    write_json(out/"figure_recipe.json",rec)
    report(out,comparison,fitrows,w,diagrows)
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    print(json.dumps(dict(groups=len(recipe),bytes=size,comparison=comparison,worst=w),ensure_ascii=False))


def report(out,comparison,fitrows,w,diag):
    lines=["# V3科学图与结果说明", "", "本轮不改变冻结模型。预测评估复用原9个测试工况，标签此前已曝光，不能称新的独立盲测；拟合版明确使用全部60个已知工况，两类误差分别统计。", "", "## 转子同量比较", "", "以下均为二维范数sqrt(Br²+Bt²)，不混用Br/Bt向量误差。", "", "| 模型 | relative L2 / % | RMSE / T | MAE / T | 最大绝对误差 / T |", "|---|---:|---:|---:|---:|"]
    for r in comparison:lines.append(f"| {r['label']} | {r['relative_L2_pct']:.8f} | {r['rmse_T']:.8g} | {r['mae_T']:.8g} | {r['max_abs_T']:.8g} |")
    previous=next(r for r in comparison if r["label"]=="V2 kernel");current=next(r for r in comparison if r["label"]=="V3 mixed")
    lines += ["", "V3混合模型是75%纯神经网络集成与25%V2核模型的固定组合，不称纯NN。纯NN结果独立列出。更小的全场平均误差不意味着每个位置更准确，最大局部误差没有删除。",
        f"相对V2核模型，V3混合模型的relative L2仅下降{previous['relative_L2_pct']-current['relative_L2_pct']:.6f}个百分点，而最大绝对误差由{previous['max_abs_T']:.6f} T升至{current['max_abs_T']:.6f} T。不能据此宣称已获得很小的整体预测误差，也未达到小于1%的预测目标。", "",
        f"V3混合最大单点误差为{w['absolute_error_T']:.8g} T，位于case{w['case_id']:02d}、frame{w['frame_index']}、绝对机械角{w['theta_m_absolute_deg']:g}°、point ID {w['point_id']}。04号图展示该点所在帧，全部case/frame/point四指标CSV保留在data目录。", "", "## 全60拟合：仅训练内复现", "", "| 区域 | 方法 | relative L2 / % | 最大绝对误差 / T |", "|---|---|---:|---:|"]
    for r in fitrows:lines.append(f"| {r['region']} | {r['label']} | {r['relative_L2_pct']:.6e} | {r['max_abs_T']:.6e} |")
    lines += ["", "V3保留已经训练的非线性神经网络隐藏层，使用全部60工况通过最小二乘校准线性输出层。误差接近浮点舍入水平，仅说明对这些有限已知数据的数值复现，不是对新工况的预测精度、物理真值精度或独立验证；不能把小数格式舍入出的0称为绝对无误差。", "",
        "## 验证残差诊断", "", "08号图统计的是冻结V2模型在9个验证工况上的残差：时间均值偏差贡献94.798415%的平方误差，时间波动残差贡献5.201585%；最高误差90个空间点（1800点的5%）贡献91.792437%。时间均值项与几何热点占比不是同一个量。该集中现象不证明网格噪声，也不构成删点依据。", "",
        "## 图形和复现口径", "", "仅保留原45°真实采样扇区，不镜像补全电机、不跨孔造面。转子世界坐标按绝对机械角旋转；3D为磁密高度散点图，不是三维体积场。case02-frame0在预测版属于复用测试，在拟合版属于训练数据，图注明确区分。拟合残差用科学计数和独立误差色标显示。", "",
        "8组图各含300dpi PNG、可编辑PDF/SVG与CSV。figure_recipe.json记录CSV与科学数据的相对路径及SHA；只需两份绘图源码、recipe和绘图CSV即可单图复现，无需再次读取模型或完整标签。示例：`python build_figures_v3.py --replay-recipe figure_recipe.json --output regenerated --only 05_reused_case02_frame0_3d`。", ""]
    for f in read_json(out/"figure_recipe.json")["figures"]:lines.append(f"- [{f['name']}]({f['name']}.png) / [PDF]({f['name']}.pdf) / [SVG]({f['name']}.svg)")
    (out/"V3科学说明.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path(__file__).parent);p.add_argument("--output",type=Path,required=True);p.add_argument("--replay-recipe",type=Path);p.add_argument("--only")
    a=p.parse_args()
    if a.replay_recipe:
        a.output.mkdir(parents=True,exist_ok=True);d=read_json(a.replay_recipe)
        for s in d["figures"]:
            if a.only and s["name"]!=a.only:continue
            source=a.replay_recipe.parent/s["input_csv"]
            if digest(source)!=s["input_sha256"]:raise ValueError("CSV SHA mismatch")
            draw(s,read_csv(source),a.output)
    else:build(a.root,a.output)
