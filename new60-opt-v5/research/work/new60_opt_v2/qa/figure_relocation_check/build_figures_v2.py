"""Frozen V2 scientific figures and portable CSV/recipe replay; never trains models."""
from pathlib import Path
import argparse,csv,hashlib,json,textwrap,shutil,os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

COLORS=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
    "legend.fontsize":8,"legend.frameon":False,"axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"svg.fonttype":"none","savefig.dpi":300,"axes.unicode_minus":True})


def read_json(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def write_json(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def write_csv(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def read_csv(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def nums(rows,key):return np.array([float(r[key]) for r in rows])
def metric(y,p):
    e=np.asarray(p,dtype=float)-np.asarray(y,dtype=float);den=float(np.sum(y*y));sse=float(np.sum(e*e))
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),max_abs_T=float(np.max(np.abs(e))),relative_L2_pct=100*np.sqrt(sse/den) if den else None)
def save(fig,path,caption):
    fig.text(.02,.015,textwrap.fill(caption,150),fontsize=7,color="#424b55",ha="left",va="bottom")
    for ext in ("png","pdf","svg"):fig.savefig(path.with_suffix("."+ext),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def draw(spec,rows,out):
    kind=spec["kind"];p=spec["parameters"];caption=spec["caption"];name=spec["name"]
    if kind=="cloud":
        panels=p["panels"];fig,axes=plt.subplots(len(panels),3,figsize=(12,2.95*len(panels)+1.15),squeeze=False)
        fig.subplots_adjust(left=.07,right=.96,top=.91,bottom=.12,wspace=.3,hspace=.38)
        for k,panel in enumerate(panels):
            data=[r for r in rows if r["representative"]==panel["representative"] and r["component"]==panel["component"]]
            x,y=nums(data,"x_world_mm"),nums(data,"y_world_mm");comp=panel["component"]
            for j,key in enumerate(("truth_T","prediction_T","absolute_error_T")):
                ax=axes[k,j];limits=p["scales"][comp]["field" if j<2 else "error"]
                cmap=("RdBu_r" if comp in ("Br","Bt") else "viridis") if j<2 else "magma"
                art=ax.scatter(x,y,c=nums(data,key),s=9,cmap=cmap,vmin=limits[0],vmax=limits[1],linewidths=0)
                ax.set_aspect("equal");ax.set_xlabel("World x (mm)" if p["region"]=="rotor" else "x (mm)")
                ax.set_ylabel("World y (mm)" if p["region"]=="rotor" else "y (mm)")
                ax.set_title(["Maxwell "+comp,"Model "+comp,"Absolute "+comp+" error"][j])
                fig.colorbar(art,ax=ax,fraction=.045,pad=.025,label="T")
            axes[k,0].text(-.25,.5,panel["label"],transform=axes[k,0].transAxes,rotation=90,ha="center",va="center",fontsize=8)
        fig.suptitle(p["title"],fontsize=13,fontweight="bold")
    elif kind=="height":
        data=[r for r in rows if r["representative"]==p["representative"]];fig=plt.figure(figsize=(12,4.8));fig.subplots_adjust(left=.015,right=.975,top=.86,bottom=.25,wspace=.15)
        for j,key in enumerate(("truth_T","prediction_T","absolute_error_T")):
            ax=fig.add_subplot(1,3,j+1,projection="3d");factor=1000 if j==2 else 1;lo,hi=p["field" if j<2 else "error"];lo*=factor;hi*=factor
            val=nums(data,key)*factor;art=ax.scatter(nums(data,"x_world_mm"),nums(data,"y_world_mm"),val,c=val,s=7,cmap="viridis" if j<2 else "magma",vmin=lo,vmax=hi,depthshade=False)
            ax.set(xlabel="World x (mm)",ylabel="World y (mm)",zlabel="|B| (T)" if j<2 else "",zlim=(lo,hi),title=["Maxwell |B|","Model |B|","Absolute magnitude error"][j])
            if j==2:ax.text2D(.02,.91,"Height: |error| (mT)",transform=ax.transAxes,fontsize=8)
            ax.tick_params(labelsize=7);ax.view_init(28,-60);ax.set_box_aspect((1,1,.75))
            fig.colorbar(art,ax=ax,orientation="horizontal",fraction=.045,pad=.15,label="T" if j<2 else "mT")
        fig.suptitle(p["title"],fontsize=13,fontweight="bold")
    elif kind=="gap":
        fig,axes=plt.subplots(2,2,figsize=(11,6),squeeze=False);fig.subplots_adjust(bottom=.18,top=.9,hspace=.36,wspace=.25)
        for i,rep in enumerate(p["representatives"]):
            data=sorted([r for r in rows if r["representative"]==rep],key=lambda r:float(r["spatial_angle_deg"]))
            phi=nums(data,"spatial_angle_deg");axes[i,0].plot(phi,nums(data,"truth_T"),color=COLORS[0],label="Maxwell Br")
            axes[i,0].plot(phi,nums(data,"prediction_T"),color=COLORS[1],ls="--",label="Model Br");axes[i,0].set(ylabel="Signed Br (T)",title=p["labels"][rep]);axes[i,0].legend()
            axes[i,1].plot(phi,nums(data,"absolute_error_T")*1000,color=COLORS[1]);axes[i,1].set(ylabel="Absolute Br error (mT)",title=p["labels"][rep])
            for ax in axes[i]:ax.set_xlabel("Fixed spatial polar angle (deg)");ax.grid(alpha=.18)
        fig.suptitle(p["title"],fontsize=13,fontweight="bold")
    elif kind=="global":
        fig,axes=plt.subplots(1,2,figsize=(11,4.7));fig.subplots_adjust(bottom=.24,top=.84,wspace=.28)
        for ax,branch in zip(axes,("prediction","fitting_full60")):
            data=[r for r in rows if r["branch"]==branch];x=np.arange(len(data));v=nums(data,"relative_L2_pct")
            bars=ax.bar(x,v,color=COLORS[:len(data)]);ax.set_xticks(x,[r["region"]+"\n"+r["quantity"] for r in data]);ax.set_ylabel("Global relative L2 (%)")
            ax.set_title("Reused test: 9 conditions" if branch=="prediction" else "Training fit: all 60 conditions");ax.set_ylim(0,max(v)*1.2)
            for b,z in zip(bars,v):ax.text(b.get_x()+b.get_width()/2,z,f"{z:.5g}%",ha="center",va="bottom",fontsize=9)
        fig.suptitle("Different evaluation purposes; separate error scales",fontsize=13,fontweight="bold")
    elif kind=="cases":
        fig,axes=plt.subplots(3,2,figsize=(11,8));fig.subplots_adjust(bottom=.15,top=.93,hspace=.4,wspace=.25)
        for i,region in enumerate(("rotor","stator","airgap")):
            for j,branch in enumerate(("prediction","fitting_full60")):
                data=sorted([r for r in rows if r["region"]==region and r["branch"]==branch],key=lambda r:int(r["case_id"]))
                ax=axes[i,j];ax.plot(nums(data,"case_id"),nums(data,"relative_L2_pct"),"o-",color=COLORS[i],ms=3,lw=1)
                ax.set(xlabel="Original case ID",ylabel="Case relative L2 (%)",title=region+" / "+("reused test" if j==0 else "all60 training fit"));ax.grid(alpha=.18)
        fig.suptitle("Every evaluated condition retained",fontsize=13,fontweight="bold")
    elif kind=="training":
        fig,axes=plt.subplots(2,3,figsize=(12,6.3));fig.subplots_adjust(bottom=.18,top=.87,hspace=.32,wspace=.3)
        for j,region in enumerate(("rotor","stator","airgap")):
            data=[r for r in rows if r["region"]==region];x=nums(data,"step")
            axes[0,j].semilogy(x,nums(data,"fitting_relative_L2_pct"),color=COLORS[j]);axes[0,j].axhline(.05,color="#666",ls=":",label="0.05% stopping target")
            axes[0,j].set(title=region+" / "+("Br" if region=="airgap" else "|B|"),ylabel="Raw fitting relative L2 (%)");axes[0,j].legend(fontsize=7)
            grad=nums(data,"gradient_norm");valid=grad>0
            axes[1,j].semilogy(x[valid],grad[valid],color=COLORS[j]);axes[1,j].set(xlabel="Optimizer step",ylabel="Gradient norm")
            for ax in axes[:,j]:ax.grid(alpha=.18)
        fig.suptitle("All60 fitting MLP: actual training diagnostics",fontsize=13,fontweight="bold")
    elif kind=="candidates":
        groups=list(dict.fromkeys(r["family"] for r in rows));fig,ax=plt.subplots(figsize=(10,5));fig.subplots_adjust(left=.24,bottom=.24,right=.96,top=.87)
        for j,group in enumerate(groups):
            data=[r for r in rows if r["family"]==group];v=nums(data,"relative_L2_pct");offset=np.linspace(-.14,.14,len(v))
            ax.scatter(v,j+offset,c=COLORS[j%len(COLORS)],s=26,alpha=.8,label=group)
        ax.set_yticks(np.arange(len(groups)),groups);ax.invert_yaxis();ax.set_xlabel("Validation relative L2 (%); vector and magnitude labelled separately");ax.grid(axis="x",alpha=.2)
        ax.set_title("All 40 condition-MLP runs + 8 learned-coordinate kernel runs",fontsize=12,fontweight="bold")
    elif kind=="v1v2":
        fig,axes=plt.subplots(1,2,figsize=(12,5.4));fig.subplots_adjust(left=.18,bottom=.23,top=.87,wspace=.3)
        labels=[r["region"]+" / "+r["quantity"] for r in rows];y=np.arange(len(rows))
        for ax,key,title in zip(axes,("relative_L2_pct","max_abs_T"),("Global relative L2 (%)","Maximum absolute error (T)")):
            ax.barh(y-.17,[float(r["v1_"+key]) for r in rows],height=.32,color="#999999",label="V1")
            ax.barh(y+.17,[float(r["v2_"+key]) for r in rows],height=.32,color=COLORS[0],label="Frozen V2")
            ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel(title);ax.legend();ax.grid(axis="x",alpha=.15)
        fig.suptitle("Identical nine reused-test conditions and target quantities",fontsize=13,fontweight="bold")
    elif kind=="projection":
        sample=[r for r in rows if r["row_type"]=="negative_sample"];glob=[r for r in rows if r["row_type"]=="global"]
        fig,axes=plt.subplots(1,3,figsize=(12,4.6));fig.subplots_adjust(bottom=.25,top=.83,wspace=.35)
        axes[0].hist(nums(sample,"raw_T")*1000,bins=15,color=COLORS[1]);axes[0].set(xlabel="Negative raw |B| output (mT)",ylabel="Case-frame-point observations",title=f"{len(sample)} negative observations")
        art=axes[1].scatter(nums(sample,"x_mm"),nums(sample,"y_mm"),c=nums(sample,"raw_T")*1000,s=25,cmap="magma");axes[1].set_aspect("equal");axes[1].set(xlabel="x (mm)",ylabel="y (mm)",title="Locations of affected samples");fig.colorbar(art,ax=axes[1],label="Raw output (mT)",fraction=.045)
        axes[2].axis("off");table=[[r["version"],f"{float(r['relative_L2_pct']):.9f}%",f"{float(r['rmse_T']):.8g}"] for r in glob]
        axes[2].table(cellText=table,colLabels=["Output","L2","RMSE (T)"],cellLoc="center",loc="center").scale(1,1.8)
        fig.suptitle("Stator all60 fitting: explicit max(0, raw) output projection",fontsize=12,fontweight="bold")
    else:raise ValueError(kind)
    save(fig,out/name,caption)


def load_prediction(path,choice,freeze_sha):
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    meta=json.loads(str(a["metadata_json"]));y=a["truth"].astype(float);pred=a["prediction"].astype(float);points=len(a["point_id"])
    if meta["choices_sha256"]!=freeze_sha or meta["model_sha256"]!=choice["model_sha256"]:raise ValueError("Frozen identity mismatch")
    if y.shape!=pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError("Invalid scientific array")
    q=choice["quantity"];components=["Br","Bt"] if q=="Br,Bt" else ["Br" if q=="Br" else "|B|"]
    if y.shape[-1]!=points*len(components):raise ValueError("Quantity/point count mismatch")
    a.update(truth=y.reshape(*y.shape[:2],points,len(components)),prediction=pred.reshape(*pred.shape[:2],points,len(components)),
        choice=choice,components=components,quantity=q,metadata=meta,input_path=path)
    return a


def summarize(a,out,authoritative):
    ident=a["choice"];y,p=a["truth"],a["prediction"];g=metric(y,p)
    if not all(np.isclose(g[k],authoritative[k],rtol=1e-10,atol=1e-12) for k in g):raise ValueError("QA metrics mismatch")
    prefix=dict(model_id=ident["id"],branch=ident["branch"],region=ident["region"],quantity="Br,Bt" if a["quantity"]=="Br,Bt" else a["components"][0],partition=str(a["partition"]))
    case=[];frames=[];points=[]
    for c,cid in enumerate(a["case_ids"]):
        case.append(dict(prefix,case_id=int(cid),**metric(y[c],p[c])))
        for t in range(y.shape[1]):frames.append(dict(prefix,case_id=int(cid),frame_index=t,solver_index=int(a["solver_index"][t]),theta_m_absolute_deg=float(a["theta"][t]),time_s=float(a["time_s"][t]),**metric(y[c,t],p[c,t])))
    for j,pid in enumerate(a["point_id"]):
        row=dict(prefix,point_id=int(pid),x_ref_mm=float(a["xy"][j,0]),y_ref_mm=float(a["xy"][j,1]),**metric(y[:,:,j],p[:,:,j]))
        if len(a["components"])==2:
            for k,comp in enumerate(a["components"]):row.update({comp+"_"+key:v for key,v in metric(y[:,:,j,k],p[:,:,j,k]).items()})
        points.append(row)
    for key,rows in (("case",case),("frame",frames),("point",points)):write_csv(out/(ident["id"]+"_"+key+"_metrics.csv"),rows)
    frame_rmse=np.sqrt(np.mean((p-y)**2,axis=(2,3)));worst_case=int(np.argmax([r["rmse_T"] for r in case]));wc=(worst_case,int(frame_rmse[worst_case].argmax()))
    wf=np.unravel_index(frame_rmse.argmax(),frame_rmse.shape);median=np.unravel_index(np.argmin(np.abs(frame_rmse-np.median(frame_rmse))),frame_rmse.shape)
    reps=[("worst_frame",wf),("worst_case_frame",wc),("median_frame",median)]
    if wf==wc:reps=[("worst_frame_and_case",wf),("median_frame",median)]
    spatial=np.linalg.norm(p-y,axis=-1);c,t,j=np.unravel_index(spatial.argmax(),spatial.shape)
    worst=dict(model_id=ident["id"],worst_case=case[worst_case],worst_frame=max(frames,key=lambda r:r["rmse_T"]),
        worst_aggregate_point=max(points,key=lambda r:r["rmse_T"]),worst_single_spatial_error=dict(case_id=int(a["case_ids"][c]),frame_index=int(t),point_id=int(a["point_id"][j]),theta_m_absolute_deg=float(a["theta"][t]),
            error_norm_T=float(spatial[c,t,j]),definition="vector error norm" if len(a["components"])==2 else "absolute scalar error"))
    return dict(prefix,**g),case,reps,worst


def point_rows(a,reps):
    rows=[];panels=[];labels={}
    for label,(c,t) in reps:
        xy=a["xy"].astype(float);angle=np.deg2rad(float(a["theta"][t]))
        if a["choice"]["region"]=="rotor":xy=xy@np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        readable=label.replace("_"," ")+f"\ncase {int(a['case_ids'][c]):02d}, frame {t}, {a['theta'][t]:g} deg"
        labels[label]=readable
        for k,comp in enumerate(a["components"]):
            panels.append(dict(representative=label,component=comp,label=readable))
            for j,pid in enumerate(a["point_id"]):
                yy=float(a["truth"][c,t,j,k]);pp=float(a["prediction"][c,t,j,k])
                rows.append(dict(representative=label,component=comp,case_id=int(a["case_ids"][c]),frame_index=int(t),theta_m_absolute_deg=float(a["theta"][t]),time_s=float(a["time_s"][t]),point_id=int(pid),
                    x_ref_mm=float(a["xy"][j,0]),y_ref_mm=float(a["xy"][j,1]),x_world_mm=float(xy[j,0]),y_world_mm=float(xy[j,1]),
                    spatial_angle_deg=float(np.rad2deg(np.arctan2(a["xy"][j,1],a["xy"][j,0]))),truth_T=yy,prediction_T=pp,absolute_error_T=abs(pp-yy),
                    spatial_error_norm_T=float(np.linalg.norm(a["prediction"][c,t,j]-a["truth"][c,t,j]))))
    return rows,panels,labels


def build(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);data=out/"data";data.mkdir(exist_ok=True)
    freeze=root/"both_versions_choices_frozen.json";choices=read_json(freeze);freeze_sha=digest(freeze)
    if freeze_sha!=(root/"both_versions_choices_frozen.sha256").read_text().strip():raise ValueError("Freeze hash mismatch")
    reports=[read_json(root/"qa"/branch/"replay_report.json") for branch in ("prediction_replay","fitting_replay")]
    if not all(r["all_checks_pass"] and r["choices_sha256"]==freeze_sha for r in reports):raise ValueError("Final QA did not pass")
    authoritative={r["id"]:r for report in reports for r in report["results"]};byid={r["id"]:r for r in choices["models"]}
    recipe=[];sources=[freeze,*[root/"qa"/b/"replay_report.json" for b in ("prediction_replay","fitting_replay")]];datasets={};globals_=[];allcases=[];worsts=[];counter=0
    def add(kind,stem,rows,parameters,caption):
        nonlocal counter
        counter+=1;name=f"{counter:02d}_{stem}";path=data/(name+".csv");write_csv(path,rows)
        spec=dict(name=name,kind=kind,input_csv=path.relative_to(out).as_posix(),input_sha256=digest(path),parameters=parameters,caption=caption)
        recipe.append(spec);draw(spec,rows,out)
    scalar_ids=["prediction_rotor_magnitude","prediction_stator_magnitude","prediction_airgap_Br","fitting_rotor","fitting_stator","fitting_airgap"]
    for mid in [*scalar_ids,"prediction_rotor_vector","prediction_stator_vector"]:
        choice=byid[mid];partition="training_fit" if choice["branch"]=="fitting_full60" else "reused_test"
        path=root/"evaluation"/mid/(partition+"_predictions.npz");sources.append(path);a=load_prediction(path,choice,freeze_sha);datasets[mid]=a
        g,case,reps,worst=summarize(a,data,authoritative[mid]["metrics"][partition]["global_metrics"]);worsts.append(worst)
        if mid in scalar_ids:globals_.append(g);allcases.extend(case)
        else:reps=reps[:1]
        rows,panels,labels=point_rows(a,reps);scales={}
        for k,comp in enumerate(a["components"]):
            lo=min(float(a["truth"][...,k].min()),float(a["prediction"][...,k].min()));hi=max(float(a["truth"][...,k].max()),float(a["prediction"][...,k].max()))
            if comp in ("Br","Bt"):hi=max(abs(lo),abs(hi));lo=-hi
            else:lo=min(0,lo)
            scales[comp]=dict(field=[lo,hi],error=[0,float(np.max(np.abs(a["prediction"][...,k]-a["truth"][...,k])))])
        scope="all60 training fit" if partition=="training_fit" else "nine reused-test cases (not fresh blind)"
        caption=f"V2 frozen {choice['method']} | {scope} | actual sampled45-degree sector; no mirroring or invented surfaces. Units T; xy mm. Rotor world xy uses absolute mechanical angle. Equal point/component weights."
        if mid=="fitting_stator":caption+=" Model output includes explicit max(0,raw);104 raw negative observations audited separately."
        a.update(figure_rows=rows,reps=reps,labels=labels,scales=scales,caption=caption)
        add("cloud",mid+"_2d",rows,dict(region=choice["region"],panels=panels,scales=scales,title=choice["region"]+" / "+scope+" / "+("Br and Bt" if len(a["components"])==2 else a["components"][0])),caption)
    for mid in ("prediction_rotor_magnitude","fitting_rotor"):
        a=datasets[mid];rep="median_frame";scale=a["scales"]["|B|"]
        add("height",mid+"_3d",[r for r in a["figure_rows"] if r["representative"]==rep],dict(representative=rep,field=scale["field"],error=scale["error"],title=a["labels"][rep].replace("\n"," | ")+" / "+mid),a["caption"]+" Height encodes magnetic magnitude, not a 3D volume field. Truth and model share z/color limits; error height is mT.")
    for mid in ("prediction_airgap_Br","fitting_airgap"):
        a=datasets[mid];reps=[a["reps"][0][0],"median_frame"]
        add("gap",mid+"_spatial_angle",a["figure_rows"],dict(representatives=reps,labels=a["labels"],title=mid+" / signed Br profiles"),a["caption"]+" Horizontal axis is measurement-point polar angle, not rotor position; all900 sampled points retained.")
    add("global","separate_fitting_and_prediction_metrics",globals_,{},"V2 | different label access and purposes. Fitting uses all60 training conditions; prediction uses nine reused-test conditions after freezing. Quantities and percentage denominators differ across regions; no combined score is implied.")
    add("cases","all_evaluated_case_errors",allcases,{},"V2 | no difficult case removed. Prediction charts use nine reused-test cases; fitting charts use all60 training cases. Stator/rotor are planar magnitude; airgap is signed Br.")
    training=[]
    for region in ("rotor","stator","airgap"):
        path=root/"neural_route"/"fitting_full60"/region/"history.csv";sources.append(path)
        training.extend(dict(row,region=region) for row in read_csv(path))
    add("training","all60_mlp_training",training,{},"V2 all60 fitting only | 256x256x256 tanh MLP,148795 trainable parameters per region. Curves show actual raw original-T training error and gradient, not validation or test. Adam reached the predeclared0.05% threshold; no LBFGS was needed.")
    candidate=[];path=root/"neural_route"/"prediction_route_results.json";sources.append(path)
    names={"rotor":"Cycle NN / Br,Bt","rotor_magnitude":"Cycle NN / |B|","rotor_magnitude_blocks":"Time-block NN / |B|","rotor_spatial_magnitude":"Spatial-POD NN / |B|"}
    for r in read_json(path)["runs"]:candidate.append(dict(family=names[r["family"]],name=r["name"],quantity=r["quantity"],**r["validation"]))
    path=root/"neural_kernel_route"/"rotor_magnitude"/"summary.json";sources.append(path)
    for r in read_json(path)["rows"]:candidate.append(dict(family="Learned NN coordinate + kernel / |B|",name=r["name"],quantity="|B|",**r["validation"]))
    add("candidates","neural_validation_experiments",candidate,{},"Training42 / validation9 only |40 completed condition-MLP experiments plus8 learned neural-coordinate kernel experiments. Vector and magnitude errors are distinct. Multiple uses of one validation set can induce selection bias; this plot is not independent test evidence.")
    comparison=[]
    for mid in ("prediction_rotor_magnitude","prediction_rotor_vector","prediction_stator_magnitude","prediction_stator_vector","prediction_airgap_Br"):
        a=datasets[mid];region=a["choice"]["region"];original=root.parent/"new60_20261014"/"experiments"/region/"final_evaluation"/"selected_predictions.npz"
        path=out/"provenance_inputs"/"v1"/(region+"_selected_predictions.npz");path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():shutil.copyfile(original,path)
        if digest(path)!=digest(original):raise ValueError("V1 provenance copy mismatch")
        sources.append(path)
        with np.load(path,allow_pickle=False) as z:old={k:z[k] for k in ("truth","prediction","case_ids","point_id")}
        order=[old["case_ids"].tolist().index(cid) for cid in a["case_ids"].tolist()];yy=old["truth"][order].astype(float);pp=old["prediction"][order].astype(float)
        if not np.array_equal(old["point_id"],a["point_id"]):raise ValueError("V1/V2 point mismatch")
        if a["quantity"]=="planar_magnitude":yy=np.hypot(yy[...,0::2],yy[...,1::2]);pp=np.hypot(pp[...,0::2],pp[...,1::2])
        yy=yy.reshape(a["truth"].shape);pp=pp.reshape(yy.shape)
        if not np.allclose(yy,a["truth"],rtol=1e-7,atol=1e-7):raise ValueError("V1/V2 truth mismatch")
        before=metric(yy,pp);after=metric(a["truth"],a["prediction"])
        comparison.append(dict(model_id=mid,region=region,quantity="Br,Bt" if a["quantity"]=="Br,Bt" else a["components"][0],**{"v1_"+k:v for k,v in before.items()},**{"v2_"+k:v for k,v in after.items()}))
    add("v1v2","v1_v2_identical_quantity_reused_test",comparison,{},"V1 and V2 evaluated on the same original nine test IDs and identical points. These labels were exposed in V1; this is reused-test comparison, not a fresh blind test. All V2 choices were frozen before re-evaluation; models were not reselected using this plot. Magnitude is recomputed from V1 Br/Bt, not copied from its vector metric.")
    a=datasets["fitting_stator"];path=root/"neural_route"/"fitting_full60"/"stator"/"fitting_predictions.npz";sources.append(path)
    with np.load(path,allow_pickle=False) as z:raw=z["raw_prediction"];ids=z["case_ids"]
    if not np.array_equal(ids,a["case_ids"]) or not np.allclose(np.maximum(raw,0),a["prediction"][...,0],rtol=1e-12,atol=1e-12):raise ValueError("Raw/projected fitting mismatch")
    neg=np.argwhere(raw<0);projection=[]
    for c,t,j in neg:projection.append(dict(row_type="negative_sample",case_id=int(ids[c]),frame_index=int(t),point_id=int(a["point_id"][j]),x_mm=float(a["xy"][j,0]),y_mm=float(a["xy"][j,1]),raw_T=float(raw[c,t,j]),truth_T=float(a["truth"][c,t,j,0]),projected_T=0))
    for version,pred in (("Raw NN",raw),("max(0,raw)",a["prediction"][...,0])):projection.append(dict(row_type="global",version=version,**metric(a["truth"][...,0],pred)))
    add("projection","stator_nonnegative_projection",projection,{},f"All60 fitting only | {len(neg)} negative case-frame-point observations, not necessarily distinct spatial points. Projection is an explicit deployment rule. Raw training/checkpoint loss used unprojected outputs; both raw and projected global metrics are retained.")
    write_csv(data/"all_primary_global_metrics.csv",globals_);write_csv(data/"v1_v2_same_quantity_metrics.csv",comparison);write_json(out/"worst_locations.json",worsts)
    write_json(out/"v1_v2_same_quantity_comparison.json",dict(scope="same nine previously exposed test cases, not fresh blind",rows=comparison,choices_sha256=freeze_sha))
    shutil.copyfile(__file__,out/"build_figures_v2.py")
    document=dict(version="V2_FIGURES_1",choices_sha256=freeze_sha,scientific_inputs=[dict(path=Path(os.path.relpath(p,out)).as_posix(),sha256=digest(p)) for p in dict.fromkeys(sources)],
        figures=recipe,reproduction="python build_figures_v2.py --replay-recipe figure_recipe.json --output regenerated",plotting_source_sha256=digest(__file__),
        geometry="actual sampled points only, original45-degree sector, rotor rotated by absolute mechanical theta; no inferred CAD surface or mirroring",formats=["png300dpi","pdf","svg"],
        metric_note="float64 source NPZ; raw vs nonnegative explicit; scientific metrics independent of website quantization")
    write_json(out/"figure_recipe.json",document);make_report(out,globals_,comparison,choices,neg)
    print(json.dumps(dict(figure_groups=len(recipe),formats=3,output=str(out),comparison=comparison),ensure_ascii=False))


def make_report(out,glob,comparison,choices,negative):
    lines=["# 新60工况 V2：拟合版与预测版实验材料", "", "两版回答不同问题。拟合版使用全部60个已知工况，用神经网络复现这些工况在60个既有转子位置的磁密云图；预测版仅以42个工况训练、9个验证工况选型，在冻结之后复评其余9个旧测试工况。旧测试标签在V1已被查看，因此下文称“复用测试”，不称新的独立盲测。", "",
        "## 数据和输出", "", "原始简化电机数据为60工况×60帧×4500测点；定子与转子各1800点、每点有Br/Bt，气隙900点、输出带符号Br。电流输入是RMS值与原生Thet角，数学sin/cos特征不等于已校准dq电流角。速度5500rpm、4极对；保存的绝对机械角为9°至97.5°、间隔1.5°，属于初始电周期数据，不证明稳态。", "",
        "磁密大小定义为sqrt(Br²+Bt²)，只有平面内分量，不能认证为含Bz的完整三维磁密大小。Br/Bt向量误差、从向量预测取模后的误差、直接磁密大小模型误差分别记录。采样点等权统计，没有有限元面积权重。图件保留真实45°采样扇区；转子世界坐标按绝对机械角旋转，3D图的高度表示磁密或误差，不是三维实体体积磁场。", "",
        "## 实际方法与选型", "", "预测优化完成40个独立条件MLP实验，并增加8个可学习神经坐标与核插值组合实验；还比较了条件核/RBF、多项式、整周期/分帧表示及非线性输出路线。神经网络使用PyTorch Adam/LBFGS与原T场MSE；所有压缩基、均值、标准化由训练工况拟合，检查点及超参数按验证集确定。40个NN包括有记录的协议缩减，未完成候选没有虚构结果。反复探索同一验证集会产生选择偏差，不能把验证最优当独立证据。", "",
        "主磁密云图预测模型采用条件核POD的Br/Bt输出再取平面内范数，不能把它称为纯神经网络或直接范数损失训练；纯神经网络Br/Bt与直接范数控制模型都保留。转子纯NN验证优化未达到小于1%的目标，继续降低训练误差会出现验证误差上升。降维的真值投影下限只是表示能力诊断，不能当网络预测精度。", "",
        "拟合版则完全独立地使用全部60工况，采用整周期POD rank59和256×256×256的tanh MLP，每区域148795个可训练参数。以未经非负投影的训练场误差≤0.05%停止；本次Adam已达到阈值，没有实际执行LBFGS。全部60标签包含原验证/测试工况，拟合结果没有回流到预测模型选择。", "",
        "## 原精度结果", "", "| 分支 | 区域 | 量 | RMSE / T | MAE / T | 最大绝对误差 / T | relative L2 |", "|---|---|---|---:|---:|---:|---:|"]
    for r in glob:lines.append(f"| {'复用测试9工况' if r['branch']=='prediction' else '全60训练内拟合'} | {r['region']} | {r['quantity'].replace('|B|','二维范数')} | {r['rmse_T']:.8g} | {r['mae_T']:.8g} | {r['max_abs_T']:.8g} | {r['relative_L2_pct']:.7f}% |")
    lines += ["",f"定子拟合原始输出存在{len(negative)}个负值观测（工况×帧×点），部署时显式采用max(0,raw)。原始relative L2为0.043283481760453324%，投影后为0.0432827595204%；训练停止与选型均依据原始输出。图18保留负值位置和两套指标，不能将投影后的细微下降包装成网络训练提升。", "",
        "最大绝对误差仍需关注：整体相对误差低不代表每个磁场位置均准确。worst_locations.json与各模型case/frame/point CSV保留了最差工况、最差帧、最差单点和四项指标；代表图按已冻结结果选取最差与中位误差帧，仅用于展示，没有据此重选模型。", "",
        "## V1到V2：同量、同9工况、同测点", "", "| 区域与量 | V1 relative L2 | V2 relative L2 | V1最大绝对误差 / T | V2最大绝对误差 / T |", "|---|---:|---:|---:|---:|"]
    for r in comparison:lines.append(f"| {r['region']} {r['quantity'].replace('|B|','二维范数')} | {r['v1_relative_L2_pct']:.6f}% | {r['v2_relative_L2_pct']:.6f}% | {r['v1_max_abs_T']:.7g} | {r['v2_max_abs_T']:.7g} |")
    lines += ["", "V1磁密大小指标由旧保存Br/Bt预测和同一真值逐点取范数后重新计算，不把旧向量误差当范数误差。平均指标改善与局部最大误差可能方向不同，表中均保留，不隐藏负结果。所有V2模型在复用测试前已冻结，复评后不重新挑选赢家。", "",
        "## 复现与使用边界", "", "图件为18组独立内容，每组含300dpi PNG、可编辑PDF/SVG及绘图CSV。figure_recipe.json对每个绘图CSV记录相对路径和SHA，并记录原始科学NPZ/QA输入来源SHA；携带本目录的build_figures_v2.py、recipe及data即可重绘，不必调用训练模型。可用`python build_figures_v2.py --replay-recipe figure_recipe.json --output regenerated`，加`--only 09_prediction_rotor_magnitude_3d`可只重绘代表图。依赖NumPy和Matplotlib。", "",
        "模型推理入口和最终CLI由交付说明统一列出；预测模型保留对应核/纯NN入口，拟合模型为neural_route/fitting_full60.py::predict，输入原生[I_RMS,Thet]。目前仅支持固定60帧与固定采样点，未验证新转速、任意新转子角、新网格位置或真实外部Maxwell盲测。网页压缩/量化数据仅供交互展示，本文指标与图源均使用原精度数组。", "",
        "## 图件索引", ""]
    for f in read_json(out/"figure_recipe.json")["figures"]:lines.append(f"- [{f['name']}]({f['name']}.png) · [PDF]({f['name']}.pdf) · [SVG]({f['name']}.svg)")
    lines += ["", "## 冻结时序和实验计数补充", "", "神经网络预测路线先冻结，随后full60分支独立训练且不反馈预测选型；最终global/combined选择在旧测试复评前冻结。full60训练元数据绑定的是neural_route/prediction_choice_frozen.json的SHA，不是更晚的全局冻结文件。", "",
        "独立清点记录为40个条件MLP预测实验、8个神经坐标核组合实验，另有3个全60拟合MLP，共51个实际NN训练实验。解析核路线的110538个配置及552690个rank排名行、1728个power配置、32个adaptive配置均属于其他搜索记录，不能算作NN训练次数。计数依据qa/optimization_inventory.json；失败/中断记录未冒充完成训练。"]
    (out/"optim报告材料.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def replay(recipe_path,out,only=None):
    path=Path(recipe_path).resolve();d=read_json(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for spec in d["figures"]:
        if only and spec["name"]!=only:continue
        source=path.parent/spec["input_csv"]
        if digest(source)!=spec["input_sha256"]:raise ValueError("CSV SHA mismatch")
        draw(spec,read_csv(source),out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,default=Path(__file__).parent);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--replay-recipe",type=Path);parser.add_argument("--only");args=parser.parse_args()
    if args.replay_recipe:replay(args.replay_recipe,args.output,args.only)
    else:build(args.root,args.output)
