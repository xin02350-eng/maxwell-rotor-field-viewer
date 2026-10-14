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
                if "highlight_point_id" in p:
                    marked=[ii for ii,rr in enumerate(data) if int(rr["point_id"])==int(p["highlight_point_id"])]
                    for ii in marked:
                        ax.scatter([x[ii]],[y[ii]],s=95,facecolors="none",edgecolors="#00B8D9",linewidths=1.5,zorder=5)
                        ax.annotate("point "+str(p["highlight_point_id"]),(x[ii],y[ii]),xytext=(8,7),textcoords="offset points",fontsize=7,color="#006A80")
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

