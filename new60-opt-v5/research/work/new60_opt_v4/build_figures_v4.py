"""Frozen V4 original-precision science figures; CSV-only replay needs no labels/model."""
from pathlib import Path
import argparse, csv, json, os, shutil
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from v2_plot_primitives import read_json,write_json,digest,write_csv,read_csv,nums,metric,save,COLORS,draw as primitive

FREEZE='8779511c7d36061b1266ca949e37bd721eb87c9f47703c25d08f3cd4673feba3'
KEYS=('rmse_T','mae_T','max_abs_T','relative_L2_pct')
SCOPE='Original42: 42 training + 9 validation for selection. Final51: original validation is training; no independent validation. Same nine previously seen test cases reused, not a fresh blind test.'
GEOMETRY='Only the actual 45-degree sampled sector, without mirroring, hole filling, or inferred material boundaries. World xy rotates reference points by absolute mechanical angle; xy in mm.'

def draw(s,rows,out):
    p=s['parameters'];kind=s['kind']
    if kind in ('cloud','height'):return primitive(s,rows,out)
    if kind in ('comparison','families'):
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.9));fig.subplots_adjust(bottom=.27,top=.82,wspace=.3)
        for ax,key,label in zip(axes,('relative_L2_pct','max_abs_T'),('Magnitude relative L2 (%)','Maximum magnitude error (T)')):
            v=nums(rows,key);x=np.arange(len(rows));ax.bar(x,v,color=[COLORS[i%6] for i in range(len(rows))])
            ax.set_xticks(x,[r['label'].replace(' / ','\n') for r in rows]);ax.set_ylabel(label);ax.set_ylim(0,max(v)*1.22);ax.grid(axis='y',alpha=.16)
            for xx,z in zip(x,v):ax.text(xx,z+max(v)*.025,f'{z:.5f}',ha='center',fontsize=8)
            if key=='max_abs_T':ax.axhline(1,color='#777',ls=':',lw=1)
        fig.suptitle(p['title'],fontsize=12,fontweight='bold')
    elif kind=='coverage':
        fig,axes=plt.subplots(1,3,figsize=(13,4.9));fig.subplots_adjust(bottom=.26,top=.82,wspace=.34)
        obs=[r for r in rows if r['type']=='input'];t=[r for r in obs if r['role']=='train42'];v=[r for r in obs if r['role']=='added9'];test=[r for r in obs if r['role']=='reused_test']
        for data,label,marker,color in ((t,'original train42','o','#777777'),(v,'added validation9','s',COLORS[0]),(test,'reused test9','*',COLORS[1])):
            axes[0].scatter(nums(data,'xI_A'),nums(data,'yI_A'),s=28 if marker!='*' else 70,marker=marker,color=color,label=label)
        for q in test:axes[0].annotate(str(q['case_id']),(float(q['xI_A']),float(q['yI_A'])),xytext=(4,3),textcoords='offset points',fontsize=7)
        for h,color in (('hull42','#777777'),('hull51',COLORS[0])):
            hrows=[r for r in rows if r['type']==h];axes[0].plot(nums(hrows,'xI_A'),nums(hrows,'yI_A'),'--',color=color,lw=1)
        axes[0].set(xlabel='I RMS cos(Thet) (A)',ylabel='I RMS sin(Thet) (A)',title='Input coverage and convex hull');axes[0].legend(fontsize=6.5,loc='lower right')
        for i,label in enumerate(('42 mixed','51 mixed')):
            rr=[r for r in rows if r['type']=='case' and r['label']==label];axes[1].plot(np.arange(len(rr)),nums(rr,'relative_L2_pct'),'o-',color=COLORS[i],label=label,ms=4)
            axes[2].scatter(nums(rr,'nearest_train_distance_A'),nums(rr,'relative_L2_pct'),color=COLORS[i],label=label,s=35)
            for q in rr:axes[2].annotate(str(q['case_id']),(float(q['nearest_train_distance_A']),float(q['relative_L2_pct'])),xytext=(3,5 if i==0 else -10),textcoords='offset points',fontsize=7,color=COLORS[i])
        axes[1].set(xlabel='Original reused-test case ID',ylabel='Case magnitude relative L2 (%)',title='All nine test cases');axes[1].set_xticks(np.arange(len(test)),[int(q['case_id']) for q in test]);axes[1].legend(fontsize=8)
        axes[2].set(xlabel='Nearest training input distance (A)',ylabel='Case magnitude relative L2 (%)',title='Descriptive association; not causal');axes[2].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.13)
        fig.suptitle(p['title'],fontsize=12,fontweight='bold')
    elif kind=='risk':
        fig,axes=plt.subplots(1,2,figsize=(10.6,5.1));fig.subplots_adjust(bottom=.25,top=.84,wspace=.35)
        geo=[r for r in rows if r['type']=='geometry'];hot=[r for r in rows if r['type']=='risk'];train=[r for r in rows if r['type']=='train'];req=[r for r in rows if r['type']=='request']
        axes[0].scatter(nums(geo,'x_ref_mm'),nums(geo,'y_ref_mm'),s=4,color='#d1d5db')
        art=axes[0].scatter(nums(hot,'x_ref_mm'),nums(hot,'y_ref_mm'),c=nums(hot,'loo_mean_rms_T'),s=20,cmap='magma',vmin=0)
        axes[0].set_aspect('equal');axes[0].set(xlabel='Rotor reference x (mm)',ylabel='Rotor reference y (mm)',title='90 training-LOO risk points; all 1800 shown')
        fig.colorbar(art,ax=axes[0],fraction=.04,pad=.03,label='RMS error of time-mean magnitude (T)')
        axes[1].scatter(nums(train,'xI_A'),nums(train,'yI_A'),s=25,color='#7a8490',label='42 observed training inputs')
        axes[1].scatter(nums(req,'xI_A'),nums(req,'yI_A'),s=50,facecolors='none',edgecolors=COLORS[1],marker='^',label='24 requested simulations; no labels')
        axes[1].set(xlabel='I RMS cos(Thet) (A)',ylabel='I RMS sin(Thet) (A)',title='Training-only acquisition proposals');axes[1].legend(fontsize=8);axes[1].grid(alpha=.15)
        fig.suptitle(p['title'],fontsize=12,fontweight='bold')
    else:raise ValueError(kind)
    save(fig,out/s['name'],s['caption'])

def load_science(path,choice,expected,proof):
    if digest(path)!=proof['scientific_npz_sha256']:raise ValueError('Science file SHA mismatch')
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    m=json.loads(str(a['metadata_json']))
    assert m['choices_sha256']==expected and m['model_sha256']==choice['model_sha256']
    assert m['model_id']==choice.get('scientific_model_id',choice['id']) and m['partition']==proof['partition']
    y,p=a['truth'],a['prediction'];assert y.dtype==p.dtype==np.float64 and y.shape==p.shape and y.shape[-1]==len(a['point_id'])
    assert np.isfinite(y).all() and np.isfinite(p).all()
    g=metric(y,p)
    for k in KEYS:assert np.isclose(g[k],proof['metrics'][k],rtol=2e-12,atol=1e-15),(choice['id'],k,g[k],proof['metrics'][k])
    a.update(choice=choice,metadata=m,global_metrics=g);return a

def tables(a,data):
    y,p=a['truth'],a['prediction'];ch=a['choice'];prefix=dict(model_id=ch['id'],training_scope=ch['training_scope'],partition=a['metadata']['partition'],quantity=ch['quantity'])
    cases=[dict(prefix,case_id=int(cid),**metric(y[c],p[c])) for c,cid in enumerate(a['case_ids'])]
    frames=[dict(prefix,case_id=int(cid),frame_index=t,theta_m_absolute_deg=float(a['theta'][t]),**metric(y[c,t],p[c,t])) for c,cid in enumerate(a['case_ids']) for t in range(y.shape[1])]
    points=[dict(prefix,point_id=int(pid),x_ref_mm=float(a['xy'][j,0]),y_ref_mm=float(a['xy'][j,1]),**metric(y[:,:,j],p[:,:,j])) for j,pid in enumerate(a['point_id'])]
    for name,rr in (('case',cases),('frame',frames),('point',points)):write_csv(data/(ch['id']+'_'+name+'_metrics.csv'),rr)
    c,t,j=np.unravel_index(np.argmax(np.abs(p-y)),y.shape);theta=np.deg2rad(float(a['theta'][t]));x=a['xy'][j]@np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]) if ch['region']=='rotor' else a['xy'][j]
    worst=dict(prefix,case_id=int(a['case_ids'][c]),frame_index=int(t),theta_m_absolute_deg=float(a['theta'][t]),point_id=int(a['point_id'][j]),truth_T=float(y[c,t,j]),prediction_T=float(p[c,t,j]),absolute_error_T=float(abs(p[c,t,j]-y[c,t,j])),x_world_mm=float(x[0]),y_world_mm=float(x[1]))
    return cases,worst

def frame_rows(a,c,t,label):
    angle=np.deg2rad(float(a['theta'][t]));xy=a['xy']@np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    return [dict(representative=label,component='|B|',model_id=a['choice']['id'],training_scope=a['choice']['training_scope'],partition=a['metadata']['partition'],case_id=int(a['case_ids'][c]),frame_index=t,point_id=int(pid),theta_m_absolute_deg=float(a['theta'][t]),x_ref_mm=float(a['xy'][j,0]),y_ref_mm=float(a['xy'][j,1]),x_world_mm=float(xy[j,0]),y_world_mm=float(xy[j,1]),truth_T=float(a['truth'][c,t,j]),prediction_T=float(a['prediction'][c,t,j]),absolute_error_T=float(abs(a['prediction'][c,t,j]-a['truth'][c,t,j]))) for j,pid in enumerate(a['point_id'])]

def current_xy(conditions):
    c=np.asarray(conditions,dtype=float);ang=np.deg2rad(c[:,1]);return c[:,0,None]*np.stack([np.cos(ang),np.sin(ang)],axis=1)

def build(root,out):
    root=root.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True);data=out/'data';data.mkdir(exist_ok=True)
    freeze=root/'both_versions_choices_frozen.json';assert digest(freeze)==FREEZE
    qa_path=root/'qa/scientific_qa.json';qa=read_json(qa_path);assert qa['all_checks_pass'] is True and qa['choices_sha256']==FREEZE
    registry=read_json(freeze);choices={m['id']:m for m in registry['models']};sources=[freeze,qa_path];sets={};case_tables={};worsts=[];globals_=[];bindings=[]
    for mid,ch in choices.items():
        part='training_fit' if ch['branch']=='fitting_full60' else 'reused_test';q=next(r for r in qa['results'] if r['id']==mid and r['partition']==part)
        sr=root.parent/q['scientific_source_relative_root'];path=sr/q['scientific_npz_relative_path'];expected=q['expected_scientific_choices_sha256']
        assert digest(sr/'both_versions_choices_frozen.json')==expected
        a=load_science(path,ch,expected,q);sources.extend([path,sr/'both_versions_choices_frozen.json']);case_tables[mid],worst=tables(a,data);worsts.append(worst);globals_.append(dict(model_id=mid,region=ch['region'],training_scope=ch['training_scope'],partition=part,quantity=ch['quantity'],**a['global_metrics']))
        bindings.append(dict(model_id=mid,partition=part,scientific_source_root=q['scientific_source_relative_root'],scientific_sha256=digest(path),scientific_choices_sha256=expected,current_choices_sha256=FREEZE,model_sha256=ch['model_sha256']))
        if ch['region']=='rotor':sets[mid]=a
    v3=root.parent/'new60_opt_v3';oldqa=read_json(v3/'qa/scientific_qa.json');assert oldqa['all_checks_pass'];sources.append(v3/'qa/scientific_qa.json');oldchoices={m['id']:m for m in read_json(v3/'both_versions_choices_frozen.json')['models']}
    old=[]
    for mid,label in (('prediction_rotor_v2_kernel','V2 / kernel42'),('prediction_rotor_magnitude','V3 / mixed42')):
        q=next(r for r in oldqa['results'] if r['id']==mid and r['partition']=='reused_test');path=v3/'evaluation'/mid/'reused_test_predictions.npz'
        # V3 reports bind science files through a different key in older runs.
        if 'scientific_npz_sha256' not in q:q=dict(q,scientific_npz_sha256=digest(path))
        a=load_science(path,oldchoices[mid],oldqa['choices_sha256'],q);sources.append(path);old.append(dict(label=label,**a['global_metrics']))
    p42=sets['prediction_rotor_magnitude'];p51=sets['prediction_rotor_final51_magnitude'];fit=sets['fitting_rotor_magnitude'];recipe=[]
    for a in sets.values():
        order=[a['case_ids'].tolist().index(int(i)) for i in p42['case_ids']]
        assert np.array_equal(a['point_id'],p42['point_id']) and np.array_equal(a['truth'][order],p42['truth'])
    def add(kind,name,rows,parameters,caption):
        cp=data/(name+'.csv');write_csv(cp,rows);spec=dict(name=name,kind=kind,input_csv=cp.relative_to(out).as_posix(),input_sha256=digest(cp),parameters=parameters,caption=caption);recipe.append(spec);draw(spec,rows,out)
    def cloud(name,rows,panels,field,error,title,caption,highlight=None):
        params=dict(region='rotor',panels=panels,scales={'|B|':dict(field=field,error=error)},title=title)
        if highlight is not None:params['highlight_point_id']=highlight
        add('cloud',name,rows,params,caption)
    def panel(label,text):return dict(representative=label,component='|B|',label=text)
    f2=fit['case_ids'].tolist().index(2);t=0;fitrows=frame_rows(fit,f2,t,'fit');c2=p42['case_ids'].tolist().index(2);r42=frame_rows(p42,c2,t,'original42');r51=frame_rows(p51,c2,t,'final51')
    field=[0,max(float(a['truth'].max()) for a in (fit,p42,p51))];field[1]=max(field[1],float(p42['prediction'].max()),float(p51['prediction'].max()),float(fit['prediction'].max()))
    fiterror=[0,max(r['absolute_error_T'] for r in fitrows)];err=[0,max(float(np.max(np.abs(a['prediction']-a['truth']))) for a in (p42,p51))]
    fitcaption='All60 training fit, not prediction: trained nonlinear hidden layers with a calibrated least-squares output head. Case02 was trained. Frame-local residual scale shows numerical roundoff of known-data reproduction, not physical Maxwell accuracy. '+GEOMETRY
    cloud('01_known60_fitting_2d',fitrows,[panel('fit','case02 | frame0 | absolute theta=9 deg\nALL60 TRAINING FIT')],field,fiterror,'Known60 fit | planar magnitude and original floating-point residual',fitcaption)
    add('height','02_known60_fitting_height_3d',fitrows,dict(representative='fit',field=field,error=fiterror,title='Known60 training fit | case02, frame0, absolute theta=9 deg'),fitcaption+' Height encodes magnitude T or residual mT; this is not a volumetric magnetic field.')
    cloud('03_fixed_case42_51_fit_comparison',r42+r51+fitrows,[panel('original42','ORIGINAL42 MIXED\ncase02 reused test'),panel('final51','FINAL51 MIXED\ncase02 reused test'),panel('fit','ALL60 FIT\ncase02 trained')],field,err,'Same case02, frame0, absolute theta9 deg | different evaluation roles',SCOPE+' All panels show planar magnitude with common truth/prediction and error scales. Fit residual is separately resolved in figure01. '+GEOMETRY)
    comparison=old+[dict(label=label,**a['global_metrics']) for a,label in ((p42,'V4 / mixed42'),(p51,'V4 / mixed51'))]
    add('comparison','04_same_quantity_v2_v3_v4',comparison,dict(title='Rotor planar magnitude | same nine reused-test cases'),SCOPE+' Every method uses sqrt(Br^2+Bt^2); signed vector L2 is not substituted. Both the average and the largest pointwise error are retained.')
    family=[dict(label=label,**sets[mid]['global_metrics']) for mid,label in (('prediction_rotor_pure_nn','42 / pure NN'),('prediction_rotor_magnitude','42 / mixed'),('prediction_rotor_final51_pure_nn','51 / pure NN'),('prediction_rotor_final51_magnitude','51 / mixed'))]
    add('families','05_pure_neural_and_mixed',family,dict(title='Pure neural and fixed mixed models | reused test only'),SCOPE+' Mixed models include a kernel contribution and are not pure neural networks. No model was reselected during figure generation.')
    w=next(q for q in worsts if q['model_id']=='prediction_rotor_final51_magnitude');ci=p51['case_ids'].tolist().index(w['case_id']);wr=frame_rows(p51,ci,w['frame_index'],'worst')
    cloud('06_final51_maximum_error_full_field',wr,[panel('worst',f"case{w['case_id']:02d} | frame{w['frame_index']}\nabsolute theta={w['theta_m_absolute_deg']:g} deg")],field,err,f"Final51 mixed | maximum point error {w['absolute_error_T']:.6f} T",SCOPE+f" Point ID {w['point_id']} is ringed, without cropping or removing difficult points. This is the frame containing the largest point error, not necessarily largest frame RMSE. "+GEOMETRY,highlight=w['point_id'])
    # Input coverage uses conditions alone. Final51 hull includes former validation inputs.
    condmap={int(cid):cond for cid,cond in zip(fit['case_ids'],fit['conditions'])};t42=choices['prediction_rotor_magnitude']['training_case_ids'];t51=choices['prediction_rotor_final51_magnitude']['training_case_ids'];test=list(map(int,p42['case_ids']));coverage=[]
    for cid in t51+test:
        x=current_xy([condmap[cid]])[0];coverage.append(dict(type='input',case_id=cid,role='train42' if cid in t42 else 'added9' if cid in t51 else 'reused_test',xI_A=x[0],yI_A=x[1]))
    for ids,label,a in ((t42,'42 mixed',p42),(t51,'51 mixed',p51)):
        tx=current_xy([condmap[i] for i in ids]);h=ConvexHull(tx);vertices=list(h.vertices)+[int(h.vertices[0])]
        for vi in vertices:coverage.append(dict(type='hull42' if len(ids)==42 else 'hull51',xI_A=float(tx[vi,0]),yI_A=float(tx[vi,1])))
        for row in case_tables[a['choice']['id']]:
            x=current_xy([condmap[row['case_id']]])[0];dd=np.sqrt(np.sum((tx-x)**2,axis=1));inside=bool(np.all(h.equations[:,:2]@x+h.equations[:,2]<=1e-8));coverage.append(dict(row,type='case',label=label,nearest_train_distance_A=float(dd.min()),nearest_train_case_id=int(ids[int(dd.argmin())]),inside_training_convex_hull=inside))
    add('coverage','07_case_errors_and_input_coverage',coverage,dict(title='Sampling coverage and all-case error | descriptive evidence'),SCOPE+' Distances use native I_RMS cos(Thet), I_RMS sin(Thet), a mathematical encoding, not verified physical Id/Iq. Hull membership and nearest distance do not guarantee accuracy or establish a causal explanation.')
    riskpath=root/'diagnostics_train_only/train_hotspot_response.csv';requestpath=v3/'additional_simulation_requests/建议新增工况.csv';auditpath=root/'diagnostics_train_only/train_only_source_audit.json';sources.extend([riskpath,requestpath,auditpath]);audit=read_json(auditpath);assert not audit['test_and_validation_labels_loaded'] and not audit['full60_fitting_labels_or_predictions_loaded']
    hr=read_csv(riskpath);risk=[dict(type='geometry',point_id=int(pid),x_ref_mm=float(x[0]),y_ref_mm=float(x[1])) for pid,x in zip(p42['point_id'],p42['xy'])]
    for pid in sorted(set(int(r['point_id']) for r in hr)):
        rr=[r for r in hr if int(r['point_id'])==pid];assert len(rr)==42;risk.append(dict(type='risk',point_id=pid,x_ref_mm=float(rr[0]['x_ref_mm']),y_ref_mm=float(rr[0]['y_ref_mm']),loo_mean_rms_T=float(np.sqrt(np.mean(nums(rr,'loo_unsigned_mean_residual_T')**2)))))
    for cid in t42:
        x=current_xy([condmap[cid]])[0];risk.append(dict(type='train',case_id=cid,xI_A=float(x[0]),yI_A=float(x[1])))
    for r in read_csv(requestpath):
        x=current_xy([[r['I_RMS_A'],r['Thet_deg']]])[0];risk.append(dict(type='request',request_id=r['request_id'],xI_A=float(x[0]),yI_A=float(x[1]),I_RMS_A=float(r['I_RMS_A']),Thet_deg=float(r['Thet_deg'])))
    add('risk','08_training_risk_and_new_simulation_requests',risk,dict(title='Training-only spatial risk and proposals for real Maxwell simulations'),'The 90-point set and its time-mean-magnitude LOO residual come only from original42 training conditions. LOO was also used to select shared kernel parameters, so this is risk diagnosis, not an unbiased validation score. No point is removed. The 24 proposed inputs have no simulated field labels; no mesh-noise cause is established.')
    write_json(out/'same_quantity_comparison.json',dict(quantity='planar_magnitude',partition='reused_test',comparison=comparison,pure_and_mixed=family,choices_sha256=FREEZE))
    write_json(out/'global_metrics.json',globals_);write_json(out/'worst_locations.json',worsts);write_json(out/'scientific_bindings.json',bindings)
    sources=list(dict.fromkeys(p.resolve() for p in sources));csvs=sorted(data.glob('*.csv'))
    for source in (Path(__file__),Path(__file__).with_name('v2_plot_primitives.py')):
        if source.resolve()!= (out/source.name).resolve():shutil.copy2(source,out/source.name)
    write_json(out/'figure_recipe.json',dict(version='V4_FROZEN_SCIENCE_FIGURES',choices_sha256=FREEZE,figures=recipe,scientific_inputs=[dict(path=Path(os.path.relpath(p,out)).as_posix(),sha256=digest(p),bytes=p.stat().st_size) for p in sources],csv_inventory=[dict(path=p.relative_to(out).as_posix(),sha256=digest(p),rows=len(read_csv(p)),bytes=p.stat().st_size) for p in csvs],plotting_sources=[dict(path=n,sha256=digest(out/n)) for n in ('build_figures_v4.py','v2_plot_primitives.py')],requirements=['numpy','scipy','matplotlib'],reproduction='python build_figures_v4.py --replay-recipe figure_recipe.json --output regenerated --only 02_known60_fitting_height_3d'))
    report(out,comparison,family,globals_,w,recipe)
    assert digest(freeze)==FREEZE
    print(json.dumps(dict(figures=len(recipe),csv=len(csvs),total_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),comparison=comparison,worst_final51=w),ensure_ascii=False),flush=True)

def report(out,comparison,family,global_,worst,recipe):
    lines=['# V4 科学图与误差口径','', '本图集不训练、不重选模型。原42工况训练版以原9验证工况选型；最终51工况版已将原9验证工况加入训练，没有独立验证集。两者评估均复用此前已见的原9测试工况，不能称新盲测。全60拟合版包含全部既有标签，误差只能称训练内拟合误差。','', '## 转子同物理量比较','', '所有数值均来自原精度科学NPZ，磁密大小为二维范数 sqrt(Br²+Bt²)，不是Br/Bt向量误差，不由网页q16数据计算。','', '| 模型 | 相对L2 / % | RMSE / T | MAE / T | 最大绝对误差 / T |','|---|---:|---:|---:|---:|']
    for r in comparison+family:lines.append(f"| {r['label']} | {r['relative_L2_pct']:.8f} | {r['rmse_T']:.8g} | {r['mae_T']:.8g} | {r['max_abs_T']:.8g} |")
    lines+=['','混合模型含核模型分支，不称纯神经网络。平均指标改善与局部最大误差分别报告。纯NN与混合模型、42与51训练范围没有混为同一实验；新增训练数据本身也会改变比较条件。',f"最终51混合模型最大局部误差 {worst['absolute_error_T']:.8g} T，位于 case{worst['case_id']:02d}、frame{worst['frame_index']}、绝对机械角{worst['theta_m_absolute_deg']:g}°、point ID {worst['point_id']}。06图展示该帧完整测量扇区并圈出该点，未裁切、删点或将其称为已证实的噪声。",'', '## 全60拟合精度','', '| 区域和量 | 训练内相对L2 / % | RMSE / T | 最大绝对误差 / T |','|---|---:|---:|---:|']
    for r in global_:
        if r['training_scope']=='full60':lines.append(f"| {r['region']} / {r['quantity']} | {r['relative_L2_pct']:.6e} | {r['rmse_T']:.6e} | {r['max_abs_T']:.6e} |")
    lines+=['', '全60沿用V3训练后的非线性隐藏层与最小二乘校准输出层。接近机器舍入精度的残差只说明有限既有数据的数值复现，不代表新工况预测、连续空间场或Maxwell物理真值同样准确。01和02图保留原始浮点残差，没有画成绝对零。','', '## 采样范围、诊断与追加仿真','', '云图仅画原45°扇区内的1800个真实转子测点，不镜像补全、不插值跨孔。世界坐标按绝对机械角旋转参考点。3D图高度为二维磁密范数，不是三维体积B场。case02、frame0在预测图是复用测试，在拟合图是训练内数据。','', '07图保留全部9个测试工况，按I_RMS cos(Thet)、I_RMS sin(Thet)数学坐标计算到42或51训练输入的最近距离及凸包归属。该坐标不未经核实称物理Id/Iq，距离只是覆盖描述，不能保证预测准确或断定误差原因。','', '08图的90点来自原42训练工况留一风险排序，颜色为时间平均磁密模的留一残差RMS；与此前V2验证集的94.798%静态偏差SSE统计不是同一统计量。留一也参与了共享参数选择，故此处不当无偏泛化成绩。24个新增工况仅为真实Maxwell仿真的输入建议，没有新增磁密标签，不用全60拟合输出冒充新样本。','', '## 图源与复现','', '8组图均提供300dpi PNG和可编辑PDF/SVG、绘图CSV；全部9个冻结模型另有逐工况、逐工况帧、逐点RMSE/MAE/最大绝对误差/相对L2 CSV。相对L2按相应分组真值能量归一；测点等权，不等同有限元面积加权。全局/极值/来源绑定分别见global_metrics.json、worst_locations.json、scientific_bindings.json。','', 'figure_recipe.json保存全部源文件相对路径、SHA256、CSV行数与字节数。旧模型科学数组按原路径引用，未重复复制大数据。单图复现只需两份绘图源码、recipe和该图CSV：`python build_figures_v4.py --replay-recipe figure_recipe.json --output regenerated --only 02_known60_fitting_height_3d`。','']
    for r in recipe:lines.append(f"- [{r['name']}]({r['name']}.png) / [PDF]({r['name']}.pdf) / [SVG]({r['name']}.svg)")
    (out/'V4科学说明.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).parent);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--replay-recipe',type=Path);ap.add_argument('--only');args=ap.parse_args()
    if args.replay_recipe:
        args.output.mkdir(parents=True,exist_ok=True)
        for s in read_json(args.replay_recipe)['figures']:
            if args.only and args.only!=s['name']:continue
            p=args.replay_recipe.parent/s['input_csv'];assert digest(p)==s['input_sha256'];draw(s,read_csv(p),args.output)
    else:build(args.root,args.output)
