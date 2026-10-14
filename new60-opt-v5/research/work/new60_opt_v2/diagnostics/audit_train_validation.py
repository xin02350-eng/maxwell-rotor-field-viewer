"""Train42/validation9 diagnostics; excluded test field rows are never decoded.

No V1 output is modified. Diagnostic regressors are analytic controls, not
claimed neural networks or a new final selection. CPU BLAS limited to two.
"""
from pathlib import Path
from datetime import datetime, timezone
import sys
import json
import hashlib
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
V1=HERE.parents[1]/'new60_20261014'
sys.path.insert(0,str(V1/'code'))
from stage2_local import load_selection_cache
import experiment_core as core
import hybrid_cycle_rbf as hybrid


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def metrics(pred,true,ids):
    true=np.asarray(true,dtype=np.float64);e=np.asarray(pred,dtype=np.float64)-true
    sse=np.sum(e.reshape(len(ids),-1)**2,axis=1);den=np.sum(true.reshape(len(ids),-1)**2,axis=1)
    n=true[0].size;rmse=np.sqrt(sse/n)
    return dict(rmse_T=float(np.sqrt(np.sum(sse)/(n*len(ids)))),relative_L2_pct=float(100*np.sqrt(np.sum(sse)/np.sum(den))),
        mae_T=float(np.mean(np.abs(e))),max_abs_component_error_T=float(np.max(np.abs(e))),
        worst_case_rmse_T=float(rmse.max()),worst_case_id=int(ids[np.argmax(rmse)]),
        per_case=[dict(case_id=int(i),rmse_T=float(r),relative_L2_pct=float(100*np.sqrt(s/d))) for i,r,s,d in zip(ids,rmse,sse,den)])

def from_sse(sse,energy,dimension,ids):
    sse=np.maximum(sse,0);rmse=np.sqrt(sse/dimension)
    return dict(rmse_T=float(np.sqrt(sse.sum()/(dimension*len(ids)))),relative_L2_pct=float(100*np.sqrt(sse.sum()/energy.sum())),
        worst_case_rmse_T=float(rmse.max()),worst_case_id=int(ids[np.argmax(rmse)]),
        per_case=[dict(case_id=int(i),rmse_T=float(r),relative_L2_pct=float(100*np.sqrt(s/d))) for i,r,s,d in zip(ids,rmse,sse,energy)])

def embedding(c,mode,scale=None):
    c=np.asarray(c,dtype=np.float64)
    if mode=='native':return c.copy()
    angle=np.deg2rad(c[:,1]);radius=c[:,0]
    if mode=='saturated_cartesian':radius=scale*np.tanh(radius/scale)
    return np.column_stack((radius*np.cos(angle),radius*np.sin(angle)))

def normalization(train,query):
    mean=train.mean(axis=0);scale=np.maximum(train.std(axis=0),1e-12)
    return (train-mean)/scale,(query-mean)/scale

def rbf_weights(train,query,length,ridge):
    a,b=normalization(train,query)
    k=np.exp(-.5*np.sum((a[:,None,:]-a[None,:,:])**2,axis=2)/length**2)
    q=np.exp(-.5*np.sum((b[:,None,:]-a[None,:,:])**2,axis=2)/length**2)
    # Unpenalized constant: the predictor can preserve a permanent-magnet field.
    n=len(a);block=np.empty((n+1,n+1));block[:n,:n]=k+ridge*np.eye(n)
    block[:n,n]=1;block[n,:n]=1;block[n,n]=0
    return np.linalg.solve(block,np.column_stack((q,np.ones(len(q)))).T).T[:,:n]

def polynomial_weights(train,query,degree):
    a,b=normalization(train,query)
    def design(x):return np.column_stack([x[:,0]**i*x[:,1]**j for total in range(degree+1) for i in range(total+1) for j in [total-i]])
    # Fixed ridge on all nonconstant terms; same setting for every diagnostic.
    a,b=design(a),design(b);pen=np.eye(a.shape[1])*1e-6;pen[0,0]=0
    return np.linalg.solve(a.T@a+pen,(b@a.T*0)[:,:a.shape[1]].T).T if False else b@np.linalg.solve(a.T@a+pen,a.T)

def evaluate_weights(weights,gram,cross,energy,dimension,ids):
    sse=np.einsum('ij,jk,ik->i',weights,gram,weights)-2*np.sum(weights*cross,axis=1)+energy
    return from_sse(sse,energy,dimension,ids)

def diagnostics(region,split):
    start=time.perf_counter();path=V1/'cache_v1'/f'{region}.npz'
    cache,_,meta=load_selection_cache(path,V1/'cache_v1/split.json','Br' if region=='airgap' else 'Br,Bt')
    tr=np.asarray(split['train']);va=np.asarray(split['validation']);te=np.asarray(split['test'])
    assert np.isnan(cache['field'][te]).all() and np.isnan(cache['coeff'][te]).all()
    y=cache['field'][tr].reshape(len(tr),-1).astype(np.float64)
    v=cache['field'][va].reshape(len(va),-1).astype(np.float64)
    c=cache['conditions'];d=y.shape[1];gram=y@y.T;cross=v@y.T;energy=np.sum(v*v,axis=1)
    mean=y.mean(axis=0);centered=y-mean;g=centered@centered.T
    ev,u=np.linalg.eigh(g);order=np.argsort(ev)[::-1];ev=np.maximum(ev[order],0);u=u[:,order]
    rank=int(np.count_nonzero(ev>ev[0]*1e-12));basis=(u[:,:rank].T@centered)/np.sqrt(ev[:rank,None])
    valcoeff=(v-mean)@basis.T
    pod=[]
    for r in sorted(set([1,2,4,8,12,16,24,32,rank])):
        if r>rank:continue
        p=mean+valcoeff[:,:r]@basis[:r]
        pod.append(dict(rank=r,training_centered_energy_fraction=float(ev[:r].sum()/ev.sum()),validation_projection=metrics(p,v,va)))
    spatial=cache['mean']+cache['coeff'][va].astype(np.float64)@cache['basis'].astype(np.float64).T
    spatial_floor=metrics(spatial,cache['field'][va],va)
    zero_indices=np.where(c[tr,0]==0)[0];zero=[]
    for row in zero_indices:
        base=y[row];other=np.argsort(c[tr,0])[:6]
        zero.append(dict(case_id=int(tr[row]),field_rms_T=float(np.sqrt(np.mean(base*base))),max_abs_component_T=float(np.max(np.abs(base))),
            near_zero_training_cases=[dict(case_id=int(tr[j]),I_rms_A=float(c[tr[j],0]),Thet_deg=float(c[tr[j],1]),
                difference_from_zero_field_rmse_T=float(np.sqrt(np.mean((y[j]-base)**2))),
                difference_per_current_T_per_A=float(np.sqrt(np.mean((y[j]-base)**2))/c[tr[j],0]) if c[tr[j],0]>0 else None) for j in other],
            replicated_zero_current_case_count=len(zero_indices),angle_invariance_identifiable=len(zero_indices)>1))
    # A modest, fixed analytic diagnostic grid. No test numerical values are read.
    controls=[];cv_start=time.perf_counter();fit_calls=0
    for mode in ('native','cartesian','saturated_cartesian'):
        radial_scale=float(np.median(c[tr,0][c[tr,0]>0]))
        train=embedding(c[tr],mode,radial_scale);validation=embedding(c[va],mode,radial_scale)
        recipes=[('polynomial',degree,None) for degree in (1,2,3)]+[('gaussian',length,ridge) for length in (.6,1.2,2.4,4.8) for ridge in (1e-8,1e-6,1e-4)]
        for kind,arg,ridge in recipes:
            weights=rbf_weights(train,validation,arg,ridge) if kind=='gaussian' else polynomial_weights(train,validation,arg)
            validation_result=evaluate_weights(weights,gram,cross,energy,d,va)
            loo_sse=[]
            for held in range(len(tr)):
                retained=np.delete(np.arange(len(tr)),held)
                # Even saturation scale and input normalization are fitted anew within each fold.
                fold_scale=float(np.median(c[tr[retained],0][c[tr[retained],0]>0]))
                a=embedding(c[tr[retained]],mode,fold_scale);b=embedding(c[tr[[held]]],mode,fold_scale)
                w=(rbf_weights(a,b,arg,ridge) if kind=='gaussian' else polynomial_weights(a,b,arg))[0]
                error=float(w@gram[np.ix_(retained,retained)]@w-2*w@gram[retained,held]+gram[held,held])
                loo_sse.append(max(0,error));fit_calls+=1
            loo=from_sse(np.asarray(loo_sse),np.diag(gram),d,tr)
            record=dict(input_embedding=mode,family=kind,lengthscale=arg if kind=='gaussian' else None,
                ridge=ridge if kind=='gaussian' else 1e-6,degree=arg if kind=='polynomial' else None,
                saturation_scale_A=radial_scale if mode=='saturated_cartesian' else None,validation=validation_result,
                training_LOOCV=loo,LOOCV_is_nested_selection_estimate=False)
            controls.append(record)
    cv_seconds=time.perf_counter()-cv_start
    # Replay existing V1 models on only visible partitions, not the final evaluation JSON.
    baselines={}
    for name,model in [('uniform_NN',V1/'experiments'/region/'stage2/uniform_field_control_seed42/best_model.npz'),
                       ('hybrid',V1/'experiments'/region/'hybrid_seed42/model.npz'),
                       ('independent_RBF',V1/'experiments'/region/'hybrid_seed42/rbf_only_model.npz')]:
        pred=core.predict(model,cache['features'][va]) if name=='uniform_NN' else hybrid.predict(model,c[va])
        baselines[name]=dict(model_sha256=digest(model),validation=metrics(pred,cache['field'][va],va))
    native=embedding(c[tr],'native');cart=embedding(c[tr],'cartesian')
    neighboring={}
    for name,train in [('native',native),('cartesian',cart)]:
        a,b=normalization(train,embedding(c[va],name));dist=np.sqrt(np.sum((b[:,None]-a[None])**2,axis=2))
        nearest=np.argmin(dist,axis=1)
        neighboring[name]=[dict(validation_case_id=int(va[i]),nearest_training_case_id=int(tr[j]),normalized_input_distance=float(dist[i,j]),
            field_difference_rmse_T=float(np.sqrt(np.mean((v[i]-y[j])**2)))) for i,j in enumerate(nearest)]
    visible=cache['field'][np.r_[tr,va]].astype(np.float64)
    # This measures signal roughness, not stochastic noise or a solver convergence error.
    roughness=[]
    for case,row in zip(np.r_[tr,va],visible):
        roughness.append(dict(case_id=int(case),split='train' if case in tr else 'validation',
            field_rms_T=float(np.sqrt(np.mean(row*row))),first_frame_last_frame_difference_rmse_T=float(np.sqrt(np.mean((row[-1]-row[0])**2))),
            consecutive_frame_difference_rmse_T=float(np.sqrt(np.mean(np.diff(row,axis=0)**2))),
            second_frame_difference_rmse_T=float(np.sqrt(np.mean(np.diff(row,n=2,axis=0)**2)))))
    best_val=min(controls,key=lambda x:x['validation']['rmse_T']);best_cv=min(controls,key=lambda x:x['training_LOOCV']['rmse_T'])
    out=dict(region=region,quantity=meta['quantity'],units='T',xy_units='mm',field_shape=list(cache['field'].shape),
        cache_sha256=digest(path),source_loader_sha256=digest(V1/'code/stage2_local.py'),test_rows_materialized=False,
        visible_case_ids=dict(train=tr.tolist(),validation=va.tolist()),whole_cycle_condition_POD=dict(training_samples=42,output_dimension=d,
            rank=rank,eigenvalues_T_squared=ev.tolist(),rank_diagnostics=pod,mean_field_energy_fraction=float(42*np.sum(mean*mean)/np.sum(y*y)),
            projection_scope='Best orthogonal projection of validation truth on train42 full-cycle affine subspace; oracle diagnostic, not predicted performance'),
        V1_spatial_POD_validation_floor=spatial_floor,zero_current=zero,existing_V1_visible_baselines=baselines,
        diagnostic_controls=controls,diagnostic_lowest_validation=best_val,diagnostic_lowest_training_LOOCV=best_cv,
        nearest_visible_conditions=neighboring,temporal_signal_roughness_not_noise=roughness,
        analytic_cv_cost=dict(fit_calls=fit_calls,seconds=cv_seconds,seconds_per_small_fit=cv_seconds/fit_calls,
            projected_full_nested_LOOCV_seconds_upper_heuristic=cv_seconds*42,
            note='Measured nonnested fixed-recipe LOOCV with fold-only scalers. Nested 42 outer folds must rerun inner selection on 41; estimate assumes similar tiny kernel systems and cached output Gram. Not a measured nested run.'),
        elapsed_seconds=time.perf_counter()-start)
    (HERE/f'{region}_data_diagnostics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(region=region,spatial_floor=spatial_floor['rmse_T'],condition_floor=pod[-1]['validation_projection']['rmse_T'],
        best_validation=dict(embedding=best_val['input_embedding'],family=best_val['family'],rmse=best_val['validation']['rmse_T']),
        best_LOOCV=dict(embedding=best_cv['input_embedding'],family=best_cv['family'],rmse=best_cv['training_LOOCV']['rmse_T']),elapsed=out['elapsed_seconds'])),flush=True)
    return out


def main():
    if (HERE/'data_diagnostics.json').exists():raise FileExistsError('Use a fresh diagnostics version; prior report is immutable')
    split=read(V1/'cache_v1/split.json')
    with threadpool_limits(limits=2):results={r:diagnostics(r,split) for r in ('stator','rotor','airgap')}
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_split_sha256=digest(V1/'cache_v1/split.json'),
        diagnostic_code_sha256=digest(Path(__file__)),threads=2,regions=results,
        scope='Only V1 train42 and validation9 label arrays; test rows skipped into NaN placeholders. No V1 test report opened.',
        metric='RMSE over all signed sampled components; relative L2=100*norm(error)/norm(truth), same fixed 60 frames; no pointwise relative error or area weighting.',
        physical_notes=['I is native RMS in A, no division by two. Thet is native model control angle, not established dq beta.',
            'I*cos(Thet),I*sin(Thet) are only an invertible mathematical embedding away from I=0, not asserted physical dq currents.',
            'A single zero-current condition cannot establish repeated-simulation consistency or angle invariance.',
            'Nonzero zero-current field is consistent with a permanent-magnet baseline; model should not force field to zero at I=0.',
            'Differences between distinct conditions or adjacent time frames cannot identify Maxwell numerical noise. Replicates and mesh/time convergence studies are missing.'],
        evaluation_integrity=['V1 test metrics were already observed in the prior study; subsequent use is reused-holdout evaluation, not fresh blind testing.',
            'The same validation9 has also guided earlier selection; further gains require train42 nested condition CV, and ideally newly simulated locked cases.',
            'Reported training LOOCV evaluates each fixed diagnostic recipe. Choosing the best LOOCV value and reporting that minimum is selection-biased; nested outer assessment is required.',
            'For fold-safe cycle-POD neural CV, refit mean/scalers/POD using only each training fold; V1 train42 POD would leak held-fold targets.',
            'Pure full-cycle kernel controls can compute exact crossvalidated full-field SSE from Gram matrices without any global target-POD fit.'])
    (HERE/'data_diagnostics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# V2 训练/验证数据瓶颈诊断','','只使用旧划分中的 42 个训练工况和 9 个验证工况。测试标签行没有转换为数值数组，保留 NaN 占位。CPU 限 2 线程。','',
        '| 区域 | V1 空间 POD 验证投影 RMSE/T | 训练工况全周期子空间验证投影 RMSE/T | 诊断中最低验证 RMSE/T |', '|---|---:|---:|---:|']
    for r,a in results.items():
        lines.append(f"| {r} | {a['V1_spatial_POD_validation_floor']['rmse_T']:.7f} | {a['whole_cycle_condition_POD']['rank_diagnostics'][-1]['validation_projection']['rmse_T']:.7f} | {a['diagnostic_lowest_validation']['validation']['rmse_T']:.7f} |")
    lines+=['','上述投影是使用验证真值求系数的 oracle 下界，不是可部署预测成绩。诊断回归使用训练输出的线性组合，包含固定多项式及高斯核，不是神经网络。','',
        '零电流仅 case0，位于训练集。零电流场不是零，因此不能令输出整体乘以 I 并在 I=0 强制归零。可以学习训练零电流基场 B0，再学习电流响应；是否施加增量随 I 消失，需要近零数据与验证支持。',
        'I·cos(Thet)、I·sin(Thet) 可作为数学输入映射，近零电流时自然消除角度奇异性；不能无依据把它们命名为物理 dq 电流。饱和映射仅是已列明尺度的诊断特征，没有证明磁路实际饱和参数。','',
        '每区域 JSON 保留全部候选、全场相对 L2、最差工况与逐工况结果；不计算接近零真值下容易发散的逐点相对误差。相邻工况和时间差分只是信号变化，不足以认定 Maxwell 噪声。需要同工况重复求解、网格及时间步收敛数据才能估计数值误差底限。','',
        'V1 测试结果已经看过，V2 不能称为全新盲测。原验证集也已多次参与选择。建议在原 train42 内做嵌套工况 CV；内层选模型，外层仅评价一次并报告全场及最差工况。外层每折的 POD、均值、尺度和训练误差热点必须重新拟合。测试九工况仍不参与调参，最后仅作明确标注的旧留出集复评。',
        '本脚本执行的是每个固定诊断配置的训练 LOOCV，不是 nested CV。最低 LOOCV 是探索结果，不能当作无偏选型后精度。', '']
    for r,a in results.items():
        b=a['diagnostic_lowest_validation'];cv=a['diagnostic_lowest_training_LOOCV'];cost=a['analytic_cv_cost']
        lines.append(f"- {r}: 最低验证 {b['input_embedding']}/{b['family']}, RMSE={b['validation']['rmse_T']:.7f}T, L2={b['validation']['relative_L2_pct']:.4f}%, worst={b['validation']['worst_case_rmse_T']:.7f}T；最低训练LOOCV {cv['input_embedding']}/{cv['family']}, RMSE={cv['training_LOOCV']['rmse_T']:.7f}T。{cost['fit_calls']} 个小系统耗时 {cost['seconds']:.2f}s；完整nested粗略上界估计 {cost['projected_full_nested_LOOCV_seconds_upper_heuristic']:.1f}s，未实际执行nested。")
    (HERE/'data_diagnostics.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
