"""Read-only audit of frozen V2 residuals, not model selection.

Reused test values are read solely from existing frozen V2 scientific exports.
No candidate is trained or scored on test labels. All proposed spatial emphasis
must subsequently be constructed from training labels only.
"""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "new60_opt_v2"
V1 = ROOT / "new60_20261014"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2 / "neural_route"))
from cycle_neural import selected_rows

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def metrics(pred, truth):
    e=pred-truth
    return dict(rmse_T=float(np.sqrt(np.mean(e*e))),mae_T=float(np.mean(np.abs(e))),
                max_abs_T=float(np.max(np.abs(e))),relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(truth)))

def pct_sse(a, k):
    a=np.asarray(a)
    return float(100*np.sort(a)[-k:].sum()/a.sum())

split=json.loads((V1/'cache_v1/split.json').read_text('utf-8'))
cache=V1/'cache_v1/rotor.npz'
with np.load(cache,allow_pickle=False) as z:
    c=z['conditions'];xy=z['xy'];point_id=z['point_id'];geom=z['dataset_hotspot'];theta=z['theta']
train=selected_rows(cache,'field',split['train']).astype(np.float64)
trmag=np.hypot(train[...,0::2],train[...,1::2])
# A fixed train-only proxy for nonlinear response: per-point inter-condition variance.
variance=np.var(trmag,axis=0).mean(axis=0)
train_order=np.argsort(variance)[::-1]
train_response_top10=np.zeros(len(point_id),dtype=bool);train_response_top10[train_order[:180]]=True
rad=np.linalg.norm(xy,axis=-1);az=np.rad2deg(np.arctan2(xy[:,1],xy[:,0]))
tree=cKDTree(xy);nd,ni=tree.query(xy,k=6)
neighbor_jump=np.sqrt(np.mean((trmag[:,:,ni[:,1:]]-trmag[:,:,:,None])**2,axis=(0,1,3)))
current_scale=float(np.median(c[np.asarray(split['train']),0][c[np.asarray(split['train']),0]>0]))

def coords(values,kind):
    r=values[:,0]/current_scale;a=np.deg2rad(values[:,1])
    if kind=='native':return np.column_stack((r,values[:,1]/90))
    if kind=='saturated_cartesian':r=np.tanh(r)
    return np.column_stack((r*np.cos(a),r*np.sin(a)))

distribution={}
for kind in ['native','cartesian','saturated_cartesian']:
    z=coords(c,kind);tr=z[split['train']];center=tr.mean(0);scale=tr.std(0);tr=(tr-center)/scale
    hull=Delaunay(tr);dist=cKDTree(tr)
    parts={}
    for part,ids in [('train',split['train']),('validation',split['validation']),('reused_test',split['test'])]:
        values=(z[ids]-center)/scale
        if part=='train':d,ix=dist.query(values,k=2);d=d[:,1];ix=ix[:,1]
        else:d,ix=dist.query(values)
        parts[part]=[dict(case_id=int(cid),I_RMS_A=float(c[cid,0]),Thet_deg=float(c[cid,1]),
            nearest_train_case_id=int(split['train'][n]),nearest_train_standardized_distance=float(dd),
            inside_train_convex_hull=bool(hh)) for cid,n,dd,hh in zip(ids,ix,d,hull.find_simplex(values)>=0)]
    distribution[kind]=parts

report=dict(scope='Frozen V2 residual diagnosis only; reused test is already seen, never a selection score.',
    train_case_ids=split['train'],validation_case_ids=split['validation'],reused_test_case_ids=split['test'],
    source_cache_sha256=sha(cache),prediction_choices_sha256=sha(V2/'prediction_choices_frozen.json'),
    distribution=distribution,models={},data_checks={},validation_blend_diagnostics={},inputs_sha256={})
models=['prediction_rotor_magnitude','prediction_rotor_magnitude_nn',
        'prediction_rotor_magnitude_direct_nn','prediction_rotor_magnitude_neural_kernel']
val_predictions={};truth_common=None;pointtop={}
for model in models:
    m={}
    for partition in ['validation','reused_test']:
        p=V2/f'evaluation/{model}/{partition}_predictions.npz'
        report['inputs_sha256'][str(p.relative_to(ROOT))]=sha(p)
        with np.load(p,allow_pickle=False) as z:
            pred=z['prediction'];truth=z['truth'];ids=z['case_ids']
            assert np.array_equal(z['point_id'],point_id) and np.array_equal(z['xy'],xy)
            expected=selected_rows(cache,'field',ids.tolist()).astype(np.float64)
            truthcheck=np.hypot(expected[...,0::2],expected[...,1::2])
            assert np.array_equal(truthcheck,truth)
        report['data_checks'][f'{model}/{partition}']=dict(cache_truth_exact=True,point_ids_exact=True,xy_exact=True)
        e=pred-truth;sq=e*e;ps=sq.sum((0,1));fs=sq.sum((0,2));cs=sq.sum((1,2));order=np.argsort(ps)[::-1]
        met=metrics(pred,truth)
        met.update(point_SSE_share_pct={str(k):pct_sse(ps,k) for k in [1,5,18,90,180,600]},
            frame_SSE_share_pct={str(k):pct_sse(fs,k) for k in [1,5,10]},
            first10_frames_SSE_pct=float(100*fs[:10].sum()/fs.sum()),
            last10_frames_SSE_pct=float(100*fs[-10:].sum()/fs.sum()),
            geometric_hotspot_point_count=int(geom.sum()),geometric_hotspot_SSE_pct=float(100*ps[geom].sum()/ps.sum()),
            train_response_top10_pct_SSE=float(100*ps[train_response_top10].sum()/ps.sum()),
            train_response_top10_point_count=int(train_response_top10.sum()),
            case_metrics=[dict(case_id=int(cid),**metrics(pred[j],truth[j]),SSE_share_pct=float(100*cs[j]/cs.sum())) for j,cid in enumerate(ids)],
            top_points=[dict(point_id=int(point_id[j]),point_index=int(j),x_ref_mm=float(xy[j,0]),y_ref_mm=float(xy[j,1]),
                radius_mm=float(rad[j]),azimuth_deg=float(az[j]),rmse_T=float(np.sqrt(ps[j]/(len(ids)*60))),
                SSE_share_pct=float(100*ps[j]/ps.sum()),train_response_variance_T2=float(variance[j]),
                train_neighbor_jump_rms_T=float(neighbor_jump[j]),nearest_spatial_distance_mm=float(nd[j,1]),
                geometric_hotspot=bool(geom[j])) for j in order[:18]],
            frames=[dict(frame_index=int(j),theta_m_absolute_deg=float(theta[j]),rmse_T=float(np.sqrt(fs[j]/(len(ids)*1800))),
                SSE_share_pct=float(100*fs[j]/fs.sum())) for j in range(60)])
        m[partition]=met;pointtop[model,partition]=set(order[:18].tolist())
        if partition=='validation':
            val_predictions[model]=pred
            if truth_common is None:truth_common=truth
            assert np.array_equal(truth,truth_common)
    report['models'][model]=m

# Global convex combinations are diagnostic validation candidates. No test blend is computed.
base=val_predictions[models[0]]
for other in models[1:]:
    b=val_predictions[other];d=b-base;e=base-truth_common
    a=float(np.clip(-np.sum(e*d)/np.sum(d*d),0,1))
    grid=[]
    for alpha in np.linspace(0,1,21):grid.append(dict(alpha=float(alpha),**metrics((1-alpha)*base+alpha*b,truth_common)))
    report['validation_blend_diagnostics'][other]=dict(base_model=models[0],other_model=other,
        error_cosine_similarity=float(np.sum(e*(b-truth_common))/(np.linalg.norm(e)*np.linalg.norm(b-truth_common))),
        convex_alpha_validation_optimum=a,optimum_metrics=metrics((1-a)*base+a*b,truth_common),
        fixed_alpha_half_metrics=metrics(.5*base+.5*b,truth_common),grid=grid,
        warning='alpha optimum uses validation labels; exploratory validation tuning, not deployable held-out accuracy. No test blend computed.')

report['top18_spatial_overlap']=[]
for part in ['validation','reused_test']:
    for other in models[1:]:
        overlap=pointtop[models[0],part]&pointtop[other,part]
        report['top18_spatial_overlap'].append(dict(partition=part,other_model=other,
            overlap_count=len(overlap),overlap_point_ids=sorted(int(point_id[j]) for j in overlap)))

for name in ['diagnosis_v2_rotor.json','diagnosis_v2_rotor.md']:
    if (OUT/name).exists():raise FileExistsError(OUT/name)
with (OUT/'diagnosis_v2_rotor.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['# V3 开始前：冻结 V2 转子误差只读诊断','',
    '使用原 train42 与 validation9；仅从已经冻结并公开评估过的 V2 scientific exports 读取复用测试结果作诊断，不据测试比较候选。模型、冻结选择、V1/V2 文件均未修改。',
    '', '| 候选 | 验证 L2/% | 复用测试 L2/% | 验证最高误差18点的 SSE 占比/% | 验证几何热点 SSE/% |',
    '|---|---:|---:|---:|---:|']
for model in models:
    v=report['models'][model]['validation'];t=report['models'][model]['reused_test']
    lines.append(f"| {model} | {v['relative_L2_pct']:.6f} | {t['relative_L2_pct']:.6f} | {v['point_SSE_share_pct']['18']:.3f} | {v['geometric_hotspot_SSE_pct']:.3f} |")
lines.extend(['','验证全场 blend 仅是已有路线的残差互补性诊断，没有计算测试 blend。'])
for other,b in report['validation_blend_diagnostics'].items():
    lines.append(f"- 核模型 + {other}：残差 cosine={b['error_cosine_similarity']:.5f}，验证最佳公共 alpha={b['convex_alpha_validation_optimum']:.5f}，L2={b['optimum_metrics']['relative_L2_pct']:.6f}%，50/50 L2={b['fixed_alpha_half_metrics']['relative_L2_pct']:.6f}%。")
for part in ['validation','reused_test']:
    v=report['models'][models[0]][part]
    lines.extend(['',f"{part} 主核模型：top1/5/18/90/180 点 SSE 占比分别为 "+'/'.join(f"{v['point_SSE_share_pct'][str(k)]:.3f}%" for k in [1,5,18,90,180])+'.',
        f"前10/后10帧 SSE：{v['first10_frames_SSE_pct']:.3f}%/{v['last10_frames_SSE_pct']:.3f}%；train-only 响应变化最高10%点的误差占比 {v['train_response_top10_pct_SSE']:.3f}%。",
        '最差工况：'+', '.join(f"case{x['case_id']} {x['relative_L2_pct']:.3f}% (SSE {x['SSE_share_pct']:.1f}%)" for x in sorted(v['case_metrics'],key=lambda x:x['rmse_T'],reverse=True)[:3])+'.'])
lines.extend(['','所有八个导出数据包的 truth 与缓存 float32 转 float64 后再计算 hypot 完全相等，原始点 ID 与坐标完全相同；没有发现精度、配对、几何索引不一致。',
    '数值 field 标签仅从指定 case 行读取。相邻点场跳跃是真实样本响应差异，未证明为数值噪声，也不能据此删除点。',
    '', '新的优化建议与泄漏防护另见父任务结论；完整逐工况、逐帧、最高18点、train 输入距离与凸包检查保存在 JSON。'])
with (OUT/'diagnosis_v2_rotor.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(status='PASS',report=str(OUT/'diagnosis_v2_rotor.json'),md=str(OUT/'diagnosis_v2_rotor.md'),
    models={m:{p:report['models'][m][p]['relative_L2_pct'] for p in ['validation','reused_test']} for m in models},
    blends={m:b['optimum_metrics']['relative_L2_pct'] for m,b in report['validation_blend_diagnostics'].items()}),ensure_ascii=False))
