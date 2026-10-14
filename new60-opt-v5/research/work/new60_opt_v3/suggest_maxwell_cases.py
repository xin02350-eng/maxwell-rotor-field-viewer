"""Recommend additional native condition probes from TRAIN-only response slopes.
This generates requests for new simulation, never fabricated field labels.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from scipy.spatial import Delaunay
HERE=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    output=HERE/'additional_simulation_requests';output.mkdir(exist_ok=False)
    study=HERE/'static_local_control/training_loo_diagnostics.npz'
    with np.load(study,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    cache=HERE.parent/'new60_20261014/cache_v1/rotor.npz'
    with np.load(cache,allow_pickle=False) as z:conditions=z['conditions'];ids=z['case_ids'];xy=z['xy'];point_id=z['point_id']
    indices=[ids.tolist().index(int(i)) for i in a['train_case_ids']]
    c=conditions[indices]
    def cart(c):return np.column_stack([c[:,0]*np.cos(np.deg2rad(c[:,1])),c[:,0]*np.sin(np.deg2rad(c[:,1]))])
    x=cart(c);allx=cart(conditions);mean=x.mean(0);scale=np.maximum(x.std(0),1.);n=(x-mean)/scale
    error=np.mean((a['shared_best_point_loo_predictions']-a['train_mean_fields'])**2,axis=0)
    hot=np.argsort(error)[-90:];field=a['train_mean_fields'][:,hot]
    edges=set()
    for simplex in Delaunay(n).simplices:
        for i in range(3):
            for j in range(i):edges.add(tuple(sorted([int(simplex[i]),int(simplex[j])])))
    rows=[]
    for i,j in sorted(edges):
        distance=float(np.linalg.norm(n[i]-n[j]));jump=float(np.sqrt(np.mean((field[i]-field[j])**2)))
        if distance<1e-5:continue
        center=(x[i]+x[j])/2;current=float(np.linalg.norm(center));theta=float(np.rad2deg(np.arctan2(center[1],center[0])))
        if np.min(np.linalg.norm((allx-center)/scale,axis=1))<.005:continue
        rows.append(dict(I_RMS_A=current,Thet_deg=theta,ranking_score=jump/distance,
            train_edge_case_a=int(a['train_case_ids'][i]),train_edge_case_b=int(a['train_case_ids'][j]),
            edge_input_distance_scaled=distance,train_hotspot_response_rms_jump_T=jump,
            speed_rpm=5500,initial_mechanical_deg=7.5,saved_frame_count=60,
            theta_m_absolute_start_deg=9.,theta_m_absolute_stop_deg=97.5))
    rows.sort(key=lambda r:r['ranking_score'],reverse=True)
    chosen=[]
    for r in rows:
        rx=cart(np.array([[r['I_RMS_A'],r['Thet_deg']]]))[0]
        if any(np.linalg.norm((rx-cart(np.array([[q['I_RMS_A'],q['Thet_deg']]]))[0])/scale)<.06 for q in chosen):continue
        r={'request_id':f'V3_AL_{len(chosen)+1:02d}',**r};chosen.append(r)
        if len(chosen)==24:break
    with (output/'建议新增工况.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(chosen[0]));w.writeheader();w.writerows(chosen)
    locations=[]
    for j in hot[np.argsort(error[hot])[::-1]]:
        locations.append(dict(point_id=int(point_id[j]),x_ref_mm=float(xy[j,0]),y_ref_mm=float(xy[j,1]),
            train_LOO_RMSE_T=float(np.sqrt(error[j])),train_spatial_neighbor_jump_T=float(a['roughness'][j])))
    with (output/'优先检查的90个转子测点.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(locations[0]));w.writeheader();w.writerows(locations)
    meta=dict(training_only=True,training_case_ids=a['train_case_ids'].tolist(),requested_case_count=len(chosen),
        source_sha256=digest(__file__),train_diagnostics_sha256=digest(study),cache_sha256=digest(cache),
        no_generated_field_labels=True,rank_rule='Largest train-only local response slope on Delaunay condition edges; midpoint of native mathematical Cartesian currents',
        no_numerical_noise_claim=True)
    (output/'request_provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (output/'补采样说明.md').write_text('''# 追加 Maxwell 数据建议

本目录只给出建议输入，没有生成或替换任何仿真磁密真值。排序仅使用42个训练工况的局部响应和训练留一误差，不使用验证/旧测试标签。

`建议新增工况.csv`列出原生电流RMS与Thet参数的候选中间工况。数学Cartesian坐标仅用于插值选择输入，不是已经标定的dq电流。维持原模型、5500rpm、初始机械角7.5°及原60帧（绝对角9°至97.5°），导出同一空间点ID上的有符号Br/Bt。

`优先检查的90个转子测点.csv`从训练留一误差定义。重点检查采样点是否位于材料交界、坐标/材料侧是否确定；采用固定材料侧的探针或材料内部采样，保留旧坐标真值用于对照，不能删掉旧难点后声称误差降低。对最敏感工况与测点重复求解，并分别细化网格和时间步，才能判断局部变化属于物理响应还是离散误差；现有资料不支持直接判定Maxwell噪声。

如要做新的盲测，另保留独立新工况，不参与选型，模型冻结后才打开其标签。不能把本次补采样/训练内拟合的误差冒充旧划分或新盲测精度。
''',encoding='utf8')
    print(json.dumps(meta),flush=True)

if __name__=='__main__':run()
