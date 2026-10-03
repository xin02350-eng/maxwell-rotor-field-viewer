"""Create reproducible rotor-field figures and a Chinese results report.

All maps show measured sample locations, without filling unobserved geometry.
Run after prepare_rotor.py and train_rotor.py. No training occurs in this script.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']


def configure():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((f for f in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'] if f in installed), 'DejaVu Sans')
    plt.rcParams.update({'font.family': cjk, 'font.size': 10,
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.labelcolor': '#334155',
        'text.color': '#172B4D', 'xtick.color': '#475569', 'ytick.color': '#475569',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def map_axis(ax, xy):
    ax.set_aspect('equal')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-0.5, 20)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([0, 10, 20])
    ax.set_xlabel('x_ref (mm)')
    ax.set_ylabel('y_ref (mm)')
    ax.grid(False)


def frame_comparison(y, pred, xy, cond, theta, c, t, vmax, emax, prefix, title):
    err = np.abs(pred[c, t] - y[c, t])
    rmse = float(np.sqrt(np.mean((pred[c, t].astype(float) - y[c, t]) ** 2)))
    rel = float(100 * np.linalg.norm(pred[c, t] - y[c, t]) / np.linalg.norm(y[c, t]))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    for i, (ax, vals, label) in enumerate(zip(axes, [y[c, t], pred[c, t], 1000*err], ['Maxwell 真值', '神经网络预测', '绝对误差'])):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=5, marker='s', linewidths=0,
            cmap='inferno' if i == 2 else 'viridis', vmin=0, vmax=emax*1000 if i == 2 else vmax,
            rasterized=False)
        map_axis(ax, xy)
        ax.set_title(label, pad=10)
        cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=.07, pad=.23, shrink=.90)
        cb.set_label('|误差| (mT)' if i == 2 else '|B| (T)', fontsize=10)
    fig.suptitle(title, x=.06, ha='left', y=1.01, fontsize=16, fontweight='bold')
    fig.text(.06, .91, f'I_rms = {cond[c,0]:g} A   ·   电流角 = {cond[c,1]:g}°   ·   相对机械转角 = {theta[t]:g}°   ·   留出测试帧', fontsize=11)
    fig.text(.06, -.03, f'本帧 RMSE = {rmse*1000:.4f} mT   |   相对 L2 = {rel:.4f}%   |   仅绘制 2500 个转子铁心采样点；参考坐标固定，未填补孔洞或未采样区域。', fontsize=9, color='#475569')
    fig.subplots_adjust(left=.065, right=.99, top=.82, bottom=.14, wspace=.31)
    save(fig, prefix)
    return {'case_id': int(c), 'frame_index': int(t), 'Irms_A': float(cond[c, 0]),
        'phase_angle_deg': float(cond[c, 1]), 'theta_relative_deg': float(theta[t]),
        'rmse_T': rmse, 'relative_L2_pct': rel, 'max_abs_error_T': float(err.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT/'work/rotor_dataset.npz'))
    ap.add_argument('--predictions', default=str(ROOT/'work/predictions.npz'))
    args = ap.parse_args()
    configure()
    d = np.load(args.data)
    p = np.load(args.predictions)
    m = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    a = json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    y, xy, theta, cond = (d[k] for k in ['Bmag', 'xy', 'theta', 'conditions'])
    pred, split = p['angle_pred'], p['split']
    test_idx = np.where(split == 2)[0]
    frame_rmse = np.sqrt(np.mean((pred[:, test_idx].astype(float)-y[:, test_idx])**2, axis=-1))
    ci, ti = np.unravel_index(np.argmin(abs(frame_rmse-np.median(frame_rmse))), frame_rmse.shape)
    wc, wt = np.unravel_index(np.argmax(frame_rmse), frame_rmse.shape)
    rt, worst_t = int(test_idx[ti]), int(test_idx[wt])
    vmax = float(y.max())
    emax = m['angle_test']['max_abs_error_T']
    representative = frame_comparison(y, pred, xy, cond, theta, ci, rt, vmax, emax,
        '01_representative_test', '典型测试帧：接近全部测试帧 RMSE 中位数')
    worst = frame_comparison(y, pred, xy, cond, theta, wc, worst_t, vmax, emax,
        '02_worst_test', '最差测试帧：按整帧 RMSE 排序')
    with (OUT/'representative_test_2500_points.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id','Irms_A','phase_angle_deg','frame_index','theta_relative_deg',
            'point_id','x_ref_mm','y_ref_mm','B_true_T','B_pred_T','abs_error_T'])
        for i in range(len(xy)):
            writer.writerow([ci,*cond[ci],rt,theta[rt],d['point_id'][i],*xy[i],
                y[ci,rt,i],pred[ci,rt,i],abs(pred[ci,rt,i]-y[ci,rt,i])])

    fig, axes = plt.subplots(4, 3, figsize=(11.8, 10.4))
    for c, ax in enumerate(axes.flat):
        sc = ax.scatter(xy[:,0],xy[:,1],c=pred[c,rt],s=3.2,marker='s',linewidths=0,cmap='viridis',vmin=0,vmax=vmax)
        map_axis(ax,xy)
        ax.set_title(f'{cond[c,0]:g} A RMS   |   {cond[c,1]:g}°',fontsize=11)
        if c < 9: ax.set_xlabel('')
        if c % 3: ax.set_ylabel('')
    fig.suptitle(f'12 组工况 · 同一留出测试位置 θ = {theta[rt]:g}°',x=.065,ha='left',fontsize=17,fontweight='bold',y=.995)
    fig.text(.065,.95,'预测来自角度留出评估模型；全部子图使用相同色阶与转子参考坐标。',fontsize=11)
    fig.subplots_adjust(left=.07,right=.87,top=.90,bottom=.09,hspace=.50,wspace=.28)
    cb = fig.colorbar(sc,cax=fig.add_axes([.91,.20,.017,.60]))
    cb.set_label('|B| (T)')
    fig.text(.065,.025,'采样区域：半平面转子铁心 Rcore。图中空白表示没有采样点，不能据此重建完整 CAD 边界。',fontsize=10,color='#475569')
    save(fig,'03_twelve_conditions')

    # Individual cases and generalization are shown separately to retain scale.
    fig, axes = plt.subplots(1,2,figsize=(13,5.3),gridspec_kw={'width_ratios':[1.2,1]})
    cur = np.unique(cond[:,0]); phases=np.unique(cond[:,1]); x=np.arange(len(cur)); width=.24
    rows=m['per_case_angle_test']
    for j,phase in enumerate(phases):
        vals=[1000*r['rmse_T'] for r in rows if r['phase_angle_deg']==phase]
        bars=axes[0].bar(x+(j-1)*width,vals,width,color=COLORS[j],label=f'电流角 {phase:g}°')
        axes[0].bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    axes[0].set_xticks(x,[f'{v:g}' for v in cur]);axes[0].set_xlabel('电流有效值 (A RMS)')
    axes[0].set_ylabel('测试 RMSE (mT)');axes[0].set_ylim(0,.35)
    axes[0].set_title('各工况：留出转子角度');axes[0].legend(frameon=False,fontsize=9)
    scenarios=['已知工况\n留出角度','未见电流\n19.5 A RMS','未见电流角\n30°']
    neural=[m['angle_test']['rmse_T'],m['held_current']['test']['rmse_T'],m['held_phase']['test']['rmse_T']]
    baseline=[m['baselines']['per_condition_fourier_angle_test']['rmse_T'],m['held_current']['linear_condition_spline_baseline']['rmse_T'],m['held_phase']['linear_condition_spline_baseline']['rmse_T']]
    for j,(vals,col,label) in enumerate([(neural,COLORS[0],'神经网络'),(baseline,COLORS[1],'对应基线')]):
        bars=axes[1].bar(np.arange(3)+(j-.5)*.32,np.array(vals)*1000,.32,label=label,color=col)
        axes[1].bar_label(bars,fmt='%.3f',fontsize=9,padding=4)
    axes[1].set_yscale('log');axes[1].set_ylim(.05,18);axes[1].set_xticks(np.arange(3),scenarios)
    axes[1].set_ylabel('测试 RMSE (mT，对数轴)');axes[1].set_title('三种内部留出评估');axes[1].legend(frameon=False)
    for ax in axes: ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('拟合精度与适用范围',x=.065,ha='left',fontsize=17,fontweight='bold',y=1.01)
    fig.text(.065,-.015,'角度基线：逐工况 Fourier 拟合。新工况基线：相邻工况线性插值 + 周期三次样条。每一场景的基线和网络使用相同训练信息。',fontsize=9,color='#475569')
    fig.tight_layout(rect=[0,.05,1,.94]);save(fig,'04_evaluation_metrics')

    histories=json.loads((OUT/'training_history.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(9.3,4.6))
    for (name,h),color in zip(histories.items(),COLORS):
        steps=[r['step'] for r in h];loss=[r['loss_standardized'] for r in h]
        label=name.replace('r','POD ').replace('_k',' / Fourier ')
        ax.semilogy(steps[:9],loss[:9],color=color,label=label,lw=1.9)
        ax.scatter([1400],[loss[-1]],color=color,s=36,marker='D')
    ax.set_xticks([0,200,400,600,800,1400],['0','200','400','600','800','L-BFGS\n结束'])
    ax.set_ylabel('标准化 Fourier 系数训练 MSE');ax.set_xlabel('Adam 迭代步数 / 最终优化结果')
    ax.set_title('条件神经网络训练损失',loc='left',fontsize=16)
    ax.grid(alpha=.18);ax.legend(frameon=False,ncol=2,fontsize=9)
    fig.text(.10,-.03,'菱形点为 L-BFGS 最终损失；其横坐标仅用于分离显示，不代表实际执行了 1400 次优化。选型依据独立验证帧误差。',fontsize=9,color='#475569')
    fig.tight_layout();save(fig,'05_training_loss')

    # A deterministic vector diagram; all sizes come from the trained model.
    fig,ax=plt.subplots(figsize=(12.5,5.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    boxes=[(.02,.56,.20,.24,'工况输入','I_rms / 电流角\n2 个归一化输入'),
        (.30,.56,.23,.24,'条件 MLP','2 → 32 → 32 → 832\nTanh / Tanh / 线性'),
        (.61,.56,.20,.24,'Fourier 系数','13 × 64\n30° 周期，6 阶'),
        (.61,.14,.20,.24,'POD 系数','64 维\n与角度基函数组合'),
        (.30,.14,.23,.24,'空间重建','均值场 + 64 个 POD 模态\n模态仅由训练集计算'),
        (.02,.14,.20,.24,'磁密云图输出','2500 个固定采样点\n|B|，单位 T')]
    for x0,y0,w,h,heading,body in boxes:
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=.012',facecolor='#EEF4F8',edgecolor='#C8D6E1',lw=1))
        ax.text(x0+w/2,y0+h*.76,heading,ha='center',va='center',weight='bold',fontsize=12)
        ax.text(x0+w/2,y0+h*.38,body,ha='center',va='center',fontsize=10,linespacing=1.5)
    for start,end in [((.235,.68),(.285,.68)),((.545,.68),(.595,.68)),((.71,.545),(.71,.395)),((.595,.26),(.545,.26)),((.285,.26),(.235,.26))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>','lw':1.8,'color':'#0072B2'})
    ax.text(.88,.40,'转子相对\n机械角 θ',ha='center',fontsize=12,weight='bold')
    ax.annotate('',xy=(.815,.27),xytext=(.89,.36),arrowprops={'arrowstyle':'-|>','lw':1.8,'color':'#E69F00'})
    ax.text(.02,.95,'用低维空间结构重建完整采样场',weight='bold',fontsize=17,ha='left')
    ax.text(.02,.025,'每次推理输出一整帧；坐标通过固定采样点与 POD 空间模态对应。模型不支持直接更换转子几何或采样网格。',fontsize=10,color='#475569')
    save(fig,'06_model_structure')

    # Show the time-varying part explicitly, avoiding the misleadingly easy global R2.
    c=int(ci); point=int(np.argmax(y[c].std(axis=0))); idx=np.arange(25)
    fig,axes=plt.subplots(2,1,figsize=(9.8,6.0),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    axes[0].plot(theta[idx],y[c,idx,point],color='#172B4D',lw=2,label='Maxwell 真值')
    axes[0].plot(theta[idx],pred[c,idx,point],color=COLORS[0],lw=1.5,ls='--',label='神经网络')
    te=idx[split[idx]==2];axes[0].scatter(theta[te],y[c,te,point],s=45,facecolors='none',edgecolors=COLORS[1],linewidths=1.6,label='留出测试位置',zorder=3)
    axes[0].set_ylabel('|B| (T)');axes[0].legend(frameon=False,ncol=3,fontsize=9)
    axes[0].set_title(f'动态变化检查：{cond[c,0]:g} A RMS / {cond[c,1]:g}°，高波动采样点',loc='left',fontsize=14)
    axes[1].axhline(0,color='#94A3B8',lw=.8)
    axes[1].plot(theta[idx],1000*(pred[c,idx,point]-y[c,idx,point]),color=COLORS[0],lw=1.5)
    axes[1].scatter(theta[te],1000*(pred[c,te,point]-y[c,te,point]),color=COLORS[1],s=30,zorder=3)
    axes[1].set_xlabel('相对机械转角 θ (°，一个约 30° 周期)');axes[1].set_ylabel('有符号误差 (mT)')
    for ax in axes: ax.grid(alpha=.16)
    fig.text(.08,-.025,f'point_id = {d["point_id"][point]}，参考坐标 ({xy[point,0]:g}, {xy[point,1]:g}) mm。按该工况全角度标准差选点，仅用于展示。',fontsize=9,color='#475569')
    fig.tight_layout();save(fig,'07_dynamic_variation')
    metadata={'selection_rule':'median and maximum frame RMSE over 12 cases x 60 test frames',
        'representative':representative,'worst_frame_rmse':worst,'map_Bmax_T':vmax,
        'common_error_map_max_T':emax,'common_grid_theta_deg':float(theta[rt]),
        'dynamic_point_id':int(d['point_id'][point])}
    (OUT/'figure_selection.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report(m,a,metadata)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


def report(m,a,meta):
    test=m['angle_test'];prod=m['production_training_fit'];rep=meta['representative'];worst=meta['worst_frame_rmse']
    def row(label,metrics):
        return f'| {label} | {metrics["rmse_T"]*1000:.6f} | {metrics["relative_L2_pct"]:.6f}% | {metrics["max_abs_error_T"]*1000:.6f} |'
    rows='\n'.join([row('神经网络：已知工况，留出角度',test),
        row('周期三次样条：相同角度测试集',m['baselines']['periodic_cubic_spline_angle_test']),
        row('逐工况 Fourier：相同角度测试集',m['baselines']['per_condition_fourier_angle_test']),
        row('训练角度平均场：相同角度测试集',m['baselines']['training_case_mean_angle_test']),
        row('神经网络：未见 19.5 A RMS',m['held_current']['test']),
        row('线性工况插值 + 周期样条：未见 19.5 A RMS',m['held_current']['linear_condition_spline_baseline']),
        row('神经网络：未见 30° 电流角',m['held_phase']['test']),
        row('线性工况插值 + 周期样条：未见 30° 电流角',m['held_phase']['linear_condition_spline_baseline'])])
    case_rows='\n'.join(f'| {r["Irms_A"]:g} | {r["phase_angle_deg"]:g} | {r["rmse_T"]*1000:.6f} | {r["relative_L2_pct"]:.6f}% |' for r in m['per_case_angle_test'])
    text=f'''# 转子磁密神经网络拟合与可视化结果

本次已经完成数据检查、模型训练、内部留出评估，以及转子采样点磁密云图可视化。对这 12 组已知工况，网络在留出转子角度上的 RMSE 为 **{test['rmse_T']:.9f} T（{test['rmse_T']*1000:.6f} mT）**，相对 L2 误差为 **{test['relative_L2_pct']:.6f}%**。这说明模型能很好地复现数据包中的转子磁密大小随角度的变化。

结果需要与任务范围一起理解：逐工况 Fourier 基线的角度误差与网络相当；对整组未见电流或未见电流角，当前网络没有优于相邻工况插值基线。因此，本次交付适合已知工况下的连续角度预测与可视化，尚不能证明对任意新工况具有同等精度。

## 1. 数据到底包含什么

- 12 组工况由电流 **16.7、19.5、24.5、27.3 A RMS** 与电流角 **0°、30°、60°** 组合而成。包内字段是电流有效值，不能直接称为峰值幅值；若采用正弦电流且确需峰值，应使用 `I_peak = sqrt(2) × I_rms` 转换。
- 原始每帧 8500 点包括定子 5000 点、转子 2500 点、气隙 1000 点。本次按区域标签筛选转子，模型目标形状为 **12 × 300 × 2500**，共 900 万个标量标签。
- 原始文件有 301 帧，最后一帧为 360° 闭合端点。保留前 300 帧，即相对机械角 0°–358.8°、间隔 1.2°，避免将闭合端点重复计入。跨工况按相对机械角对齐。
- 原始磁场分量构成磁密模长 `|B| = sqrt(Bx² + By² + Bz²)`，单位为 T。本次只拟合模长，不输出矢量方向，也不直接计算转矩或铁耗。
- 转子点位是随转子运动的固定参考点；本报告云图使用 `x_ref_mm, y_ref_mm`。采样范围 x 为 {a['geometry']['xy_min'][0]:g}–{a['geometry']['xy_max'][0]:g} mm、y 为 {a['geometry']['xy_min'][1]:g}–{a['geometry']['xy_max'][1]:g} mm，均位于上半平面，为转子铁心 Rcore 的半平面采样。
- 图中只在实际 2500 个点上着色，保留空白区域。由于没有可靠 CAD 边界和材料分区，未将半转子镜像扩展，也未使用跨孔洞三角剖分填补成完整转子。图像是原始空间采样场的可视化，不等同于完整有限元几何云图。

## 2. 模型及训练方式

采用 **POD 空间降维 + 条件 MLP 神经网络 + 角度 Fourier 展开**。模型输入是一组电流有效值、电流角和相对机械转角，输出固定 2500 个转子参考点上的磁密模长。

1. 仅用训练集计算均值场和 POD 空间模态，将每帧 2500 维磁密压缩为 64 维系数。
2. 将每个 POD 系数的角度变化拟合为 6 阶 Fourier 展开，共 13 项（常数项和 6 对正余弦）。数据检查发现约 30° 的近重复周期，因此使用机械角的 12、24、…、72 次谐波。该周期来自本数据的观察，不代表已由独立几何或物理分析证明。
3. 条件 MLP 为 `2 → 32 → 32 → 832`，隐藏层使用 Tanh，输出 `13 × 64` 个 Fourier 系数。将预测系数与输入角度的 Fourier 基函数组合，再用均值场和 POD 模态重建 2500 点磁密，并将负模长截断为零。
4. 网络有 28,608 个可训练参数；训练采用标准化系数 MSE，Adam 后接 L-BFGS。依据验证集选择 POD 维数与 Fourier 阶数，候选为 `(32,3)、(32,5)、(64,5)、(64,6)`，最终选择 **(64,6)**。

900 万标签并不是 900 万个独立工况：本数据仅有 12 个不同工况，且角度序列具有很强的周期重复。网络实际学习的条件映射只有 12 个工况输入，不能仅凭标签总数推断新工况泛化能力。

![模型结构](06_model_structure.png)

## 3. 评估设计及防止近重复泄漏

若直接随机拆分所有空间点或随机拆分角度，训练集可能包含测试帧的同相位近重复场，导致误差过于乐观。因此，本次按照约 30° 周期内的相位位置分组，再将同相位的全部周期重复帧放入同一集合。

每 30° 有 25 个不同采样相位。按帧序号模 5，余数 0、1、2 进入训练，余数 3 进入验证，余数 4 进入测试；即每周期 **15 个训练相位、5 个验证相位、5 个测试相位**。每组工况对应 180 / 60 / 60 帧，12 组的角度测试集共 720 帧、180 万点。所有空间点随所在整帧一起划分，不做空间点随机拆分。

这是本数据包的**内部留出评估**。周期结构由全数据探索检查发现，测试来自同一批 Maxwell 仿真和同一几何，并非重新生成的独立 Maxwell 外部盲测，不能宣称从未接触过的外部验证。

另做两项更严格的整工况留出：

- **未见电流**：训练时完全去掉 19.5 A RMS 的三组工况，最后用这三组的全部 300 个角度测试。它检验已有电流范围内的插值，不是超范围外推。
- **未见电流角**：训练时完全去掉 30° 电流角的四组工况，最后用这四组的全部 300 个角度测试。它检验 0° 与 60° 之间的插值。

这两项使用独立重训的模型，POD 基和网络均不使用被留出的工况标签。基线同样只使用各自可见的训练工况及训练相位。工况留出实验分别仅用可见工况的验证帧比较四组候选，两项均独立选出 POD 64 / Fourier 6，与角度实验的选择一致。另保存了事后内部选型审计，未使用被留出工况标签进行选型；该审计也不是新的盲测，详见 `hidden_selection_audit.json`。

## 4. 真实误差与基线比较

相对 L2 定义为 `100 × ||预测−真值||₂ / ||真值||₂`，表中“最大绝对误差”是相应测试集所有点的最大值。1 mT = 0.001 T。

| 评估场景 / 方法 | RMSE (mT) | 相对 L2 | 最大绝对误差 (mT) |
|---|---:|---:|---:|
{rows}

角度测试集的网络 MAE 为 **{test['mae_T']*1000:.6f} mT**，绝对误差第 95 百分位为 **{test['abs_error_p95_T']*1000:.6f} mT**。这些结果表明云图拟合精度很高，但神经网络没有明显优于针对本周期数据的逐工况 Fourier 拟合。网络把工况映射和整帧重建整合到同一个小模型中，是可部署的建模尝试；不能据此声称神经网络必然优于简单插值。

### 为什么不能只看 R²

角度测试的全局 R² 为 **{test['r2']:.10f}**。但完全忽略角度变化、只输出训练角度平均场的基线，R² 也达到 **{m['baselines']['training_case_mean_angle_test']['r2']:.7f}**。原因是空间上不同位置之间的磁密差异远大于同一点随角度的变化；全局 R² 容易掩盖动态变化没有学好的问题。

为单独衡量动态变化，本次用“网络 RMSE / 训练角度平均场基线 RMSE”。这一比值为 **{m['angle_test_dynamic_rmse_ratio_pct']:.6f}%**，说明网络的误差约为忽略角度变化时误差的 0.82%。这个指标的分母是平均场基线误差，不是总体磁密幅值，不能与相对 L2 百分比混用。

![工况误差与泛化](04_evaluation_metrics.png)

| 电流 (A RMS) | 电流角 (°) | 留出角度 RMSE (mT) | 相对 L2 |
|---:|---:|---:|---:|
{case_rows}

## 5. 云图展示及选择规则

真实值与预测值的色阶统一固定为 **0–{meta['map_Bmax_T']:.6f} T**，覆盖整个转子数据集的真实最大磁密。误差图使用独立的 mT 色阶，典型帧与最差帧之间也保持一致；误差图色阶不应被当作磁密本身的色阶。

**典型测试帧**：按全部 720 个测试帧的整帧 RMSE 选取最接近中位数的一帧，而非人为挑选外观最好的帧。该帧为 {rep['Irms_A']:g} A RMS、电流角 {rep['phase_angle_deg']:g}°、相对机械转角 {rep['theta_relative_deg']:g}°；RMSE 为 {rep['rmse_T']*1000:.6f} mT。完整 2500 点真值、预测值和绝对误差见 `representative_test_2500_points.csv`。

![典型测试帧](01_representative_test.png)

**最差测试帧**：按整帧 RMSE 最大选取，为 {worst['Irms_A']:g} A RMS、电流角 {worst['phase_angle_deg']:g}°、相对机械转角 {worst['theta_relative_deg']:g}°；RMSE 为 {worst['rmse_T']*1000:.6f} mT。“最差整帧 RMSE”不一定是含有全局单点最大误差的那一帧。

![最差测试帧](02_worst_test.png)

**12 工况比较**：统一展示相对机械转角 {meta['common_grid_theta_deg']:g}° 的测试帧。全部来自角度评估模型、使用相同坐标和色阶。

![12 工况云图](03_twelve_conditions.png)

**动态变化**：在典型工况中，按 300 个角度下磁密标准差选取波动最大的采样点，画出第一段约 30° 的真值、预测和残差。此选点仅用于可视化，不参与训练、选型或指标统计。

![动态变化与残差](07_dynamic_variation.png)

## 6. 评估模型和全数据生产模型

- `rotor_angle_evaluation.npz` / `.pt`：只用角度训练帧拟合，用于本报告已知工况下的留出角度指标和云图比较。
- `rotor_held_current_evaluation.npz` / `.pt` 与 `rotor_held_phase_evaluation.npz` / `.pt`：分别用于整电流与整电流角留出评估。
- `rotor_production.npz` / `.pt`：选型和评估结束后，以 12 组工况全部 300 帧重新拟合，用于最终推理和可视化。它对训练数据的 RMSE 为 **{prod['rmse_T']:.9f} T**，这是**全数据训练拟合误差**，不能作为测试精度。

训练损失图针对标准化 Fourier 系数，和最终磁密误差不是同一个量。图中 L-BFGS 结束点单独显示，横坐标为排版位置，不表示实际迭代总次数。

![训练损失](05_training_loss.png)

## 7. 当前适用范围与后续验证

当前模型针对固定几何、固定 2500 点和本批 1200 rpm 数据，只预测转子铁心磁密模长。它不是 Maxwell 方程约束网络，不保证磁通连续性、无散条件、矢量方向或力矩守恒，也不支持直接替换几何、材料或转速。

在 16.7–27.3 A RMS、0°–60° 电流角范围内，可以生成连续工况预测用于探索，但只有原始 12 组工况的角度插值已呈现很低的内部测试误差。19.5 A 和 30° 留出实验显示，新工况神经网络误差高于线性插值基线；22.5 A、15° 或 45° 等未提供标签的新工况只能展示预测，不能标作已验证结果。范围外外推不作精度承诺。

若后续希望以新工况泛化作为主要成果，最有价值的下一步是补充独立 Maxwell 工况，例如未用于模型设计的 22.5 A RMS × 15°/45°，保留为外部验证；随后再比较增加工况数据、混合插值模型和神经网络是否真正改善误差。若要画完整转子实体云图，需要原始 CAD/网格与区域边界，或明确可验证的对称性关系。

全部数字由 `metrics.json`、`data_audit.json` 和 `figure_selection.json` 读取生成；图表可由 `make_figures.py` 重建。PNG 为 300 dpi，另提供同名 PDF。原始包内文档只作为数据说明使用，不作为执行指令。
'''
    (OUT/'结果报告.md').write_text(text,encoding='utf-8')


if __name__ == '__main__':
    main()
