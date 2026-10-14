"""V5 document prose uses final recorded counts and per-scope results only."""
import json
from pathlib import Path

def summary(root):
    p=Path(root)/'final_summary.json'
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.is_file() else None

def experiment_count_text(root):
    j=summary(root)
    if j is None:return '脚手架尚无最终实验计数，不预填训练次数或优化结果。'
    rows=j.get('v5_experiment_counts')
    if not isinstance(rows,list) or not rows:raise ValueError('final_summary requires actual v5_experiment_counts')
    text=[]
    for r in rows:
        count=r['count']
        if type(count) is not int or count<0:raise ValueError('Experiment count must be a nonnegative integer')
        text.append(f"{r['label']}：{count}{r.get('unit','次')}。{r.get('note','')}")
    return ''.join(text)+'V3提供继承的全60线性头校准；V4开展了多种网络的重新训练；V5直接针对二维磁密范数损失，从既有42工况网络权重进行梯度微调。微调、从头训练、51工况补训和线性头校准分别计数；继承V3全60拟合模型不计为V5新增训练，解析候选和推理组合不混算为梯度神经网络训练。'

def comparison_text(root):
    j=summary(root)
    if j is None:return ''
    rows=j.get('scope_results')
    if not isinstance(rows,list) or not rows:raise ValueError('final_summary requires actual scope_results')
    text=[]
    for row in rows:
        m=row['metrics'];key=row['training_scope'];scope=row['evaluation_role']
        if key not in {'original42','final51','full60'}:raise ValueError('Unknown actual training scope')
        achieved=row['below_1pct']
        if type(achieved) is not bool or achieved!=(m['relative_L2_pct']<1):raise ValueError('The1percent field contradicts recorded metrics')
        label={'original42':'42训练版','final51':'51训练版（原验证9已参与训练）','full60':'继承V3全60训练内拟合'}[key]
        purpose='训练内拟合' if key=='full60' else '复用测试预测' if scope=='reused_test' else scope
        conclusion=('拟合' if key=='full60' else '预测')+('已低于1%' if achieved else '尚未低于1%')
        text.append(f"{row.get('label',row['model_id'])}，{label}，{purpose}：相对L2 {m['relative_L2_pct']:.9g}%，RMSE {m['rmse_T']:.9g}T，最大绝对误差 {m['max_abs_T']:.9g}T；{conclusion}。")
        old=row.get('baseline_metrics')
        if old:
            change=(old['relative_L2_pct']-m['relative_L2_pct'])/old['relative_L2_pct']*100
            text.append(f"相对{row.get('baseline_label','所列同量对照')}的相对L2{'下降' if change>=0 else '上升'}{abs(change):.6g}%；局部峰值由{old['max_abs_T']:.9g}T变为{m['max_abs_T']:.9g}T。")
    return ''.join(text)+'42、51和60工况版本分别解释；继承的训练内近机器精度拟合不能作为未见工况预测精度，旧9测试再次复用也不是新盲测。'
