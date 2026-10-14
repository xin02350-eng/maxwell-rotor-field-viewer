"""Shared document prose from final_summary; no scientific array access."""
import json
from pathlib import Path

EXPERIMENT_COUNTS='本轮完成13次完整神经网络训练（11次点/共享网络、2次算子网络）；604个解析候选（96个联合核、360个静态局部、94个有理模型、6个邻域模型、48个局部均值模型）；82个推理组合（42个旧混合、10个纯NN组合、25个门控NN组合、5个混合组合）。另有3次已训练NN最后线性层的最小二乘校准，它们不是新的梯度网络训练。中断训练不计入13次完整训练，解析候选和推理组合也不能算作额外NN训练。'

def comparison_text(root):
    path=Path(root)/'final_summary.json'
    if not path.is_file():return '' # Scaffold generation contains no invented result.
    j=json.loads(path.read_text(encoding='utf-8-sig'))
    c=j.get('rotor_magnitude_comparison')
    if not c:raise ValueError('final_summary needs rotor_magnitude_comparison for the explicit V2-to-V3 conclusion')
    old=c['v2_reused_test'];new=c['v3_reused_test']
    achieved=bool(c['v3_prediction_below_1pct'])
    if achieved!=(new['relative_L2_pct']<1):raise ValueError('The less-than-1-percent statement contradicts original-precision summary')
    change=(old['relative_L2_pct']-new['relative_L2_pct'])/old['relative_L2_pct']*100
    direction='相对下降' if change>=0 else '相对上升'
    goal='达到' if achieved else '尚未达到'
    maximum='下降' if new['max_abs_T']<=old['max_abs_T'] else '上升'
    return (f"转子二维磁密范数的9工况复用测试相对L2，从V2的{old['relative_L2_pct']:.9g}%变为V3的{new['relative_L2_pct']:.9g}%，{direction}{abs(change):.6g}%；"
            f"RMSE从{old['rmse_T']:.9g}T变为{new['rmse_T']:.9g}T。预测误差{goal}小于1%的目标。"
            f"最大绝对误差从{old['max_abs_T']:.9g}T变为{new['max_abs_T']:.9g}T（{maximum}），须与整体误差一并看待。"
            '这些是同量、同9工况的再次复评，不是新盲测；接近机器精度的全60训练内拟合不能替代这一预测结论。')
