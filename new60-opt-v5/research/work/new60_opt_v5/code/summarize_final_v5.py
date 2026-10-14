"""Write final statements from completed frozen scientific QA, without labels."""
from pathlib import Path
import hashlib,json
from delivery_document_v5 import docx
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    qa=read(ROOT/'qa/scientific_qa.json');old=read(ROOT.parent/'new60_opt_v4/qa/scientific_qa.json')
    if qa['all_checks_pass'] is not True or qa['choices_sha256']!=sha(ROOT/'both_versions_choices_frozen.json'):raise ValueError('Final QA required')
    oldrows={(r['id'],r['partition']):r for r in old['results']}
    labels={'prediction_rotor_magnitude':'转子42训练工况混合模型','prediction_rotor_pure_nn':'转子42训练工况纯神经网络',
        'prediction_rotor_final51_magnitude':'转子51训练工况混合模型','prediction_rotor_final51_pure_nn':'转子51训练工况纯神经网络',
        'prediction_stator_magnitude':'定子42训练工况预测','prediction_airgap_Br':'气隙42训练工况有符号Br预测',
        'fitting_rotor_magnitude':'转子全60训练内神经网络拟合','fitting_stator_magnitude':'定子全60训练内神经网络拟合','fitting_airgap_Br':'气隙全60训练内神经网络拟合'}
    rows=[]
    for r in qa['results']:
        if r['partition'] not in ('reused_test','training_fit'):continue
        row=dict(model_id=r['id'],label=labels[r['id']],training_scope=r['training_scope'],evaluation_role=r['partition'],metrics=r['metrics'],below_1pct=r['metrics']['relative_L2_pct']<1)
        prior=oldrows.get((r['id'],r['partition']))
        if prior and r['id'].endswith('pure_nn'):
            row.update(baseline_metrics=prior['metrics'],baseline_label='V4相同训练范围的纯神经网络')
        rows.append(row)
    counts=[dict(label='原42工况直接磁密损失梯度微调',count=4),dict(label='固定配方51工况追加梯度微调',count=1),
        dict(label='本轮从头训练神经网络',count=0),dict(label='本轮失败微调',count=0),dict(label='预先限定的全场组合比较',count=60),
        dict(label='继承全60最后线性层校准模型',count=3,unit='个',note='属于V3既有结果，本轮没有重新训练。')]
    summary=dict(version='NEW60_OPT_V5_FINAL',choices_sha256=qa['choices_sha256'],v5_experiment_counts=counts,scope_results=rows,
        target='Reduce both average and local error without confusing training fitting and prediction.',
        rotor_prediction_below_1pct=False,training_fit_is_not_prediction=True,previous_test_already_seen=True,
        validation_repeatedly_used_for_selection=True,final51_independent_validation=False,
        model_recipe_unchanged_after_final_replay=True,scientific_qa_pass_count=qa['pass_count'])
    dest=ROOT/'final_summary.json'
    with dest.open('x',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    mixed=next(r for r in rows if r['model_id']=='prediction_rotor_final51_magnitude')['metrics']
    pure=next(r for r in rows if r['model_id']=='prediction_rotor_final51_pure_nn')
    fit=next(r for r in rows if r['model_id']=='fitting_rotor_magnitude')['metrics']
    p=pure['metrics'];o=pure['baseline_metrics']
    l2_improvement=(o['relative_L2_pct']-p['relative_L2_pct'])/o['relative_L2_pct']*100
    peak_improvement=(o['max_abs_T']-p['max_abs_T'])/o['max_abs_T']*100
    text=f'''# 沈夏鑫：第五轮磁密模型优化结果说明

## 实际结果

转子51训练工况纯神经网络的复用测试相对L2由V4的{o['relative_L2_pct']:.8f}%降至{p['relative_L2_pct']:.8f}%，相对下降{l2_improvement:.3f}%；RMSE为{p['rmse_T']:.9f}T。最大绝对误差由{o['max_abs_T']:.9f}T降至{p['max_abs_T']:.9f}T，下降{peak_improvement:.3f}%。这个版本完全使用神经网络，没有高斯核模型。

混合51训练版的相对L2为{mixed['relative_L2_pct']:.8f}%，RMSE为{mixed['rmse_T']:.9f}T，最大误差为{mixed['max_abs_T']:.9f}T。它包含神经网络和核回归，配方与V4一致。本轮新候选没有超过该混合版的验证选择结果，因此保持原模型。

转子全60工况拟合版训练内相对L2为{fit['relative_L2_pct']:.8g}%，最大残差为{fit['max_abs_T']:.8g}T。这是已知数据的数值重现，不能作为未见工况的预测准确率，也不表示Maxwell物理误差为零。

**目前转子预测还没有达到1%以下，也不能称为已经获得很小的局部误差。**全部1800个转子点、60个位置和9个复用测试工况都参与指标，没有删去大误差点。

## 这次改变了什么

原来主要拟合Br、Bt分量，最终再取√(Br²+Bt²)。这次让误差从最终磁密大小直接反向传播到神经网络，同时保留小权重的有符号分量约束。实际完成4个42训练工况微调（2/3门、分量约束权重0.02/0.2），比较预先规定的60个全场组合，再按原验证集结果确定纯神经网络组合；随后只追加1个同配方51工况微调。

42版使用原42训练、9验证、9测试工况。51版把原验证9组并入训练，因此没有独立验证集。新纯NN组合在所有点统一采用25%旧纯NN磁密加75%新直接磁密NN磁密，不为某个测试点设置单独权重。混合版比例也在复评前固定。

本轮实际5次梯度微调与60次组合比较分别计数。前轮V4完成12次从头NN训练，另有2次失败尝试；解析回归候选不计成神经网络训练。全60版继承既有神经网络的隐层和空间基，使用全60标签校准最后线性层；不是这次额外得到的泛化能力。

## 为什么局部误差仍然较大

训练数据核查确认原ZIP、整合数据和缓存一致。误差在部分半径约80.8mm的采样区域集中，工况变化时局部磁场分量变化明显。现有证据尚不足以判定这是仿真噪声、材料界面问题、坐标问题或真实的非线性变化，因此没有擅自把这些点当异常值删除。

现有60工况提供60个不同电流幅值/角度输入，而同一工况的60位置、1800空间点不能等同于增加新的电流工况。增加网络容量和训练轮数会改善训练拟合，但不保证补足工况之间的信息。

若要可靠地继续压低预测峰值，应增加真实Maxwell标签：先对同一高风险工况重复导出并核查点坐标、材料归属、网格/求解收敛；再在已有工况过渡较快的区域补充工况。包内保留24个新增工况输入请求和90个重点点位核查请求；它们是待仿真的输入清单，没有伪造新输出。旧12工况属于不同电机和坐标采样，不能直接追加成72工况。

## 评价和可视化口径

原9测试工况已在前轮使用，这次是冻结后的复用测试，不是新盲测。原验证集也已被多轮模型选择使用。报告保留这些限制，不以训练误差代替预测误差。

2D是真实45°扇区点云，3D高度是磁密T，横纵坐标为mm；3D图不是实体三维磁场。磁密是√(Br²+Bt²)，未包含Bz。转子按绝对机械角旋转。误差指标采用原float32组件转float64计算；网页误差云图独立编码原精度残差，避免显示量化后相减制造假误差。

完整包附原始输入ZIP、模型权重、NumPy推理、训练源代码/日志、分区评价、2D/3D网站、科学图表与检查记录。完整解压后保留同级目录关系，按V5实验与使用说明运行。冻结SHA：{qa['choices_sha256']}。
'''
    out=ROOT/'第五轮优化结果说明.md';out.write_text(text,encoding='utf8');docx(ROOT/'第五轮优化结果说明.docx',text)
    print(json.dumps(dict(summary=str(dest),rotor51pure=p,rotor51mixed=mixed,rotorfit=fit),ensure_ascii=False))
if __name__=='__main__':main()
