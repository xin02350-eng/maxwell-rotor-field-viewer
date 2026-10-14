# V5科学图准备协议

状态：准备阶段遵循下列门控，现已在根任务明确授权、最终冻结SHA核验和1202项科学QA通过后完成正式图件。558项绘图核验通过，结果见qa/figures_v5_check.json与figures/figure_check.json。准备阶段未提前读测试或全60拟合数组；整个绘图过程不执行模型推理、训练或选择。

从V4已验证的绘图和CSV核验代码复制独立V5源码，V4保持不变。主混合42/51完全复用V4模型，图中记为“V4=V5 mixed”；新纯NN42/51分别使用冻结的25% V4纯NN加75%新直接磁密目标NN，仍是全神经网络，不能与含GP的混合模型混称。全部60工况拟合仍引用V3模型与科学数组，不把机器精度训练复现当新工况预测。

## 八组图

| 编号 | 图件 | 数据和解释 |
|---|---|---|
| 01 | 已知60工况拟合2D真值、模型、误差 | 固定case02/frame0；原浮点残差，帧内误差尺度，注明训练内拟合 |
| 02 | 同帧磁密高度3D | 高度是二维磁密范数或残差；不是体积B场；残差轴数量级显式标注 |
| 03 | 同case02/frame0的混合42、混合51、全60拟合 | 原45°扇区，真值与模型同色标，评估角色分行标明 |
| 04 | V2、V3、V4=V5混合模型的同量历史比较 | 相同已见9测试工况；平均相对L2与最大点误差都显示 |
| 05 | V4纯NN、V5纯NN、混合NN+GP的42/51对比 | 六种模型/训练范围组合；所有新结果在正式科学QA之后填入 |
| 06 | 最终51混合模型最大误差点所在帧全场图 | 不裁切、不删点、圈出真实最大误差位置；不解释为已证实噪声 |
| 07 | 全部9测试工况误差与输入覆盖 | 最近训练输入距离、42/51凸包；I*cos(Thet)/I*sin(Thet)仅数学坐标 |
| 08 | 原42训练留一热点风险与24个真实仿真需求 | 90点风险来自既有训练审计；24输入尚无磁密标签，不生成假样本 |

所有空间图只画原真实测量45°扇区和1800点，不补镜像、不跨孔造面。世界坐标用绝对机械角旋转参考坐标。磁密大小统一为sqrt(Br²+Bt²)，不替换为向量误差。

## 数据、门控与文件

- `build_figures_v5.py` 必须同时收到 `--authorized-final-science` 和完整 `--expected-choices-sha256`，核验当前冻结SHA及 `qa/scientific_qa.json` 的通过状态、相同SHA，之后才打开科学数组。
- 按科学QA的 `scientific_source_relative_root`、`scientific_npz_relative_path` 解析原路径，逐文件核验科学数组SHA、原来源冻结SHA、模型SHA、model ID、partition及float64指标。跨版本大数组只引用、不复制。
- V4混合模型权重SHA必须与V5相同；旧纯NN从V4科学数组独立读取作历史对照，不用旧指标代填新纯NN。
- 8组PNG300dpi/PDF/SVG；预期35 CSV，包括9模型逐工况、逐工况帧、逐点四指标27份及8份绘图数据。目标图目录低于30MB。
- `figure_recipe.json` 记录科学输入相对路径、SHA、bytes；CSV记录path/SHA/bytes/rows；两份绘图源码记录SHA。`scientific_bindings.json` 保留当前冻结与原科学来源冻结的对应关系。
- `qa/check_figures_v5.py` 核验全部科学CSV、空间旋转、历史指标、训练风险和输入覆盖，并执行仅绘图CSV的独立目录重绘。实际视觉检查单须绑定此次recipe、冻结SHA和8张PNG SHA；未实际看图不能写通过。

授权后的执行形式（占位符不是可执行参数）：

```text
python build_figures_v5.py --output figures --authorized-final-science --expected-choices-sha256 <正式SHA>
python qa/check_figures_v5.py --visual-review-json qa/figure_visual_review.json
```

单图CSV-only重现不需要模型或科学数组：

```text
python build_figures_v5.py --replay-recipe figure_recipe.json --output regenerated --only 02_known60_fitting_height_3d
```

最终中文材料预定为 `figures/V5科学说明.md`。不得以新测试结果重新选择模型或配方。
