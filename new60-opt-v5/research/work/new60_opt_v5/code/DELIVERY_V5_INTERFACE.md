# V5 范围、科学数组复用、网页与精选交付接口

当前是独立 V5 脚手架；V4及更早冻结文件保持原样。只复制代码/模板，没有复制旧站点、科学数组或完整旧ZIP。尚未对真实模型导出、发布或打包；等待父任务最终冻结、科学检查与明确授权。

## 1. 固定推理接口

`inference_numpy_v5.py`、`predict_v5.py`、`scope_utils_v5.py` 已完成合成检查，通知父任务后固定，可加入 support_files SHA。动态导入时临时加入模块所在目录，恢复完整 sys.path；保留 NumPy + threadpoolctl 最小依赖。CLI 只访问缓存的 conditions/case_ids/point_id/xy/theta/time_s/solver_index/theta_m_relative_deg，不读取 field、coeff 或真值标签。

`--model-id` 唯一指定模型时，无需重复默认区域/分支；若显式传入冲突 region/branch 则拒绝。不指定 model-id 时，region/branch 默认为 rotor/fitting_full60，且要求该组恰有一个 role=primary。最终9个模型在预测/区域组内也必须只有一个主模型。

```text
python code/predict_v5.py --model-id prediction_rotor_final51_magnitude --case-id 1 --frame 0 --coordinates world --output case01_final51.csv
```

此例的原case01属于原验证集，但在51训练版已参加训练，元数据必须标 effective_role=train。CSV/NPZ及其JSON侧车拒绝覆盖。可用 --current-rms/--thet-deg 代替caseID；新输入明确未验证外推。CLI可推理全部原60工况和60帧，网页科学分区精简不限制模型推理入口。

## 2. 每模型真实训练范围

每行冻结choice需要：

- training_scope：original42、final51、full60 之一。
- train_case_ids（接受 training_case_ids 别名，两者并存必须集合一致）：真实原工况ID；保留传入顺序，不排序。final51为 split.train + split.validation 的原顺序。
- validation_case_ids：original42为原验证9；final51/full60为空。
- reused_test_case_ids：预测版原测试9；full60为空。
- branch：prediction 或 fitting_full60，后者只配full60。
- display_label、family、role=primary/control，可选 display_note。
- scientific_partitions：original42为['validation','reused_test']；final51为['reused_test']；full60为['training_fit']。

函数会核对实际ID集合。original42必须42+9+9，final51必须原训练42并原验证9、复用测试9，full60全部60。网页与CLI分开保存 original_split 和 effective_role，不能把51版原验证工况称作未见。全60分支继承V3的线性头校准结果，不计为V5新增训练。

根任务当前计划9模型：原42转子混合/纯NN两模型；final51转子混合/纯NN两模型；原42定子/气隙两模型；继承全60拟合三模型。预测图只保存指定科学分区，共18+18+9+9+18+18+60+60+60=270载荷。case选择器只列当前模型有科学结果的工况；不为原42或final51训练分区复制大数组。训练误差保留CSV/说明。

## 3. 科学数组可引用旧研究，须由新QA明确认可

默认本地科学路径：`evaluation/{choice.id}/{partition}_predictions.npz`。新增本地数组绑定当前V5冻结SHA，choice省略source字段（或 expected='CURRENT'），避免在冻结JSON中自引用其自身SHA。

复用V2/V3/V4数组的choice必须给：

```json
{
  "scientific_source_relative_root":"new60_opt_v3",
  "expected_scientific_choices_sha256":"原V3冻结SHA",
  "scientific_model_id":"原模型ID"
}
```

source_relative_root相对work根，表示研究根目录；实际读取其内部 `evaluation/{scientific_model_id或choice.id}/{partition}_predictions.npz`。不复制数组。原冻结文件和该原模型/推理模块SHA必须匹配；NPZ的原freeze/model/inference/source SHA必须一致，物理量与点序不可变。

V5导出只接受绑定当前V5 choices_sha256的新PASS报告。报告results按模型分区提供以下字段，不能直接把旧PASS当新检查：

```json
{
  "id":"V5模型ID","partition":"reused_test","pass":true,
  "scientific_source_relative_root":"new60_opt_v3",
  "scientific_model_id":"原模型ID",
  "expected_scientific_choices_sha256":"原V3冻结SHA",
  "scientific_npz_sha256":"这份NPZ的SHA",
  "source_sha256":"NPZ metadata中的原数据source SHA",
  "model_sha256":"冻结模型SHA",
  "scientific_reuse_explicitly_verified":true
}
```

本地新数组同样提供绑定字段，source root为new60_opt_v5，expected SHA为实际当前freeze SHA；只有旧source复用要求最后一个标记为true。V4正式检查已使用、V5接口继续接受等价字段组合 scientific_arrays_reused=true 且 scientific_arrays_saved=true，导出器同样接受；这两个字段仅在原来源报告、原数组身份/坐标/真值/独立前向/原精度指标逐项通过且数组SHA未变后写入，其他SHA绑定条件不变。NPZ keys沿用V3：truth/prediction[C,60,D]、case_ids、conditions、point_id、xy、theta（绝对机械角）、time_s、quantity、metadata_json。metadata必须含 choices_sha256/model_sha256/inference_module_sha256（兼容inference_sha256）/source_sha256。新QA每个指定分区都必须明确覆盖。

## 4. 网页schema3与原精度误差

保持真值/预测各独立q16与原精度逐帧/工况指标；residuals[component]独立编码原float64 prediction−truth，误差图取其绝对值，不能用两个量化磁密值相减。否则继承V3接近机器精度拟合会被磁密显示舍入误差淹没。CSV的 absolute_original_residual_T_q16 明确独立残差。

新增model.training_scope规范化对象及available_case_count；payload新增training_scope键和effective_role。默认完整拟合版；预测按每模型范围显示42或51标签及再次复用测试。真实45°采样扇区不镜像补圆；转子世界坐标按绝对theta旋转；3D是磁密高度T，不是实体体积场；误差独立T标尺。气隙Br保留有符号量，不能称为磁密大小。

操作选择器保留 #predictionTab/#fittingTab、#region/#model/#case/#component/#coordinates、#view2d/#view3d/#frame/#prev/#next/#play/#reset、三个Canvas、#csv/#png/#retry。坏载荷校验通过前不缓存；重试不需人工清缓存。请求代次、AbortController、单RAF循环、异步PNG固定文件名与每图独立hover保留。

真实导出等待明确授权：

```text
python code/export_website_v5.py --qa-report qa/scientific_qa.json
python qa/check_web_payloads_v5.py
python code/serve_website_v5.py
```

路由 `/new60-opt-v5/`，只绑定localhost。源码模板在code/templates/website，真实输出website，已有manifest/data/index/app拒绝覆盖。输出不会训练或推理模型。实际浏览器脚本qa/browser_v5_check.js须在任务独立Chrome中运行，不能用Node模拟代替。

## 5. 最终说明与精简交付

final_summary必须绑定choices_sha256，提供真实计数及每种范围的最终结果，不硬编码V3/V4旧次数。字段使用 v5_experiment_counts；V5梯度微调、从头训练、final51补训和线性头校准必须分别记录，最终只有实际完成项目计数。V3校准和V4重新训练是继承来源，不冒充本轮新增训练：

```json
{
 "v5_experiment_counts":[{"label":"本轮实际训练","count":0,"unit":"次","note":"这里由最终真实统计填写"}],
 "scope_results":[{"model_id":"模型ID","label":"真实名称","training_scope":"final51","evaluation_role":"reused_test","metrics":{"relative_L2_pct":2.0,"rmse_T":0.02,"max_abs_T":0.3},"below_1pct":false}]
}
```

以上数字仅是schema示例，绝不进入页面或正文。可给baseline_metrics及baseline_label作同量对照；below_1pct必须与真实L2严格一致。说明分别解释42预测、51预测及继承全60训练内拟合；平均改善也必须同时列出局部最大误差变化。

交付沿用SHA白名单 delivery_dependencies.json、只读源文件、流式ZIP64；不复制旧完整包。files路径相对work，保留必要同级源目录。必须包含当前冻结、选定模型及依赖、源码/完整日志、原合并与缓存/元数据/split、指定科学数组（可能来自V2/V3/V4原路径）、原freeze、图/CSV/QA/网站。source_plan强制website/results列出的每份科学源和原冻结在包内可解析。原完整V1/V2包仅保留链接/真实SHA。

每省略项记录路径、字节、SHA、原因及保留重建源码；不删工作原文件。已有输出拒绝覆盖，ZIP旁只保留中文Word/MD/TXT及manifest。正式 --zip-only 等父任务批准；之后另做逐member解压SHA/CRC/bytes和最小重定位推理。

## 6. 当前脚手架检查

qa/test_scaffold_v5.py调用临时合成检查：9模型270包、新V5授权旧科学数组复用、原验证case1在final51标训练、模型ID选取与冲突拒绝、不可读field毒丸证明CLI不读标签、覆盖保护、Word表格与V4源SHA不变。qa/scaffold_v5_check.json及viewer_components_v5_check.json均标 validation_stage=SCAFFOLD。它们不代表真实模型、真实科学数据或浏览器交互已经检查。
