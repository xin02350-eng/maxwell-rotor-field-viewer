# 双版本模型推理说明

完整包解压后，保留并列的 `new60_20261014` 和 `new60_opt_v2` 文件夹。前者提供原始数据、几何输入和固定划分；后者提供本轮模型。进入 `new60_opt_v2`，用 Python 3.10 及 NumPy 2.2.6 运行下列命令。纯推理不需要 PyTorch、scikit-learn，也不需要重新训练。

本地网页可运行 `python code/serve_website_v2.py`，再打开 `http://127.0.0.1:8769/`。这个入口同时映射包内V1网站，旧版链接也可访问；仅监听本机地址。

```powershell
python -m pip install -r code/requirements-inference.txt
python code/predict_v2.py --branch fitting_full60 --region rotor --case-id 2 --frame 44 --output my_rotor_fit.npz
python code/predict_v2.py --branch prediction --region rotor --case-id 2 --frame 44 --output my_rotor_prediction.csv
```

第一条导出已参与训练的工况2的拟合结果；第二条导出预测分支对原复用测试工况2的结果。两者的模型、训练数据范围及误差口径不同。输出文件必须是尚不存在的新文件，后缀为 `.npz` 或 `.csv`；另附同名 `.json` 记录模型与误差适用范围。

`--frame` 是从0开始的帧序号，范围0～59。省略它即导出全部60帧。`--coordinates world` 为默认：转子随体坐标按该帧绝对机械角旋转到世界坐标，单位mm；`--coordinates reference` 使用原始随体坐标。磁密单位为T。转子与定子默认导出二维磁密大小 `sqrt(Br²+Bt²)`，未包含缺失的Bz；气隙输出有符号Br。

若需纯神经网络预测对照或有符号分量：

```powershell
python code/predict_v2.py --branch prediction --region rotor --model-id prediction_rotor_magnitude_nn --case-id 2 --output my_rotor_nn.csv
python code/predict_v2.py --branch prediction --region rotor --quantity Br,Bt --model-id prediction_rotor_vector --case-id 2 --output my_rotor_components.npz
python code/predict_v2.py --branch prediction --region airgap --case-id 2 --output my_airgap.csv
```

模型清单见 `both_versions_choices_frozen.json`。拟合版三个主模型均为Tanh神经网络；转子预测Br/Bt及神经网络大小对照是神经网络。预测主云图的解析核模型另有明确方法标记，不能称为纯神经网络。所有权重由推理加载并计算，程序不会按工况编号读取真值云图，也不会在推理中训练。

还可以输入原生RMS电流与Thet：

```powershell
python code/predict_v2.py --branch prediction --region rotor --current-rms 360 --thet-deg 54.4 --output my_native_input.npz
```

电流按原始RMS值输入，不除以2；Thet按原生角度输入，未认定为物理dq角。新输入的真实误差尚未通过新Maxwell工况验证。拟合版对新输入的输出尤其不能沿用0.0448%的训练内误差。程序输出仅覆盖原60个位置及原空间测点，不能把连续工况输入接口理解为任意空间/时间位置的已验证精度。

推理只从原缓存读取白名单中的输入、点ID、坐标、角度和时间，未读取field/coeff等目标字段。加载前校验冻结选择、模型、原推理模块和输入缓存SHA256。`qa/cli_v2_check.json` 记录11个冻结模型在三个输入和三个帧上与原方程的逐值对照，最大差异均为0T；实际CSV/NPZ和世界坐标输出也已检查。

网页CSV采用q16显示精度，网页指标来自量化前原精度。进行计算、论文绘图或再次评价时，请使用 `evaluation` 下的float64科学数组或本脚本输出，勿从网页截图反算误差。科学数组中的truth仅用于评价，不是推理必需输入。
