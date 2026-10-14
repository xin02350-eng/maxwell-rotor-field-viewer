"""Build a new immutable two-version delivery. Never train, infer or publish.

The first sibling is a byte-identical copy of the completed V1 delivery. The
second preserves V2 completed experiments, frozen scientific evaluation,
models and their dependencies. NPZ files are only copied/hashed as bytes.
Use --preflight before formal authorization; --zip is explicit and refuses
an existing output directory or archive. See DELIVERY_V2_INTERFACE.md.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid
import zipfile
import zlib
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
DEFAULT_V1=WORKSPACE/'outputs'/'沈夏鑫_新60工况实验_20261014'
DEFAULT_OUTPUT=WORKSPACE/'outputs'/'沈夏鑫_新60双版本优化_20261014'
EXCLUDE_DIRS={'__pycache__','.git','.venv','venv','node_modules','.pytest_cache','site-packages','adaptive_pilot','portable_delivery_check','delivery_preview'}
ALLOWED={'.npz','.json','.jsonl','.csv','.md','.txt','.py','.sha256','.png','.pdf','.svg','.jpg','.jpeg','.html','.js','.css','.gz','.bin','.ico','.docx','.toml'}
OMITTED_KERNEL={'all_validation_coefficient_predictions.npz','cycles.npz'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def regular(root,rel):
    p=(root/rel).resolve()
    if not p.is_relative_to(root) or p.is_symlink() or not p.is_file():raise ValueError('Missing/unsafe source: '+str(rel))
    return p

def verify_frozen(root):
    path=root/'both_versions_choices_frozen.json';j=read(path);h=digest(path)
    if h not in (root/'both_versions_choices_frozen.sha256').read_text():raise ValueError('Both-version freeze SHA mismatch')
    for m in j['models']:
        for pathkey,hashkey in [('model_relative_path','model_sha256'),('inference_module_relative_path','inference_module_sha256'),('model_source_module_relative_path','model_source_module_sha256')]:
            if pathkey in m and digest(regular(root,m[pathkey]))!=m[hashkey]:raise ValueError('Frozen artifact changed: '+m[pathkey])
    if j.get('previous_test_already_seen') is not True or j.get('full60_fitting_scores_used_in_prediction_choice') is not False:raise ValueError('Two-version interpretation fields invalid')
    return j,h

def verify_v1(root):
    manifest=read(root/'delivery_manifest.json');records=manifest['records']
    for row in records:
        name=row.get('path',row.get('relative_path'));p=regular(root,name)
        if digest(p)!=row['sha256'] or p.stat().st_size!=row.get('bytes',row.get('size_bytes')):raise ValueError('V1 source manifest mismatch: '+name)
    listed={row.get('path',row.get('relative_path')) for row in records}|{'delivery_manifest.json'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if listed!=actual:raise ValueError('V1 source file inventory mismatch')
    return dict(manifest_sha256=digest(root/'delivery_manifest.json'),file_count=len(actual),bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))

def omit_reason(rel):
    parts=rel.parts
    if 'delivery_preview' in parts:return 'non-final document preview; final documents are regenerated when packaging'
    if 'portable_delivery_check' in parts:return 'temporary minimal extraction for relocation QA; not a research source'
    if 'adaptive_pilot' in parts:return 'incomplete pilot without a completed summary'
    if any(p in EXCLUDE_DIRS or p.lower().startswith(('synthetic_fixture','toy_fixture','failedrun','failed_run')) for p in parts):return 'runtime, incomplete pilot, failed run or synthetic fixture directory'
    if rel.suffix.lower() not in ALLOWED and rel.name!='.gitattributes':return 'not a delivery file type'
    if parts[0]=='kernel_route' and rel.name in OMITTED_KERNEL:return 'rebuildable kernel candidate coefficient/intermediate array; selected self-contained models and all search parameters/logs remain'
    if rel.name in {'running_ranking.json'}:return 'superseded progress snapshot; complete ranking retained'
    return None

def source_plan(root):
    files=[];omitted=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Symlink in study: '+str(p))
        if not p.is_file():continue
        rel=p.relative_to(root);reason=omit_reason(rel)
        if reason:omitted.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,reason=reason));continue
        # A neural run is formal only after its final result exists.
        if 'runs' in rel.parts:
            idx=rel.parts.index('runs');trial=root.joinpath(*rel.parts[:idx+2])
            if not (trial/'result.json').is_file():omitted.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,reason='neural run missing completed result.json'));continue
        files.append((p,rel))
    return files,omitted

def deployment(root):
    path=root/'website'/'deployment.json'
    if not path.is_file():return None
    j=read(path)
    if j.get('verified') is not True:return dict(verified=False,url=None,sha256=digest(path))
    u=j.get('url');x=urlsplit(u or '')
    if x.scheme!='https' or not x.netloc or x.username or x.password or any(c in (u or '') for c in '\r\n <>[]()'):raise ValueError('Invalid deployment URL')
    return dict(verified=True,url=u,sha256=digest(path))

def preflight(root,v1,reviews=()):
    root=Path(root).resolve();v1=Path(v1).resolve();choice,ch=verify_frozen(root)
    required=['final_summary.json','website/manifest.json','website/results.json','website/display_precision.json','code/predict_v2.py','code/inference_numpy_v2.py','neural_route/fitting_full60/fitting_summary.json','qa/prediction_replay/replay_report.json','qa/fitting_replay/replay_report.json','figures/figure_recipe.json','figures/optim报告材料.md','qa/optimization_inventory.json']
    missing=[p for p in required if not (root/p).is_file()]
    for p in reviews:
        if not (root/p).is_file():missing.append(str(p))
    if missing:return dict(ready=False,missing=missing,choices_sha256=ch)
    reports=[read(root/'qa/prediction_replay/replay_report.json'),read(root/'qa/fitting_replay/replay_report.json')]
    for r in reports:
        if r.get('all_checks_pass') is not True or r.get('choices_sha256')!=ch:raise ValueError('Final numeric QA failed/different freeze')
    for p in reviews:
        j=read(root/p)
        if not any(j.get(k) is True for k in ['all_checks_pass','all_pass','pass','passed','verified']) and str(j.get('status','')).upper()!='PASS':raise ValueError('Requested independent review not PASS: '+str(p))
    m=read(root/'website/manifest.json')
    if m.get('choices_sha256')!=ch or m.get('schema_version')!=2:raise ValueError('Website does not match final freeze')
    if read(root/'final_summary.json').get('choices_sha256')!=ch:raise ValueError('Final summary does not match final freeze')
    recipe=read(root/'figures/figure_recipe.json')
    if recipe.get('choices_sha256')!=ch:raise ValueError('Figure recipe does not match final freeze')
    for ref in recipe['scientific_inputs']:
        if digest(regular(root,Path('figures')/ref['path']))!=ref['sha256']:raise ValueError('Figure source SHA mismatch: '+ref['path'])
    for figure in recipe['figures']:
        if digest(regular(root,Path('figures')/figure['input_csv']))!=figure['input_sha256']:raise ValueError('Figure CSV SHA mismatch')
        for ext in ['.png','.pdf','.svg']:regular(root,Path('figures')/(figure['name']+ext))
    if digest(root/'figures/build_figures_v2.py')!=recipe['plotting_source_sha256']:raise ValueError('Plotting source SHA mismatch')
    source,omitted=source_plan(root)
    keep={r.as_posix() for _,r in source}
    for ref in recipe['scientific_inputs']:
        rel=regular(root,Path('figures')/ref['path']).relative_to(root).as_posix()
        if rel not in keep:raise ValueError('A figure source would be omitted: '+rel)
    for row in choice['models']:
        for key in ['model_relative_path','inference_module_relative_path','model_source_module_relative_path']:
            if key in row and row[key] not in keep:raise ValueError('A frozen dependency would be omitted')
    return dict(ready=True,choices_sha256=ch,models=len(choice['models']),figure_groups=len(recipe['figures']),source_file_count=len(source),source_bytes=sum(p.stat().st_size for p,_ in source),omitted=omitted,v1_directory=str(v1),website_deployment=deployment(root),reviews=[str(p) for p in reviews])

def plain(text):
    text=re.sub(r'\[([^]]*)\]\(([^)]+)\)',r'\1（\2）',text)
    text=re.sub(r'(?m)^#{1,6}\s*','',text)
    return text.replace('**','').replace('`','')

def docx(path,markdown):
    def paragraph(text,style=None):
        prop=f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        return '<w:p>'+prop+'<w:r><w:t xml:space="preserve">'+escape(plain(text))+'</w:t></w:r></w:p>'
    body=[];pending=[]
    def flush():
        if not pending:return
        rows=[]
        for line in pending:
            cells=re.split(r'(?<!\\)\|',line.strip().strip('|'))
            if all(re.fullmatch(r'\s*:?-+:?\s*',c) for c in cells):continue
            rows.append('<w:tr>'+''.join('<w:tc>'+paragraph(c.strip().replace('\\|','|'))+'</w:tc>' for c in cells)+'</w:tr>')
        body.append('<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr>'+''.join(rows)+'</w:tbl>');pending.clear()
    for line in markdown.splitlines():
        if line.startswith('|'):pending.append(line);continue
        flush()
        if not line.strip():continue
        h=re.match(r'^(#{1,3})\s+(.*)',line)
        body.append(paragraph(h[2],'Title' if len(h[1])==1 else 'Heading1') if h else paragraph(line))
    flush();ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    document='<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="'+ns+'"><w:body>'+''.join(body)+'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="850"/></w:sectPr></w:body></w:document>'
    styles='<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="'+ns+'"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:eastAsia="宋体"/><w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>'
    for name,size in [('Title',34),('Heading1',28)]:styles+=f'<w:style w:type="paragraph" w:styleId="{name}"><w:name w:val="{name}"/><w:rPr><w:b/><w:sz w:val="{size}"/></w:rPr></w:style>'
    styles+='</w:styles>'
    ct='<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
    rel='<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    dr='<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in [('[Content_Types].xml',ct),('_rels/.rels',rel),('word/document.xml',document),('word/styles.xml',styles),('word/_rels/document.xml.rels',dr)]:z.writestr(name,data.encode())

def metric_table(results,branch,partition):
    rows=['|模型 / 物理量|区域|RMSE (T)|相对 L2 (%)|最大绝对误差 (T)|','|---|---|---:|---:|---:|']
    for r in results['models']:
        if r['branch']!=branch or partition not in r['metrics']:continue
        key='signed_vector' if r['quantity']=='Br,Bt' else 'Br' if r['quantity']=='Br' else 'magnitude';m=r['metrics'][partition][key]
        rel='未定义' if m['relative_L2_pct'] is None else f"{m['relative_L2_pct']:.8g}"
        rows.append(f"|{r['id']} / {r['quantity'].replace('|',chr(92)+'|')}|{r['region']}|{m['rmse_T']:.9g}|{rel}|{m['max_abs_T']:.9g}|")
    return '\n'.join(rows)

def report(root,context):
    results=read(root/'website/results.json');fit=read(root/'neural_route/fitting_full60/fitting_summary.json');dep=context.get('website_deployment')
    inventory=read(root/'qa/optimization_inventory.json')['counts']
    online=f"已验证的在线页面：[打开优化双模式网站]({dep['url']})。" if dep and dep.get('verified') else '包内提供完整本地网站；未附已验证在线部署记录。'
    raw=['|区域|原始输出 RMSE (T)|原始输出相对 L2 (%)|原始负值数|','|---|---:|---:|---:|']
    for r in fit['rows']:raw.append(f"|{r['region']}|{r['rmse_T']:.9g}|{r['relative_L2_pct']:.8g}|{r.get('raw_negative_count')}|")
    main=f'''# 新60工况：原版与双模式优化版说明

本包保留完整 V1，并交付优化后的预测版和全60工况拟合版，附模型、复现代码、实验日志、数值检查、图与交互网站。

转子二维磁密大小的核心结果是：全60纯神经网络训练内相对L2为0.044774598%，RMSE约0.000540592T，局部最大绝对误差仍有0.04080646T；预测主云图在9个复用测试工况相对L2为3.158007363%，RMSE约0.035582226T，局部最大误差0.91381222T。两组误差用途不同，小平均误差不代表每个位置都同样准确。预测主模型是核模型Br/Bt取范数；同量纯神经网络对照为3.315663347%，完整指标列于后表。

## 先选对版本，再看误差

|文件夹 / 页面模式|数据使用|误差含义|适用场景|
|---|---|---|---|
|new60_20261014|完整原 V1，逐字节保留|按原报告解释|追溯原实验、对照与旧结论|
|new60_opt_v2 / prediction|42 工况训练，9 工况验证选型，9 工况复用测试|训练、验证和复用测试分别标注|检查原划分未训练工况的预测表现|
|new60_opt_v2 / fitting_full60|全部 60 工况参加训练|仅训练内拟合误差|复现给定 60 工况的云图与已知数据压缩|

拟合版的小误差不能替代预测误差。原 9 个测试工况在 V1 已查看，本轮最终冻结后再次评估属于复用测试，不是新盲测。大量核参数与神经网络候选使用同 9 工况验证集，验证数值可能偏乐观。需要新增 Maxwell 独立工况才可作新的盲测确认。

冻结顺序是：神经网络预测路线先冻结；全60拟合分支随后独立训练，其标签和分数不反馈预测选型；最终全局两版选择在原测试集复评前冻结。全60拟合元数据绑定的是神经网络路线冻结记录，不能据此声称全局两版选择文件早于全60拟合训练。

## 全60拟合版：冻结部署输出（训练内误差）

{metric_table(results,'fitting_full60','training_fit')}

网页遵循冻结推理接口：大小模型 nonnegative=True；气隙 Br 保留符号。定子原始大小输出存在少量负值，非负约束后的网页和科学回放指标见上表；原始未裁剪输出误差及负值计数在下表完整保留。

{chr(10).join(raw)}

## 预测版：验证集结果（用于选型）

{metric_table(results,'prediction','validation')}

## 预测版：冻结后的复用测试结果（未据此重选）

{metric_table(results,'prediction','reused_test')}

表内 Br/Bt 向量误差按全部有符号标量分量计；二维大小误差按每点 √(Br²+Bt²) 计；气隙仅 Br。不同物理量的相对 L2 不得混成同一排行榜。主云图可能使用 Br/Bt 模型后取范数，这是一项确定性输出变换，不是另训练了一张大小标签模型。

## 为什么继续优化，以及做到哪一步

V2 对照包括数学工况坐标变换、多种核/径向基/多项式趋势、核宽度与正则化、整60帧训练子空间、分块模型、纯神经网络、神经网络特征与核模型，以及单独的全60拟合。I·cos(Thet)、I·sin(Thet) 及饱和变换只是数学坐标，未声称 Thet 是物理 dq 角。没有强加稳态或相邻电周期相等。

完成神经网络相关训练{inventory['all_completed_neural_training_runs']}次，其中预测路线{inventory['prediction_neural_training_runs']}次（MLP {inventory['prediction_MLP_training_runs']}次、神经网络特征核{inventory['prediction_neural_frame_kernel_training_runs']}次），全60拟合{inventory['full60_fitting_NN_training_runs']}次。解析核完成{inventory['kernel_completed_unique_configuration_solves']}组配置求解，对不同秩合计{inventory['kernel_configuration_rank_validation_rows']}条验证排行记录；另有{inventory['adaptive_completed_validation_variants']}种自适应验证变体和{inventory['power_completed_validation_configurations']}种幂次参数配置。核配置和秩排行记录不等于神经网络训练次数。完成次数逐条见qa/optimization_inventory.json。

训练误差下降与未见工况插值误差是两个问题。全60网络的小训练误差说明当前样本可被网络较精细地拟合，不证明未见工况同样准确。转子整60帧、共同42标量权重的训练仿射子空间对验证有约1.317%的有符号向量投影下限；不同表示、非线性范数输出或分块的下限不能混用。具体负结果与投影诊断见 diagnostics/、kernel_route/route_report.md 和 neural_route/PREDICTION_ROUTE_REPORT.md。未承诺转子预测误差降至1%以下。

候选均保留完成日志与参数，未采用的配置也保留。为避免重复体积，省略可重建的核候选系数大数组及 kernel prepared cycles.npz 中间数组，逐项列于 packaging_omissions.json；最终模型、科学评估数组、源码及所有参数/排行仍在。神经网络依赖的 representation.npz 不省略。

## 如何看图和使用模型

{online} 网站默认全60训练内拟合，模式切换可查看预测。网页以真实约45°采样扇区呈现2D磁密点图和3D磁密高度图；x/y为毫米，z为磁密T，不是实体体积场，误差图高度独立标尺。未镜像扩圆或跨孔插值。CSV和PNG是显示精度副本；原精度指标来自 evaluation/ 中 float64 科学导出。

本地可在 new60_opt_v2 内运行 `python code/serve_website_v2.py`，按终端给出的本机网址访问；此服务同时映射V1链接，绑定127.0.0.1。网站全部资源同源相对路径，无外部统计脚本。模型纯 NumPy 推理入口为 new60_opt_v2/code/predict_v2.py，运行 `python code/predict_v2.py --help` 查看冻结模型和原工况/自定输入选项；任意输入不代表外推已验证。

固定60帧相对机械角1.5°至90°，绝对角加初始7.5°；转子世界坐标按绝对角旋转采样点，定子和气隙固定。Br/Bt是极坐标分量，不作为笛卡尔向量重复旋转。二维大小未包含未提供的Bz。

帧编号约定：科学图标题中的frame与CSV的frame_index从0到59；网页显示“第1至60帧”，原solver_index也从1到60。例如图中frame1对应网页第2帧，绝对机械角10.5°。比较图表时同时核对角度，避免将两种编号错移一帧。

## 文件、复现与完整性

new60_20261014 是原完整交付目录的逐字节副本，含合并原数据、元数据、原 split、缓存、模型、代码、图、网站和原说明。new60_opt_v2 保存优化实验及两个新模式，引用 ../new60_20261014 的原数据路径在包内可解析。原60个case不重复抽取；原ZIP真实性沿用V1记录的SHA校验。

V2 figures/ 保存PNG/PDF/SVG及配方/源文件；evaluation/ 保存每模型每分区原精度真值/预测和原工况/点编号；qa/ 保存独立复算和网页检查；both_versions_choices_frozen.json 绑定全部模型和推理源码SHA；两模式选择SHA为 `{context['choices_sha256']}`。历史PPT Round4仍因原权重及原划分未齐而不能宣称被严格复现；本轮新60数据已经齐备。

交付清单 delivery_manifest.json 列每文件字节数和SHA256，自身不循环哈希。ZIP另附SHA256，正式完成后需逐成员解压字节SHA、CRC与目录清单一致性复核。打包程序只复制、生成说明和校验，不训练、不重新选型、不发布网站。完整数值结论以 new60_opt_v2/final_summary.json 与独立QA报告为准。
'''
    material_path=root/'figures/optim报告材料.md'
    if material_path.is_file():
        material=material_path.read_text(encoding='utf-8-sig')
        def link(match):
            label,target=match.groups()
            if target.startswith(('https://','http://','#')):return match.group(0)
            return '['+label+'](new60_opt_v2/figures/'+target+')'
        material=re.sub(r'\[([^]]*)\]\(([^)]+)\)',link,material)
        main+='\n\n## 科学图表与分析材料\n\n来源：new60_opt_v2/figures/optim报告材料.md；图表配方和输入SHA同目录保留。\n\n'+material+'\n'
    return main

def build(root=ROOT,v1=DEFAULT_V1,output=DEFAULT_OUTPUT,make_zip=False,reviews=()):
    root=Path(root).resolve();v1=Path(v1).resolve();output=Path(output).resolve();archive=output.with_suffix('.zip')
    if output.exists() or archive.exists():raise FileExistsError('Refuse to overwrite an existing delivery/archive')
    context=preflight(root,v1,reviews)
    if not context['ready']:raise ValueError('Delivery not ready: '+json.dumps(context,ensure_ascii=False))
    context['v1_integrity']=verify_v1(v1)
    source,omitted=source_plan(root)
    total=context['v1_integrity']['bytes']+sum(p.stat().st_size for p,_ in source)
    required=int(total*(2.01 if make_zip else 1.01))+512*1024*1024
    free=shutil.disk_usage(output.parent).free
    if free<required:raise OSError(f'Insufficient space for independent directory/ZIP and 512MiB reserve: need {required}, free {free}. No files were copied.')
    stage=output.parent/(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    records=[]
    def copy(src,rel):
        dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);h=digest(src);shutil.copy2(src,dst)
        if digest(dst)!=h or digest(src)!=h:raise ValueError('Source changed while copying: '+str(src))
    for p in sorted(v1.rglob('*')):
        if p.is_file():copy(p,Path('new60_20261014')/p.relative_to(v1))
    for p,rel in source:copy(p,Path('new60_opt_v2')/rel)
    verify_frozen(root)
    write(stage/'packaging_omissions.json',dict(omitted=omitted,source_files_deleted=False,all_selected_models_and_dependencies_preserved=True))
    text=report(root,context)
    (stage/'两版本说明.md').write_text(text,encoding='utf-8');(stage/'两版本说明.txt').write_text(plain(text),encoding='utf-8');docx(stage/'两版本说明.docx',text)
    dep=context.get('website_deployment');link=f"\n在线双模式网站：[{dep['url']}]({dep['url']})\n" if dep and dep.get('verified') else ''
    readme='# 新60工况完整交付：V1 + 双模式优化V2\n\n先阅读 [两版本说明](两版本说明.md)。\n\n- `new60_20261014/`：原V1逐字节副本。\n- `new60_opt_v2/`：优化后预测版与全60训练拟合版。\n\n预测误差和训练内拟合误差分别标明；原测试已见，本轮为复用测试，非新盲测。\n'+link+'\n本地网站：`new60_opt_v2/website/index.html`（在new60_opt_v2内运行 `python code/serve_website_v2.py`）。纯NumPy入口：`new60_opt_v2/code/predict_v2.py --help`。\n\n逐文件SHA在 `delivery_manifest.json`；省略的可重建中间量在 `packaging_omissions.json`。\n'
    (stage/'README.md').write_text(readme,encoding='utf-8')
    for p in sorted(stage.rglob('*')):
        if p.is_file():records.append(dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    manifest=dict(schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),builder_sha256=digest(Path(__file__)),choices_sha256=context['choices_sha256'],v1=context['v1_integrity'],website_deployment=context.get('website_deployment'),file_count=len(records),bytes=sum(r['bytes'] for r in records),records=records,manifest_self_excluded=True,npz_values_loaded=False,model_selection_performed=False,tests_recomputed=False,published=False,published_scope='The packager itself never deploys; deployment record is independent')
    write(stage/'delivery_manifest.json',manifest);stage.rename(output)
    result=dict(output=str(output),files=len(records)+1,bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),manifest_sha256=digest(output/'delivery_manifest.json'))
    if make_zip:
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(output).as_posix())
        h=digest(archive);archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n',encoding='utf-8')
        result.update(zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=h,zip_entries=result['files'])
    return result

def build_zip_only(root=ROOT,v1=DEFAULT_V1,output=DEFAULT_OUTPUT,reviews=()):
    """Stream existing sources into a complete ZIP; keep only small docs beside it.

    The archive has a single named top directory. A regular extraction gives
    both sibling studies and their working relative paths. No source mutation,
    hard links, whole-tree temporary copy or whole-archive extraction is used.
    """
    root=Path(root).resolve();v1=Path(v1).resolve();output=Path(output).resolve();archive=output.with_suffix('.zip')
    if output.exists() or archive.exists():raise FileExistsError('Refuse existing delivery/archive')
    context=preflight(root,v1,reviews)
    if not context['ready']:raise ValueError('Delivery not ready: '+json.dumps(context,ensure_ascii=False))
    context['v1_integrity']=verify_v1(v1);source,omitted=source_plan(root)
    total=context['v1_integrity']['bytes']+sum(p.stat().st_size for p,_ in source)
    required=int(total*1.01)+512*1024*1024
    if shutil.disk_usage(output.parent).free<required:raise OSError('Insufficient space for streaming ZIP plus512MiB reserve; no files written')
    def progress(stage,**details):
        if total>100*1024*1024:print(json.dumps(dict(stage=stage,**details),ensure_ascii=False),flush=True)
    token=uuid.uuid4().hex[:8];stage=output.parent/(output.name+'.building-'+token);stage.mkdir(parents=True)
    tempzip=archive.parent/(archive.name+'.building-'+token)
    write(stage/'packaging_omissions.json',dict(omitted=omitted,source_files_deleted=False,all_selected_models_and_dependencies_preserved=True))
    text=report(root,context);(stage/'两版本说明.md').write_text(text,encoding='utf-8');(stage/'两版本说明.txt').write_text(plain(text),encoding='utf-8');docx(stage/'两版本说明.docx',text)
    dep=context.get('website_deployment');link=f"\n在线双模式网站：[{dep['url']}]({dep['url']})\n" if dep and dep.get('verified') else ''
    readme='# 新60工况完整交付：V1 + 双模式优化V2\n\n请先完整解压同名ZIP，再阅读 [两版本说明](两版本说明.md)。为节省磁盘，ZIP旁的此目录只保留说明和清单，完整资料在ZIP内。\n\n- `new60_20261014/`：原V1逐字节副本。\n- `new60_opt_v2/`：优化后预测版与全60训练拟合版。\n\n预测误差和训练内拟合误差分别标明；原测试已见，本轮为复用测试，非新盲测。\n'+link+'\n解压后的本地网站：`new60_opt_v2/website/index.html`（在new60_opt_v2内运行 `python code/serve_website_v2.py`）。纯NumPy入口：`new60_opt_v2/code/predict_v2.py --help`。\n\n逐文件SHA在 `delivery_manifest.json`；省略的可重建中间量在 `packaging_omissions.json`。\n'
    (stage/'README.md').write_text(readme,encoding='utf-8')
    mapping=[(p,Path('new60_20261014')/p.relative_to(v1)) for p in sorted(v1.rglob('*')) if p.is_file()]
    mapping += [(p,Path('new60_opt_v2')/rel) for p,rel in source]
    mapping += [(p,p.relative_to(stage)) for p in sorted(stage.rglob('*')) if p.is_file()]
    progress('hash_source_files',files=len(mapping),bytes=total)
    records=[dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p,rel in mapping]
    if len({r['path'] for r in records})!=len(records):raise ValueError('Duplicate archive path')
    manifest=dict(schema_version=2,delivery_mode='stream_zip_only',archive_root=output.name,created_utc=datetime.now(timezone.utc).isoformat(),builder_sha256=digest(Path(__file__)),choices_sha256=context['choices_sha256'],v1=context['v1_integrity'],website_deployment=dep,file_count=len(records),bytes=sum(r['bytes'] for r in records),records=records,manifest_self_excluded=True,npz_values_loaded=False,model_selection_performed=False,tests_recomputed=False,published=False,published_scope='Packager never deploys; deployment record is independent',local_directory_scope='Documents and manifest only; full sibling trees are inside ZIP')
    write(stage/'delivery_manifest.json',manifest)
    mpath=stage/'delivery_manifest.json';records_with_manifest=records+[dict(path='delivery_manifest.json',bytes=mpath.stat().st_size,sha256=digest(mpath))]
    mapping.append((mpath,Path('delivery_manifest.json')))
    expected={r['path']:r for r in records_with_manifest}
    verify_frozen(root)
    progress('write_streaming_zip',files=len(mapping))
    with zipfile.ZipFile(tempzip,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for index,(src,rel) in enumerate(mapping,1):
            already_compressed=src.suffix.lower() in {'.npz','.gz','.png','.jpg','.jpeg','.pdf','.docx'}
            z.write(src,output.name+'/'+rel.as_posix(),compress_type=zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED)
            if src.stat().st_size!=expected[rel.as_posix()]['bytes'] or digest(src)!=expected[rel.as_posix()]['sha256']:raise ValueError('Source changed during ZIP write: '+str(src))
            if index%200==0:progress('write_streaming_zip',completed=index,total=len(mapping))
    # Stream every decompressed member, not only ZipFile.testzip().
    checked_bytes=0
    progress('verify_all_zip_members',files=len(mapping))
    with zipfile.ZipFile(tempzip) as z:
        infos=z.infolist()
        if len(infos)!=len(expected):raise ValueError('ZIP entry count mismatch')
        for info in infos:
            rel=info.filename[len(output.name)+1:];row=expected[rel];h=hashlib.sha256();crc=0;size=0
            with z.open(info) as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk);crc=zlib.crc32(chunk,crc);size+=len(chunk)
            if size!=row['bytes'] or h.hexdigest()!=row['sha256'] or crc&0xffffffff!=info.CRC:raise ValueError('ZIP stream SHA/bytes/CRC mismatch: '+rel)
            checked_bytes+=size
    verify_frozen(root);tempzip.rename(archive);stage.rename(output)
    progress('hash_final_zip',bytes=archive.stat().st_size)
    h=digest(archive);archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n',encoding='utf-8')
    return dict(output_documents=str(output),delivery_mode='stream_zip_only',archive_root=output.name,files=len(expected),uncompressed_bytes=checked_bytes,zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=h,zip_entries=len(expected),stream_crc_sha_verified=True,manifest_sha256=digest(output/'delivery_manifest.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--v1',type=Path,default=DEFAULT_V1);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--preflight',action='store_true');p.add_argument('--zip',action='store_true');p.add_argument('--zip-only',action='store_true',help='Stream complete ZIP from sources; leave only small documents/manifest directory');p.add_argument('--review',type=Path,action='append',default=[])
    a=p.parse_args();result=preflight(a.root,a.v1,a.review) if a.preflight else build_zip_only(a.root,a.v1,a.output,a.review) if a.zip_only else build(a.root,a.v1,a.output,a.zip,a.review);print(json.dumps(result,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':main()
