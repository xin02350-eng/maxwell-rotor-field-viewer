"""Stream a selected, standalone V5 delivery from an explicit SHA-bound plan.

No training, prediction, test evaluation, NPZ decoding, publication or source
mutation. No V1/V2 full-tree copies. All archive paths preserve their paths
relative to work/, so selected sibling model dependencies remain resolvable.
The final source plan is prepared only after choices and scientific QA freeze.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import uuid
import zipfile
import zlib
from urllib.parse import urlsplit
from delivery_document_v5 import docx,plain
from comparison_narrative_v5 import experiment_count_text,comparison_text
from scope_utils_v5 import normalize_scope

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT.parents[1]/'outputs'/'沈夏鑫_新60第五轮优化独立交付_20261015'
FORBIDDEN={'.git','__pycache__','.venv','venv','site-packages','node_modules',
           'delivery_preview','portable_delivery_check','synthetic_fixture','toy_fixture'}
REQUIRED_ROLES={'merged_source','source_metadata','split','cache','frozen_choices',
                'inference_cli','source_code','training_log','evaluation','website','figure','qa'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def safe_relative(relative):
    if not isinstance(relative,str) or '\\' in relative or ':' in relative:
        raise ValueError('Only slash-separated relative paths are permitted')
    p=PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(x in {'..','.'} or x in FORBIDDEN for x in p.parts):
        raise ValueError('Unsafe/runtime archive path: '+relative)
    return p

def regular(work,relative):
    rel=safe_relative(relative);raw=work.joinpath(*rel.parts)
    for p in [raw,*raw.parents]:
        if p==work.parent:break
        if p.is_symlink():raise ValueError('Symlink in dependency: '+relative)
    path=raw.resolve()
    if not path.is_relative_to(work) or not path.is_file():raise FileNotFoundError(relative)
    return path

def study_reference(root,relative):
    """Choice paths may point into selected siblings, never outside work/."""
    path=(root/relative).resolve()
    if not path.is_relative_to(root.parent):raise ValueError('Study reference outside work/')
    return path.relative_to(root.parent).as_posix()

def passed(report):
    return any(report.get(k) is True for k in ['all_checks_pass','all_pass','pass','passed','verified']) or str(report.get('status','')).upper()=='PASS'

def verify_choices(root):
    path=root/'both_versions_choices_frozen.json';j=read(path);h=digest(path)
    if h not in (root/'both_versions_choices_frozen.sha256').read_text():raise ValueError('Global frozen SHA mismatch')
    if j.get('previous_test_already_seen') is not True or j.get('full60_fitting_scores_used_in_prediction_choice') is not False:
        raise ValueError('Reused-test and fitting-isolation fields must be explicit')
    if len({m['id'] for m in j['models']})!=len(j['models']):raise ValueError('Duplicate frozen model ID')
    for m in j['models']:
        for pk,hk in [('model_relative_path','model_sha256'),('inference_module_relative_path','inference_module_sha256'),('model_source_module_relative_path','model_source_module_sha256')]:
            if pk in m and digest(regular(root.parent,study_reference(root,m[pk])))!=m[hk]:raise ValueError('Frozen source changed: '+m[pk])
        for ref in m.get('provenance_files',[]):
            if digest(regular(root.parent,study_reference(root,ref['relative_path'])))!=ref['sha256']:
                raise ValueError('Frozen provenance changed: '+ref['relative_path'])
    for ref in j.get('support_files',[]):
        if digest(regular(root.parent,study_reference(root,ref['relative_path'])))!=ref['sha256']:
            raise ValueError('Frozen support source changed: '+ref['relative_path'])
    if digest(regular(root.parent,study_reference(root,j['split_relative_path'])))!=j['split_sha256']:
        raise ValueError('Split SHA mismatch')
    split=read(root/j['split_relative_path'])
    for m in j['models']:normalize_scope(m,split)
    return j,h

def source_plan(root,plan,choices,choices_sha):
    if plan.get('schema_version')!=1 or plan.get('choices_sha256')!=choices_sha or plan.get('dependencies_complete') is not True:
        raise ValueError('A complete V5 SHA-bound selected-dependency plan is required')
    work=root.parent;mapping=[];seen=set();roles=set()
    for row in plan['files']:
        rel=safe_relative(row['source_relative_path']).as_posix()
        if rel in seen:raise ValueError('Duplicate planned file: '+rel)
        if rel==root.name+'/delivery_dependencies.json':raise ValueError('The plan must not self-hash; it is added by the builder')
        src=regular(work,rel);size=src.stat().st_size
        if size!=row['bytes'] or digest(src)!=row['sha256']:raise ValueError('Planned source changed: '+rel)
        seen.add(rel);roles.add(row['role']);mapping.append((src,rel,row))
    if missing:=REQUIRED_ROLES-roles:raise ValueError('Required delivery roles absent: '+','.join(sorted(missing)))
    rootrel=root.relative_to(work).as_posix()
    required={rootrel+'/'+p for p in ['protocol.json','prediction_choices_frozen.json','both_versions_choices_frozen.json',
              'both_versions_choices_frozen.sha256','final_summary.json','website/manifest.json','website/results.json',
              'website/display_precision.json','code/predict_v5.py','code/serve_website_v5.py','code/requirements-inference.txt']}
    required.add(study_reference(root,choices['split_relative_path']))
    required.update(study_reference(root,r['relative_path']) for r in choices.get('support_files',[]))
    for m in read(root/'website/results.json')['models']:
        for ref in m['sources']:
            required.add(ref['path_relative_to_work'])
            required.add(ref['scientific_source_relative_root']+'/both_versions_choices_frozen.json')
    dependencies=plan.get('model_dependencies',{})
    for m in choices['models']:
        if m['id'] not in dependencies or not dependencies[m['id']]:raise ValueError('Explicit dependency closure missing: '+m['id'])
        required.update(dependencies[m['id']])
        for key in ['model_relative_path','inference_module_relative_path','model_source_module_relative_path']:
            if key in m:required.add(study_reference(root,m[key]))
        for ref in m.get('provenance_files',[]):required.add(study_reference(root,ref['relative_path']))
    if absent:=required-seen:raise ValueError('Needed model/code/data/provenance omitted: '+','.join(sorted(absent)))
    for row in plan.get('omitted',[]):
        if not row.get('reason') or not row.get('rebuild_source_relative_paths'):
            raise ValueError('Every omission needs a reason and retained reconstruction source paths')
        if row['source_relative_path'] in seen:raise ValueError('File both omitted and retained')
        for p in row['rebuild_source_relative_paths']:
            if p not in seen:raise ValueError('Omission rebuild source not packaged: '+p)
    return mapping

def deployment(root):
    p=root/'website/deployment.json'
    if not p.is_file():return None
    j=read(p)
    if j.get('verified') is not True:return {'verified':False}
    url=j.get('url','');u=urlsplit(url)
    if u.scheme!='https' or not u.netloc or u.username or u.password or any(c in url for c in '\r\n <>[]()'):
        raise ValueError('Invalid verified deployment URL')
    return {'verified':True,'url':url,'sha256':digest(p),'commit':j.get('commit')}

def preflight(root=ROOT,plan_path=None,reviews=()):
    root=Path(root).resolve();plan_path=Path(plan_path or root/'delivery_dependencies.json').resolve()
    initial=['both_versions_choices_frozen.json','both_versions_choices_frozen.sha256',
             'final_summary.json','website/manifest.json','website/results.json','website/display_precision.json']
    missing=[p for p in initial if not (root/p).is_file()]
    if not plan_path.is_file():missing.append(str(plan_path))
    if missing:return {'ready':False,'missing':missing,'npz_values_loaded':False}
    choices,ch=verify_choices(root);plan=read(plan_path);mapping=source_plan(root,plan,choices,ch)
    review_paths=list(plan.get('required_reviews',[]))+list(map(str,reviews))
    if not review_paths:raise ValueError('Formal delivery requires independent final QA gates')
    for rel in review_paths:
        path=root/rel;j=read(path)
        if not passed(j):raise ValueError('QA gate not PASS: '+rel)
        if j.get('choices_sha256') and j['choices_sha256']!=ch:raise ValueError('QA belongs to a different freeze: '+rel)
        if study_reference(root,rel) not in {r for _,r,_ in mapping}:raise ValueError('Required QA must be packaged: '+rel)
    for name in ['final_summary.json','website/manifest.json','website/results.json']:
        if read(root/name).get('choices_sha256')!=ch:raise ValueError('Artifact freeze mismatch: '+name)
    manifest=read(root/'website/manifest.json')
    if manifest.get('schema_version')!=3:raise ValueError('Website schema must be V5 with independent residuals')
    packed={rel for _,rel,_ in mapping}
    for name in ['manifest.json','results.json','display_precision.json','app.js','styles.css','index.html','guide.html','favicon.svg']:
        if f'{root.name}/website/{name}' not in packed:raise ValueError('Website dependency omitted: '+name)
    precision=read(root/'website/display_precision.json')
    for p in precision['payloads']:
        rel=f"{root.name}/website/{p['path']}"
        if rel not in packed or digest(root/'website'/p['path'])!=p['sha256']:raise ValueError('Payload missing/changed: '+rel)
    for mode in manifest['modes'].values():
        for region in mode['regions'].values():
            if f"{root.name}/website/{region['geometry']}" not in packed:raise ValueError('Geometry omitted')
    # The final plot recipe is optional during scaffold preparation, mandatory
    # in a formal plan via role=figure. Its scientific file references are bound.
    recipe_path=root/'figures/figure_recipe.json'
    if recipe_path.is_file():
        recipe=read(recipe_path)
        if recipe.get('choices_sha256')!=ch:raise ValueError('Figure recipe freeze mismatch')
        for ref in recipe.get('scientific_inputs',[]):
            rel=study_reference(root,'figures/'+ref['path'])
            if rel not in packed or digest(regular(root.parent,rel))!=ref['sha256']:raise ValueError('Figure scientific source omitted/changed')
        for fig in recipe.get('figures',[]):
            rel=study_reference(root,'figures/'+fig['input_csv'])
            if rel not in packed or digest(regular(root.parent,rel))!=fig['input_sha256']:raise ValueError('Figure CSV omitted/changed')
            for ext in ['.png','.pdf','.svg']:
                if f"{root.name}/figures/{fig['name']}{ext}" not in packed:raise ValueError('Figure export missing')
    return dict(ready=True,choices_sha256=ch,models=len(choices['models']),files=len(mapping),
                bytes=sum(p.stat().st_size for p,_,_ in mapping),plan_sha256=digest(plan_path),
                website_deployment=deployment(root),reviews=review_paths,npz_values_loaded=False,
                model_selection_performed=False,tests_recomputed=False,published=False)

def report(root,context,plan):
    results=read(root/'website/results.json');rows=[]
    for m in results['models']:
        part='training_fit' if m['branch']=='fitting_full60' else 'reused_test'
        quantity='Br' if m['quantity']=='Br' else 'magnitude';v=m['metrics'][part][quantity]
        scope='全60训练内拟合' if part=='training_fit' else '9工况复用测试'
        cells=[m.get('label',m['id']),m['training_scope']['label']+'；'+scope,m['quantity'],f"{v['relative_L2_pct']:.10g}",f"{v['rmse_T']:.10g}",f"{v['max_abs_T']:.10g}"]
        rows.append('| '+' | '.join(str(c).replace('|','\\|') for c in cells)+' |')
    dep=context.get('website_deployment');site=f"[V5 在线云图]({dep['url']})" if dep and dep.get('verified') else '网站已随包保留；尚无经核实的公开部署记录。'
    old=plan.get('previous_delivery',{});previous=f"旧完整包 `{old.get('filename','V1/V2 完整交付包')}`；SHA256 `{old.get('sha256','见旧包校验文件')}`。本包不重复复制旧版全部文件。"
    text=f'''# 新60工况 V5：预测优化与训练内拟合

本包提供冻结后的预测模型与全60已知工况的拟合模型。模型、来源代码、选定模型依赖、原数据及必要缓存、日志、原精度评价、科学图表和网站一起保留，可在完整解压后独立运行。

{site}

## 误差口径与总体结果

| 冻结模型 | 评估范围 | 量 | 相对 L2 (%) | RMSE (T) | 最大绝对误差 (T) |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+f'''

{comparison_text(root)}

## 实际优化工作量

{experiment_count_text(root)}

42训练版使用原42工况训练、9验证工况选型；51训练版在既定配方下合并原42训练与原9验证工况，原验证9因此属于训练内数据，不能再称未见验证。两版的原9测试均已在前轮查看，本轮为再次复用，非新盲测。网页42版只展示验证与复用测试科学分区，51版只展示复用测试；训练误差保留CSV/说明，不为训练分区重复保存巨型数组。

模型配方、实际训练caseIDs、选型时序与每个科学数组来源以V5冻结选择及独立检查为准。继承V2/V3/V4科学数组必须同时绑定原模型/原冻结SHA和新的V5显式复用检查；不会把旧PASS直接当作V5复评，也不复制未变化的大数组。网页与打包不改变冻结配方。

拟合版继承V3已完成的全60线性头校准：固定已训练NN隐层，用全60样本校准最后线性层。它不计为V5新增训练；接近机器精度的训练内残差不证明新工况预测准确，也不表示物理真值无误差。各版本的整体RMSE、相对L2和局部峰值同时保留，不混淆42、51与60工况的误差口径。

## 云图与原精度残差

转子和定子展示二维范数√(Br²+Bt²)，未包含Bz；气隙是有符号Br。真实45°采样扇区不镜像补圆、不跨孔插值。3D高度是磁密T，横纵坐标是mm，不是实体三维磁场。转子世界坐标按绝对机械角旋转，定子和气隙坐标固定；Br/Bt极坐标分量不额外作笛卡尔旋转。

真值和预测图采用独立q16显示副本，误差图从原精度残差单独量化，避免量化后相减制造假误差。CSV的误差列因而不必等于两列显示磁密值之差；原精度指标保存在evaluation、website/results及QA中。网页帧1至60对应科学frame_index0至59。仅验证已给出的60位置和采样点，未证明新位置或输入范围外的外推。

## 使用与复现

完整解压ZIP后进入new60_opt_v5，运行 `python code/serve_website_v5.py`，浏览器打开终端给出的localhost网址。运行 `python code/predict_v5.py --help` 查看NumPy推理（运行依赖NumPy及threadpoolctl）。必要的旧模型、源码与原数据分别位于同级精简new60_opt_v2、new60_opt_v3、new60_opt_v4及new60_20261014目录；请完整保留这些目录的相对结构，不单独搬动模型文件。

当前冻结推理不需要恢复旧V2拟合基模型。若要重跑旧原始模型检查，可先在交付根目录执行 `python new60_opt_v4/code/restore_original_fit_bases.py --verify-only`，用保留的V3模型和小型原始头补丁逐字节校验。无参数运行才恢复缺失的原base文件；已有文件SHA不符时拒绝覆盖。恢复证据保存在new60_opt_v5/qa/fitbase_restore_check.json。

搬迁检查使用独立路径的NTFS硬链接，核对原路径与搬迁路径的冻结推理、坐标、训练范围和每个源文件SHA，并阻止未声明项目路径和标签读取。它是本机不同路径的可移植证据；最终ZIP另做逐成员SHA、CRC和字节数核验，不冒充另一台机器的安装测试。

推理依赖安装：`python -m pip install -r code/requirements-inference.txt`。例如导出冻结转子预测模型原case02、第0帧、世界坐标：`python code/predict_v5.py --branch prediction --region rotor --case-id 2 --frame 0 --coordinates world --output rotor_case02_frame00.csv`。输出文件已存在时程序拒绝覆盖；请使用新的文件名。训练环境另见code/requirements-training.txt与来源环境记录。

{previous}

本包通过白名单流式打包，不训练、不推理、不重新选模型、不发布。省略的大型可重建中间数组逐路径记录理由、大小、SHA及保留的重建代码，未删除任何原工作文件。必要模型、依赖和来源日志保留；未把旧完整包再次复制。delivery_manifest.json记录逐文件SHA且不循环哈希自身，ZIP另附SHA。冻结选择SHA：`{context['choices_sha256']}`。
'''
    material=root/'figures/V5科学说明.md'
    if material.is_file():text+='\n\n## 科学图表与数值分析\n\n'+material.read_text(encoding='utf-8-sig')
    return text

def verify_archive(path,archive_root,records):
    expected={r['path']:r for r in records};total=0
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(expected) or len(set(names))!=len(names):raise ValueError('ZIP member inventory mismatch')
        for info in z.infolist():
            if not info.filename.startswith(archive_root+'/'):raise ValueError('ZIP root mismatch')
            rel=info.filename[len(archive_root)+1:];r=expected[rel];h=hashlib.sha256();crc=0;size=0
            with z.open(info) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);crc=zlib.crc32(b,crc);size+=len(b)
            if h.hexdigest()!=r['sha256'] or size!=r['bytes'] or crc&0xffffffff!=info.CRC:
                raise ValueError('ZIP SHA/bytes/CRC mismatch: '+rel)
            total+=size
    return {'members':len(expected),'uncompressed_bytes':total,'all_member_sha_crc_pass':True}

def build(root=ROOT,plan_path=None,output=DEFAULT_OUTPUT,reviews=()):
    root=Path(root).resolve();output=Path(output).resolve();archive=output.with_suffix('.zip')
    if output.exists() or archive.exists():raise FileExistsError('Existing delivery is immutable; choose a new output')
    plan_path=Path(plan_path or root/'delivery_dependencies.json').resolve();context=preflight(root,plan_path,reviews)
    if not context['ready']:raise ValueError('Not ready: '+json.dumps(context,ensure_ascii=False))
    plan=read(plan_path);choices,ch=verify_choices(root);mapping=source_plan(root,plan,choices,ch)
    if shutil.disk_usage(output.parent).free<int(context['bytes']*1.01)+256*1024*1024:
        raise OSError('Insufficient free space for one streaming ZIP plus256MiB reserve')
    token=uuid.uuid4().hex[:8];stage=output.parent/(output.name+'.building-'+token);stage.mkdir(parents=True)
    temporary=archive.parent/(archive.name+'.building-'+token)
    text=report(root,context,plan)
    (stage/'V5实验与使用说明.md').write_text(text,encoding='utf-8');(stage/'V5实验与使用说明.txt').write_text(plain(text),encoding='utf-8')
    docx(stage/'V5实验与使用说明.docx',text)
    write(stage/'packaging_omissions.json',{'omitted':plan.get('omitted',[]),'source_files_deleted':False})
    readme='# 新60 V5 独立交付\n\n请完整解压同名ZIP，阅读[V5说明](V5实验与使用说明.md)。ZIP旁目录只有说明与清单，完整依赖在ZIP内。\n\n进入new60_opt_v5后：\n\n1. 安装最小推理依赖 NumPy + threadpoolctl：`python -m pip install -r code/requirements-inference.txt`。\n2. 本地云图：`python code/serve_website_v5.py`，打开终端中的localhost网址。\n3. 导出原case02转子预测第0帧：`python code/predict_v5.py --branch prediction --region rotor --case-id 2 --frame 0 --coordinates world --output rotor_case02_frame00.csv`。\n\n保留各同级文件夹的相对结构。原42训练版保留原验证9组；最终51训练版已把这9组纳入训练，两版均使用原复用测试9组。全部60校准为训练内拟合，不能与预测误差混比。网页只列每个模型指定科学分区的工况，完整60工况仍可通过CLI推理。\n'
    dep=context.get('website_deployment')
    if dep and dep.get('verified'):readme+=f"\n[已核实的 V5 在线云图]({dep['url']})\n"
    (stage/'README.md').write_text(readme,encoding='utf-8')
    # Include the exact dependency inventory even if its own SHA cannot be in itself.
    mapping.append((plan_path,root.name+'/delivery_dependencies.json',{'role':'delivery_plan'}))
    mapping.extend((p,p.relative_to(stage).as_posix(),{'role':'generated_document'}) for p in stage.rglob('*') if p.is_file())
    records=[dict(path=rel,bytes=p.stat().st_size,sha256=digest(p),role=row['role']) for p,rel,row in mapping]
    if len({r['path'] for r in records})!=len(records):raise ValueError('Duplicate final archive path')
    manifest=dict(schema_version=3,delivery_mode='selected_standalone_stream_zip',archive_root=output.name,
                  created_utc=datetime.now(timezone.utc).isoformat(),builder_sha256=digest(Path(__file__)),
                  choices_sha256=ch,source_plan_sha256=digest(plan_path),records=records,file_count=len(records),
                  bytes=sum(r['bytes'] for r in records),manifest_self_excluded=True,published=False,
                  website_deployment=context.get('website_deployment'),npz_values_loaded=False,
                  model_selection_performed=False,tests_recomputed=False,previous_delivery=plan.get('previous_delivery'))
    write(stage/'delivery_manifest.json',manifest);mp=stage/'delivery_manifest.json'
    records.append(dict(path='delivery_manifest.json',bytes=mp.stat().st_size,sha256=digest(mp),role='manifest'))
    mapping.append((mp,'delivery_manifest.json',{'role':'manifest'}));expected={r['path']:r for r in records}
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for src,rel,_ in mapping:
            compressed=src.suffix.lower() in {'.npz','.gz','.png','.jpg','.jpeg','.pdf','.docx'}
            z.write(src,output.name+'/'+rel,compress_type=zipfile.ZIP_STORED if compressed else zipfile.ZIP_DEFLATED)
            if digest(src)!=expected[rel]['sha256'] or src.stat().st_size!=expected[rel]['bytes']:raise ValueError('Source changed during streaming: '+rel)
    result=verify_archive(temporary,output.name,records);verify_choices(root)
    temporary.rename(archive);stage.rename(output);h=digest(archive)
    archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n',encoding='utf-8')
    return dict(result,zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=h,
                output_documents=str(output),manifest_sha256=digest(output/'delivery_manifest.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--review',action='append',default=[]);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--preflight',action='store_true');m.add_argument('--zip-only',action='store_true')
    a=p.parse_args();result=preflight(a.root,a.plan,a.review) if a.preflight else build(a.root,a.plan,a.output,a.review)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
