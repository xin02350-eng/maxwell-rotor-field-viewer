"""Copy reproducible public source and audit website contents; never publish data/weights."""
import json,hashlib,re,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent.parent
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 web=ROOT/'website';dst=web/'research';dst.mkdir(exist_ok=True)
 guide=web/'guide.html'
 guide.write_text(guide.read_text(encoding='utf8').replace('https://github.com/xin02350-eng/maxwell-rotor-field-viewer/blob/main/research/README.md','https://github.com/xin02350-eng/maxwell-rotor-field-viewer/blob/main/new60-opt-v2/research/README.md'),encoding='utf8')
 selected=['protocol.json','both_versions_choices_frozen.json','both_versions_choices_frozen.sha256','prediction_choices_frozen.json','prediction_choices_frozen.sha256','final_summary.json','final_summary.csv','freeze_both_versions.py','make_choices_v2.py','neural_frame_kernel.py','adaptive_spectral.py','power_magnitude.py','baseline_validation.py','code/inference_numpy_v2.py','code/predict_v2.py','code/PREDICT_V2.md','code/requirements-inference.txt','code/requirements-training.txt']
 selected.append('code/serve_website_v2.py')
 selected += [p.relative_to(ROOT).as_posix() for group in ['neural_route','kernel_route'] for p in (ROOT/group).glob('*.py')]
 records=[]
 for rel in selected:
  src=ROOT/rel;content=src.read_text(encoding='utf8')
  assert not re.search(r'xwechat|wxid_|(?<![a-z])[A-Z]:[\\/]|Bearer\s+[\w.-]+',content,re.I),rel
  target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
  records.append(dict(path=rel,bytes=target.stat().st_size,sha256=digest(target)))
 (dst/'README.md').write_text('''# 新60工况双版本优化源码

这里提供冻结选择、指标、纯NumPy推理及优化训练源码。全部模型、原始数据、缓存、最终float64评价数组和图，位于用户的完整交付ZIP；大数组与权重未重复发布到网站。

请保留完整包内并列的new60_20261014和new60_opt_v2目录，在new60_opt_v2运行code/predict_v2.py。网页research是供阅读和版本追溯的源码副本，单独下载它并不包含运行所需的数据和权重。详细命令见code/PREDICT_V2.md。

全部60工况拟合版是纯神经网络，误差是训练内拟合误差。预测分支按42训练/9验证/9复用测试划分；纯神经网络与解析核对照分别保留。主转子磁密大小云图采用验证集选定的解析核模型，同时提供纯神经网络对照。不同方法均明确标记。

source_manifest.json记录本源码副本的字节SHA256。大规模候选比较主要使用验证集，复用旧测试集不构成新的盲测。3D可视化是磁密高度图。
''',encoding='utf8')
 (dst/'source_manifest.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),files=records),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 files=list(p for p in web.rglob('*') if p.is_file())
 assert max(p.stat().st_size for p in files)<100*1024*1024
 assert sum(p.stat().st_size for p in files)<1024**3
 assert not any(p.suffix.lower() in ['.npz','.pt','.pth','.zip','.docx'] for p in files)
 for p in files:
  if p.suffix.lower() in ['.html','.js','.json','.py','.md','.txt','.css','.svg']:
   t=p.read_text(encoding='utf8')
   assert not re.search(r'xwechat|wxid_|(?<![a-z])[A-Z]:[\\/]|Bearer\s+[\w.-]+',t,re.I),p
 result=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),website_file_count=len(files),website_bytes=sum(p.stat().st_size for p in files),max_file_bytes=max(p.stat().st_size for p in files),public_source_files=len(records),checks=['No source raw arrays or weights published','No WeChat paths or drive absolute paths in public text','No individual file above100MiB','New public site below1GiB','Public source files are byte-identical copies'],source_manifest_sha256=digest(dst/'source_manifest.json'),model_freeze_sha256=digest(ROOT/'both_versions_choices_frozen.json'))
 (ROOT/'qa/public_site_preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
