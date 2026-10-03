"""Build a GitHub Pages site with independently loaded magnetic-field cases.

Usage: python outputs/build_website.py
Inputs: work/rotor_dataset.npz, work/predictions.npz, rotor_surface_mesh.npz.
Runtime source and style additions are website_runtime.js and website_styles.css.
Each gzip pack contains little-endian uint16 truth[300,2500] followed by
prediction[300,2500], both C-contiguous, without a header. Decode with
B_T = quantized_value * metadata.step + metadata.offset.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np


def build(dataset: Path, predictions: Path, mesh: Path, output: Path) -> None:
    source_dir = Path(__file__).resolve().parent
    with np.load(dataset) as d:
        truth = d['Bmag'].astype(np.float32)
        xy, point_ids = d['xy'], d['point_id']
        conditions, theta, absolute = d['conditions'], d['theta'], d['theta_absolute']
    with np.load(predictions) as d:
        pred, split = d['angle_pred'].astype(np.float32), d['split'].astype(np.int8)
    with np.load(mesh) as d:
        if not np.array_equal(d['xy'], xy):
            raise ValueError('Mesh vertex ordering does not match dataset.xy')
        triangles = d['triangles'].astype(np.int32)
    if truth.shape != pred.shape or truth.shape != (12, 300, 2500):
        raise ValueError('Expected both fields to have shape [12,300,2500]')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.min() < 0 or triangles.max() >= len(xy):
        raise ValueError('Invalid mesh triangles')
    if not np.isfinite(truth).all() or not np.isfinite(pred).all() or not np.isin(split, [0, 1, 2]).all():
        raise ValueError('Invalid input field or split')
    error = pred.astype(np.float64) - truth
    squared, absolute_error = error**2, np.abs(error)
    truth_squared = truth.astype(np.float64)**2
    metrics = np.stack([np.sqrt(squared.mean(-1)), absolute_error.mean(-1),
                        np.sqrt(squared.sum(-1)/np.maximum(truth_squared.sum(-1), 1e-30)),
                        absolute_error.max(-1)], axis=-1)
    low, high = float(min(0, truth.min(), pred.min())), float(max(truth.max(), pred.max()))
    step = max((high-low)/65535.0, 1e-12)
    summaries = []
    for label in range(3):
        selected = split == label
        summaries.append({'frames':int(selected.sum()*truth.shape[0]),
                          'rmse':float(np.sqrt(squared[:,selected].mean())),
                          'mae':float(absolute_error[:,selected].mean()),
                          'relative':float(np.sqrt(squared[:,selected].sum()/truth_squared[:,selected].sum()))})
    assets, data = output/'assets', output/'data'
    assets.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    case_files = []
    for index in range(12):
        # Float64 arithmetic makes the declared half-step rounding bound exact.
        values = np.stack([truth[index], pred[index]]).astype(np.float64)
        packed = np.rint((values-low)/step).clip(0,65535).astype('<u2').tobytes(order='C')
        compressed = gzip.compress(packed, compresslevel=6, mtime=0)
        name = f'case_{index+1:02d}.bin.gz'
        (data/name).write_bytes(compressed)
        case_files.append({'url':'data/'+name,'compressedBytes':len(compressed),'uncompressedBytes':len(packed),
                           'sha256':hashlib.sha256(compressed).hexdigest()})
    metadata = {'formatVersion':1,'shape':list(truth.shape),'conditions':conditions.tolist(),
                'theta':theta.tolist(),'thetaAbsolute':absolute.tolist(),'xy':xy.tolist(),
                'pointId':point_ids.tolist(),'split':split.tolist(),'metrics':metrics.tolist(),
                'summaries':summaries,'offset':low,'step':step,'displayErrorBound':step/2,
                'maxField':float(np.ceil(high*10)/10),'maxError':float(absolute_error.max()),
                'triangles':triangles.tolist(),'caseFiles':case_files,'defaultCondition':4,'defaultFrame':44,
                'packFormat':{'compression':'gzip','dtype':'uint16','byteOrder':'little-endian',
                              'shape':[2,300,2500],'axis0':['truth','prediction'],'order':'C',
                              'decode':'B_T = uint16_value * step + offset'}}
    (data/'metadata.json').write_text(json.dumps(metadata, separators=(',',':')), encoding='utf-8')
    template = (source_dir/'viewer_2d3d_template.html').read_text(encoding='utf-8')
    css = re.search(r'<style>([\s\S]*?)</style>',template).group(1)
    css += '\n' + (source_dir/'website_styles.css').read_text(encoding='utf-8')
    javascript = template.rsplit('<script>',1)[1].split('</script>',1)[0]
    javascript = javascript.replace("const M=JSON.parse(document.getElementById('dataMeta').textContent), $=id=>document.getElementById(id);", "let M; const $=id=>document.getElementById(id);")
    javascript = javascript.replace("const N=M.shape[2], F=M.shape[1], labels=", "let N=2500,F=300; const labels=")
    javascript = javascript.replace('(condition*F+frame)*N', 'frame*N')
    javascript = re.sub(r'async function unpack\(id\)\{.*?\}\n', '', javascript)
    javascript = re.sub(r'async function init\(\)\{.*?\nconst projectedPointCache=', 'const projectedPointCache=', javascript)
    javascript = javascript.replace('setupSceneControls();\ninit();', '')
    javascript = javascript.replace("function step(direction){", "function step(direction){if(isLoading||!TRUTH)return;")
    start, end = javascript.index('function setPlaying('), javascript.index('function download(')
    javascript = javascript[:start] + '''let animationFrameId=null,animationGeneration=0;
function setPlaying(value){animationGeneration++;if(animationFrameId!==null)cancelAnimationFrame(animationFrameId);animationFrameId=null;playing=Boolean(value&&!isLoading&&TRUTH);$('play').textContent=playing?'Ⅱ 暂停':'▶ 播放';if(playing){lastTick=performance.now();const token=animationGeneration;animationFrameId=requestAnimationFrame(now=>tick(now,token));}}
function tick(now,token){if(!playing||token!==animationGeneration)return;if(now-lastTick>=1000/Number($('speed').value)){step(1);lastTick=now;}animationFrameId=requestAnimationFrame(next=>tick(next,token));}
''' + javascript[end:]
    javascript = javascript.replace("function exportCSV(){", "function exportCSV(){if(isLoading||!TRUTH)return;")
    javascript = javascript.replace("function exportPNG(){", "function exportPNG(){if(isLoading||!TRUTH)return;const exportName=filename('png');")
    javascript = javascript.replace("out.toBlob(blob=>download(blob,filename('png')),'image/png');", "out.toBlob(blob=>{if(blob)download(blob,exportName);},'image/png');")
    javascript = javascript.replace('局部三角面', '局部三角面着色')
    javascript += '\n' + (source_dir/'website_runtime.js').read_text(encoding='utf-8')
    (assets/'viewer.js').write_text(javascript, encoding='utf-8')
    (assets/'viewer.css').write_text(css, encoding='utf-8')
    html = template[:template.rfind('<script>')] + '<script src="assets/viewer.js" defer></script>\n</body></html>\n'
    html = re.sub(r'<script id="(?:dataMeta|truthData|predData)"[^>]*>[\s\S]*?</script>\n?', '', html)
    html = re.sub(r'<style>[\s\S]*?</style>', '<link rel="stylesheet" href="assets/viewer.css">\n<link rel="icon" type="image/svg+xml" href="favicon.svg">', html)
    html = re.sub(r'<div class="splash" id="splash">[\s\S]*?</div></div>\n', '', html, count=1)
    html = html.replace('<title>转子磁密 · 2D / 3D 交互可视化</title>', '<title>Maxwell 转子磁密 · 2D / 3D 神经网络可视化</title>\n<meta name="description" content="12组工况、300个转子位置的磁密云图与神经网络拟合。交互切换2D云图、3D磁密高度图和误差分析。">')
    html = html.replace('<div class="topbadge">OFFLINE ANALYSIS<br>│B│ · TESLA<br>ROTOR CORE ONLY</div>', '<div class="headeraside"><nav class="sitelinks" aria-label="网站导航"><a href="guide.html">使用说明</a><a href="downloads/rotor_production.npz" download>下载模型</a><a href="https://github.com/xin02350-eng/maxwell-rotor-field-viewer">GitHub ↗</a></nav><div class="topbadge">│B│ · TESLA<br>ROTOR CORE ONLY</div></div>')
    load_panel = '''<section class="loadpanel" id="caseLoadPanel" aria-live="polite"><div class="loadcopy"><div class="loader" id="loader"></div><h2 id="caseLoadHeading">正在准备磁密云图</h2><p id="caseLoadText">正在读取工况信息…</p><button id="retryLoad" hidden>重新加载</button><p class="loadfoot">首次仅加载所选工况，切换时按需读取，其余工况不占首屏流量。</p></div><figure><img src="assets/loading-preview.png" width="1800" height="890" alt="Maxwell真值、神经网络预测及误差的示意预览"><figcaption>静态示意预览 · 数据就绪后显示所选工况的交互结果</figcaption></figure></section>\n'''
    html = html.replace('<div class="frameheading">', load_panel+'<div class="frameheading">')
    html = html.replace('<article class="scope">', '<article class="scope" id="help">')
    html = html.replace('局部三角面', '局部三角面着色')
    html = html.replace('现代 Edge / Chrome / Firefox 离线打开', '按工况加载 · Chrome / Edge / Firefox')
    html = html.replace('<main class="wrap main">', '<main class="wrap main" aria-busy="true">')
    html = html.replace('</header>', '</header><noscript><p class="noscript">此交互页面需要启用 JavaScript。可通过顶部「使用说明」查看静态磁密图和报告。</p></noscript>')
    (output/'index.html').write_text(html,encoding='utf-8')
    shutil.copyfile(source_dir/'viewer_export_example.png',assets/'loading-preview.png')
    (output/'favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="15" fill="#112c39"/><path d="M12 43A20 20 0 0 1 52 43H42A10 10 0 0 0 22 43Z" fill="#6bd0c7"/><circle cx="21" cy="28" r="3" fill="#112c39"/><circle cx="43" cy="28" r="3" fill="#112c39"/></svg>',encoding='utf-8')
    (output/'.nojekyll').write_text('',encoding='utf-8')
    manifest = {'formatVersion':1,'cases':12,'framesPerCase':300,'pointsPerFrame':2500,
                'defaultConditionZeroBased':4,'defaultFrameZeroBased':44,
                'metadataBytes':(data/'metadata.json').stat().st_size,
                'initialCaseCompressedBytes':case_files[4]['compressedBytes'],
                'allCasesCompressedBytes':sum(c['compressedBytes'] for c in case_files),
                'indexBytes':(output/'index.html').stat().st_size,
                'packFormat':metadata['packFormat']}
    (output/'build-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=root/'work/rotor_dataset.npz')
    parser.add_argument('--predictions',type=Path,default=root/'work/predictions.npz')
    parser.add_argument('--mesh',type=Path,default=root/'outputs/rotor_surface_mesh.npz')
    parser.add_argument('--output',type=Path,default=root/'outputs/website')
    a=parser.parse_args()
    build(a.dataset,a.predictions,a.mesh,a.output)
