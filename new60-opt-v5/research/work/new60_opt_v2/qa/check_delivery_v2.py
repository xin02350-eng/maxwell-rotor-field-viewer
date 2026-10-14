"""Independent read-only ZIP/manifest audit; no whole-tree extraction or inference."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def audit(archive,docs,output):
    archive=Path(archive).resolve();docs=Path(docs).resolve();output=Path(output).resolve();checks=[]
    def check(name,condition,details=None):
        checks.append(dict(name=name,pass_=bool(condition),details=details))
        if not condition:raise AssertionError(name+': '+str(details))
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=[i.filename for i in infos];prefix=docs.name+'/'
        check('unique ZIP members',len(set(names))==len(names))
        check('all ZIP members within one named root',all(n.startswith(prefix) and '..' not in PurePosixPath(n).parts and not PurePosixPath(n).is_absolute() for n in names))
        raw=z.read(prefix+'delivery_manifest.json');manifest=json.loads(raw);records=manifest['records']
        expected={r['path']:r for r in records}
        check('unique manifest file paths',len(expected)==len(records))
        expected['delivery_manifest.json']=dict(path='delivery_manifest.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        check('ZIP exactly matches full file inventory',{n[len(prefix):] for n in names}==set(expected))
        check('local document manifest same as ZIP',(docs/'delivery_manifest.json').read_bytes()==raw)
        forbidden={'__pycache__','.venv','venv','node_modules','.git','site-packages','portable_delivery_check','delivery_preview'}
        check('no runtime or synthetic fixture directories',not any(any(p in forbidden or p.lower().startswith(('synthetic_fixture','toy_fixture','failedrun','failed_run')) for p in PurePosixPath(n).parts[:-1]) for n in names))
        totals=0;zip_rows=[]
        for info in infos:
            rel=info.filename[len(prefix):];row=expected[rel];h=hashlib.sha256();crc=0;size=0
            with z.open(info) as stream:
                for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b);crc=zlib.crc32(b,crc);size+=len(b)
            check('member SHA/bytes/CRC: '+rel,size==row['bytes']==info.file_size and h.hexdigest()==row['sha256'] and (crc&0xffffffff)==info.CRC)
            totals+=size;zip_rows.append(dict(path=rel,bytes=size,sha256=h.hexdigest(),crc32=f'{crc&0xffffffff:08x}'))
        check('manifest total accounting',manifest['file_count']==len(records) and manifest['bytes']==sum(r['bytes'] for r in records))
        for p in docs.rglob('*'):
            if p.is_file():
                rel=p.relative_to(docs).as_posix();check('local document equals archived copy: '+rel,rel in expected and sha_file(p)==expected[rel]['sha256'])
        def j(rel):return json.loads(z.read(prefix+rel))
        old=j('new60_20261014/delivery_manifest.json')
        for r in old['records']:
            row=expected.get('new60_20261014/'+r['path']);check('V1 byte identity: '+r['path'],row is not None and row['sha256']==r['sha256'] and row['bytes']==r['bytes'])
        v='new60_opt_v2/';choice=j(v+'both_versions_choices_frozen.json');ch=expected[v+'both_versions_choices_frozen.json']['sha256']
        check('both-version freeze sidecar',ch in z.read(prefix+v+'both_versions_choices_frozen.sha256').decode())
        check('manifest bound to final choices',manifest['choices_sha256']==ch)
        check('prediction and fitting branch counts',sum(m['branch']=='prediction' for m in choice['models'])==8 and sum(m['branch']=='fitting_full60' for m in choice['models'])==3)
        for m in choice['models']:
            for pk,hk in [('model_relative_path','model_sha256'),('inference_module_relative_path','inference_module_sha256'),('model_source_module_relative_path','model_source_module_sha256')]:
                if pk in m:check('frozen artifact resolves '+m['id']+' '+pk,v+m[pk] in expected and expected[v+m[pk]]['sha256']==m[hk])
            cache=f"new60_20261014/cache_v1/{m['region']}.npz";check('input cache binding '+m['id'],expected[cache]['sha256']==m['cache_sha256'])
        split=posixpath.normpath(v+choice['split_relative_path']);check('original split resolves in sibling V1',split in expected and expected[split]['sha256']==choice['split_sha256'])
        check('protocol SHA binding',expected[v+'protocol.json']['sha256']==choice['protocol_sha256'])
        for name in names:
            if name.endswith('.docx'):
                with zipfile.ZipFile(io.BytesIO(z.read(name))) as word:
                    for entry in word.namelist():
                        if entry.endswith(('.xml','.rels')):ET.fromstring(word.read(entry))
                    text=''.join(ET.fromstring(word.read('word/document.xml')).itertext())
                    check('DOCX parse and no template leftovers: '+name,not any(x in text for x in ['${','__METADATA__','__PAYLOAD__','TODO','TBD','待填写']))
        text=z.read(prefix+'两版本说明.md').decode('utf-8')
        check('two error meanings in report','训练内误差' in text and '复用测试' in text and '3.158007363' in text and '0.044774598' in text)
        check('maximum local rotor errors explicit','0.04080646' in text and '0.91381222' in text)
        web=j(v+'website/manifest.json');check('web frozen identity/default',web['choices_sha256']==ch and web['defaults']['mode']=='fitting_full60')
        precision=j(v+'website/display_precision.json');check('all660 payloads retained',len(precision['payloads'])==660)
        for r in precision['payloads']:
            key=v+'website/'+r['path'];check('web payload bound '+r['path'],key in expected and expected[key]['sha256']==r['sha256'])
        deployment_key=v+'website/deployment.json'
        if deployment_key in expected:
            dep=j(deployment_key);check('published URL correct and declared verified',dep.get('verified') is True and '/new60-opt-v2/' in dep.get('url',''))
            check('published URL in report and README',dep['url'] in text and dep['url'] in z.read(prefix+'README.md').decode())
        # Figure recipes explicitly enumerate their scientific inputs. Only
        # actual reference keys are resolved; historical path strings are data.
        figure_recipes=[n for n in names if '/new60_opt_v2/figures/' in n and n.endswith('.json') and ('recipe' in PurePosixPath(n).name or 'manifest' in PurePosixPath(n).name)]
        recipe_refs=[]
        def resolve_ref(value,recipe):
            if not isinstance(value,str) or value.startswith(('http:','https:')):return None
            path=value.replace('\\','/')
            candidates=[v+path,posixpath.normpath(posixpath.dirname(recipe[len(prefix):])+'/'+path),path]
            return next((c for c in candidates if c in expected),None)
        def walk(obj,recipe):
            if isinstance(obj,dict):
                for key,value in obj.items():
                    if key in {'path','relative_path','input_path','input_npz','input_csv','source_path','source_npz','prediction_npz','artifact_relative_path'} and isinstance(value,str):
                        resolved=resolve_ref(value,recipe)
                        if value.endswith(('.npz','.png','.pdf','.svg','.py','.json','.csv')):
                            check('figure recipe reference resolves: '+value,resolved is not None)
                            hashkey='input_sha256' if key=='input_csv' else 'sha256'
                            if hashkey in obj:check('figure recipe input SHA: '+value,expected[resolved]['sha256']==obj[hashkey])
                            recipe_refs.append(dict(recipe=recipe[len(prefix):],reference=value,resolved=resolved))
                    walk(value,recipe)
            elif isinstance(obj,list):
                for item in obj:walk(item,recipe)
        for name in figure_recipes:
            recipe=json.loads(z.read(name));walk(recipe,name)
            if 'figures' in recipe:
                for fig in recipe['figures']:
                    for ext in ['.png','.pdf','.svg']:
                        rel=posixpath.dirname(name[len(prefix):])+'/'+fig['name']+ext
                        check('figure format exists: '+rel,rel in expected)
                if 'plotting_source_sha256' in recipe:
                    rel=posixpath.dirname(name[len(prefix):])+'/build_figures_v2.py'
                    check('figure plotting source SHA',rel in expected and expected[rel]['sha256']==recipe['plotting_source_sha256'])
        check('at least18 V2 figure groups archived',len(j(v+'figures/figure_recipe.json')['figures'])>=18)
        report=dict(status='PASS',all_checks_pass=True,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha_file(archive),entry_count=len(infos),uncompressed_bytes=totals,manifest_sha256=hashlib.sha256(raw).hexdigest(),checks=checks,member_checks=zip_rows,figure_recipe_count=len(figure_recipes),figure_reference_count=len(recipe_refs),figure_references=recipe_refs,full_extraction_performed=False,models_reselected=False,source_files_modified=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=f"# 新60双版本交付完整性检查\n\n状态：PASS。共{len(checks)}项检查。\n\n- ZIP：{archive.name}\n- ZIP字节：{report['archive_bytes']}\n- 成员数：{report['entry_count']}\n- 全部解压字节：{report['uncompressed_bytes']}\n- SHA256：`{report['archive_sha256']}`\n\n每个成员独立流式核对CRC、解压字节数和SHA256，并与外层manifest逐项一致；未全量解压。原V1逐文件身份、11冻结模型/推理源码/缓存/split、660网页载荷、DOCX全部XML与说明口径均通过。图配方{report['figure_recipe_count']}份、输入引用{report['figure_reference_count']}项已检查。\n\n本检查不训练、不推理、不据复用测试重选模型，不修改任何源文件或完成交付。详见同名JSON。\n"
    output.with_suffix('.md').write_text(summary,encoding='utf-8');return {k:report[k] for k in ['status','archive_bytes','archive_sha256','entry_count','uncompressed_bytes','figure_recipe_count','figure_reference_count']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zip',type=Path,required=True);p.add_argument('--documents',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(a.zip,a.documents,a.output),ensure_ascii=False))
