"""Small temporary component fixtures; never copy real studies or train models."""
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery_builder',ROOT/'code/build_delivery_v2.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
checks=[]
def check(name,condition):
    checks.append(dict(name=name,pass_=bool(condition)))
    if not condition:raise AssertionError(name)

context=b.preflight(ROOT,b.DEFAULT_V1,['qa/cli_v2_check.json','qa/viewer_components_check.json'])
check('real frozen schema preflight ready',context['ready'])
markdown=b.report(ROOT,context)
check('both fitting and reused-test scope explicit','训练内误差' in markdown and '复用测试' in markdown)
check('deployment magnitude figure appears','0.04328276' in markdown)
with tempfile.TemporaryDirectory(prefix='maxwell_delivery_component_') as temp:
    temp=Path(temp);word=temp/'report.docx';b.docx(word,markdown)
    with zipfile.ZipFile(word) as z:
        for info in z.infolist():ET.fromstring(z.read(info))
    check('all DOCX XML parses',True)
    # Replace only readiness/source-identity guards for a tiny stream-copy fixture.
    # The actual copy, SHA/CRC, archive names, rename, and refusal code runs unchanged.
    root=temp/'v2';old=temp/'v1';root.mkdir();old.mkdir();(root/'array.npz').write_bytes(b'opaque npz fixture; never decoded')
    (old/'source.json').write_text('{"fixture":true}')
    b.write(old/'delivery_manifest.json',dict(records=[dict(path='source.json',bytes=(old/'source.json').stat().st_size,sha256=b.digest(old/'source.json'))]))
    b.preflight=lambda *a,**k:dict(ready=True,choices_sha256='fixture-only',website_deployment=None)
    b.verify_frozen=lambda *a:({},'fixture-only')
    b.report=lambda *a:'# Stream component fixture\n\n训练内误差与复用测试。'
    original=b.digest(root/'array.npz');result=b.build_zip_only(root,old,temp/'delivery')
    check('stream integrity checked internally',result['stream_crc_sha_verified'])
    check('source unchanged',b.digest(root/'array.npz')==original)
    check('no whole study copied beside ZIP',not (temp/'delivery'/'new60_opt_v2').exists())
    with zipfile.ZipFile(result['zip']) as z:
        check('single named root in archive',all(x.startswith('delivery/') for x in z.namelist()))
        check('both source siblings in archive','delivery/new60_20261014/source.json' in z.namelist() and 'delivery/new60_opt_v2/array.npz' in z.namelist())
        manifest=json.loads(z.read('delivery/delivery_manifest.json'))
        check('all manifest files have ZIP members',all('delivery/'+row['path'] in z.namelist() for row in manifest['records']))
    try:b.build_zip_only(root,old,temp/'delivery')
    except FileExistsError:check('refuses overwrite',True)
    else:check('refuses overwrite',False)

report=dict(status='PASS',all_checks_pass=True,scope='Real report schema and DOCX XML; tiny temporary streaming-copy fixtures, not a formal delivery',builder_sha256=b.digest(ROOT/'code/build_delivery_v2.py'),checks=checks)
(ROOT/'qa/delivery_components_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',checks=len(checks)),ensure_ascii=False))
