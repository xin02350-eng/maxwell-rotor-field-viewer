"""Build a final SHA whitelist; inspect model metadata, never numeric labels.

--inspect resolves the inference closure only. The default final-plan operation
requires all frozen results, website, figures, portable and publication gates.
No source file is copied, trained, inferred, changed or deleted by this tool.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','__pycache__','.venv','venv','node_modules','site-packages',
          'portable_delivery_check','delivery_preview','synthetic_fixture','toy_fixture',
          'scaffold_fixture','failed_runs','failedruns'}
REVIEWS=['qa/scientific_qa.json','qa/cli_v5_check.json','qa/web_payloads_v5_check.json',
         'qa/browser_v5_check.json','figures/figure_check.json','qa/fitbase_restore_check.json',
         'qa/portable_v5_check.json','qa/final_static_review_v5.json','qa/public_deployment_v5_check.json','website/deployment.json']
TEXT={'.py','.json','.csv','.log','.md','.txt','.sha256','.toml','.yaml','.yml'}
PATH_KEYS={'model_relative_path','module_relative_path','inference_module_relative_path',
           'model_source_module_relative_path','teacher_model_relative_path','teacher_module_relative_path',
           'baseline_model_relative_path','baseline_module_relative_path'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def passed(j):return any(j.get(k) is True for k in ('all_checks_pass','all_pass','pass','passed','verified')) or str(j.get('status','')).upper()=='PASS'
def write_new(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

class Inventory:
    def __init__(self,work):self.work=Path(work).resolve();self.rows={};self.model_dependencies={};self.visited=set();self.closures={};self.omitted=[]
    def path(self,p):
        p=Path(p).absolute()
        if any(x in EXCLUDED for x in p.parts):raise ValueError('Excluded runtime/fixture path: '+str(p))
        for q in [p,*p.parents]:
            if q==self.work.parent:break
            if q.is_symlink():raise ValueError('Symlink dependency: '+str(p))
        p=p.resolve()
        if not p.is_relative_to(self.work) or not p.is_file():raise FileNotFoundError(p)
        return p
    def add(self,p,role,expected=None,reason='explicit dependency'):
        p=self.path(p);rel=p.relative_to(self.work).as_posix()
        if rel not in self.rows:self.rows[rel]={'source_relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p),'role':role,'reasons':[reason]}
        row=self.rows[rel]
        if expected is not None and row['sha256']!=expected:raise ValueError('SHA changed: '+rel)
        if reason not in row['reasons']:row['reasons'].append(reason)
        return rel
    def sidecars(self,p,role='model_identity'):
        p=Path(p);found=[]
        for q in {p.with_suffix(p.suffix+'.sha256'),p.with_suffix('.sha256')}:
            if q.is_file():
                if sha(p) not in q.read_text(encoding='utf-8-sig'):raise ValueError('Bad SHA sidecar: '+str(q))
                found.append(self.add(q,role))
        return found
    def model(self,p,expected=None):
        p=self.path(p);rel=self.add(p,'model',expected,'frozen model/component closure');out={rel,*self.sidecars(p)}
        if p in self.closures:return set(self.closures[p])
        if p in self.visited:raise ValueError('Cyclic component graph: '+str(p))
        self.visited.add(p)
        # Only scalar JSON is decoded. No field/coeff/truth or model matrices.
        with np.load(p,allow_pickle=False) as z:
            meta={}
            for key in ('metadata_json','meta_json'):
                if key in z.files:
                    value=z[key]
                    if value.dtype.kind not in 'US' or value.size!=1:raise ValueError('Invalid model metadata scalar')
                    meta=json.loads(str(value));break
        def visit(obj):
            if isinstance(obj,list):
                for v in obj:visit(v)
            elif isinstance(obj,dict):
                for key,value in obj.items():
                    if key in PATH_KEYS and isinstance(value,str):
                        q=(p.parent/value).resolve()
                        hk=key.replace('_relative_path','_sha256')
                        expected_hash=obj.get(hk)
                        if key=='inference_module_relative_path':expected_hash=obj.get('inference_module_sha256')
                        if q.suffix=='.npz':out.update(self.model(q,expected_hash))
                        elif q.suffix=='.py':out.add(self.add(q,'source_code',expected_hash,'component/teacher source'))
                        else:raise ValueError('Unexpected dependency file type: '+value)
                    elif isinstance(value,(dict,list)):visit(value)
                    # components=['|B|'] is a physical descriptor, not a path.
        visit(meta)
        for q in p.parent.iterdir():
            if q.is_file() and q.suffix in TEXT and q.name not in {'validation_predictions.csv','predictions.csv'}:
                out.add(self.add(q,'training_log',reason='selected model sibling provenance/log'))
        self.closures[p]=set(out)
        return out

def add_code_tree(inv,study):
    """Small source files ensure lazy imports and embedded-teacher code resolve."""
    found=set()
    for p in study.rglob('*.py'):
        rel=p.relative_to(study)
        if any(x in EXCLUDED or x in {'website','figures','evaluation'} for x in rel.parts):continue
        if any(x.startswith('synthetic') or x.startswith('toy_') for x in rel.parts):continue
        found.add(inv.add(p,'source_code',reason='same-study local/lazy import and retraining sources'))
    return found

def collect_inference(root=ROOT,include_all_source=True):
    root=Path(root).resolve();work=root.parent;registry=read(root/'both_versions_choices_frozen.json');ch=sha(root/'both_versions_choices_frozen.json')
    if ch not in (root/'both_versions_choices_frozen.sha256').read_text():raise ValueError('Freeze sidecar mismatch')
    inv=Inventory(work)
    for name in ['protocol.json','prediction_choices_frozen.json','both_versions_choices_frozen.json']:
        inv.add(root/name,'frozen_choices');inv.sidecars(root/name,'frozen_choices')
    if sha(root/'protocol.json')!=registry['protocol_sha256']:raise ValueError('Protocol SHA changed')
    if sha(root/'prediction_choices_frozen.json')!=registry['prediction_freeze_sha256']:raise ValueError('Prediction freeze SHA changed')
    inv.add(root/registry['split_relative_path'],'split',registry['split_sha256'])
    shared=set()
    for ref in registry.get('support_files',[]):shared.add(inv.add(root/ref['relative_path'],'source_code',ref['sha256'],'frozen support source'))
    for m in registry['models']:
        deps=inv.model(root/m['model_relative_path'],m['model_sha256'])
        for pk,hk in [('inference_module_relative_path','inference_module_sha256'),('model_source_module_relative_path','model_source_module_sha256')]:
            if pk in m:deps.add(inv.add(root/m[pk],'source_code',m[hk],'frozen inference source'))
        for row in m.get('provenance_files',[]):deps.add(inv.add(root/row['relative_path'],'training_log',row['sha256'],'frozen training provenance'))
        cache=root/m.get('cache_relative_path',f'../new60_20261014/cache_v1/{m["region"]}.npz')
        deps.add(inv.add(cache,'cache',m['cache_sha256'],'input and geometry cache; labels not decoded'))
        inv.model_dependencies[m['id']]=deps
    studies={work/Path(p).parts[0] for p in inv.rows if Path(p).parts[0].startswith('new60_opt_')}
    if include_all_source:
        for study in studies:shared.update(add_code_tree(inv,study))
    for name in ['predict_v5.py','inference_numpy_v5.py','scope_utils_v5.py','requirements-inference.txt','serve_website_v5.py']:
        shared.add(inv.add(root/'code'/name,'inference_cli'))
        inv.rows[(root/'code'/name).relative_to(work).as_posix()]['role']='inference_cli'
    for mid in inv.model_dependencies:inv.model_dependencies[mid]=sorted(inv.model_dependencies[mid]|shared)
    return inv,registry,ch

def add_tree(inv,folder,role):
    folder=Path(folder)
    if not folder.is_dir():raise FileNotFoundError(folder)
    for p in folder.rglob('*'):
        if p.is_file() and not any(x in EXCLUDED for x in p.relative_to(folder).parts):inv.add(p,role)

def preparation(root=ROOT,output=None,preflight=False):
    root=Path(root).resolve();work=root.parent
    required=['both_versions_choices_frozen.json','final_summary.json','website/manifest.json','website/results.json','website/display_precision.json','figures/figure_recipe.json',*REVIEWS]
    missing=[str(p) for p in required if not (root/p).is_file()]
    if missing:return {'ready':False,'missing':missing,'plan_written':False,'numeric_labels_loaded':False}
    inv,registry,ch=collect_inference(root)
    for rel in REVIEWS:
        j=read(root/rel)
        if not passed(j) or (j.get('choices_sha256') and j['choices_sha256']!=ch):raise ValueError('Review failed or wrong freeze: '+rel)
        inv.add(root/rel,'qa')
    for rel in ['final_summary.json','website/manifest.json','website/results.json']:
        if read(root/rel)['choices_sha256']!=ch:raise ValueError('Final result freeze mismatch: '+rel)
    # Original archive plus unmodified merged input, metadata and all three caches.
    inv.add(root/'source_inputs/NEW60_20261010_original.zip','original_input_zip')
    base=work/'new60_20261014'
    inv.add(base/'source/combined/flux60_s60_merged.npz','merged_source')
    add_tree(inv,base/'source/metadata','source_metadata')
    for p in (base/'source').glob('*'):
        if p.is_file():inv.add(p,'source_metadata')
    for p in (base/'cache_v1').glob('*'):
        if p.is_file():inv.add(p,'cache' if p.suffix=='.npz' else 'split')
    add_code_tree(inv,base/'code')
    for p in base.iterdir():
        if p.is_file() and p.suffix in {'.json','.md','.txt'}:inv.add(p,'source_metadata')
    # Real scientific references only, retaining their original sibling paths.
    result=read(root/'website/results.json')
    for m in result['models']:
        for ref in m['sources']:
            inv.add(work/ref['path_relative_to_work'],'evaluation',ref['sha256'],'selected scientific original-precision reference')
            source=work/ref['scientific_source_relative_root']
            inv.add(source/'both_versions_choices_frozen.json','frozen_choices',ref['scientific_choices_sha256'])
            inv.sidecars(source/'both_versions_choices_frozen.json','frozen_choices')
            if (source/'qa/scientific_qa.json').exists():inv.add(source/'qa/scientific_qa.json','qa')
    # V5 completed experiment source/logs and negative results, not repeated predictions.
    keep_dirs={work/Path(p).parts[0] for p in inv.rows if Path(p).parts[0].startswith('new60_opt_')}
    skip={'evaluation','website','figures','qa','prepared','data','diagnostics','source_inputs','provenance_restore'}
    for study in keep_dirs:
        for p in study.rglob('*'):
            rel=p.relative_to(study)
            if not p.is_file() or any(x in EXCLUDED or x in skip or x.startswith('synthetic') for x in rel.parts):continue
            if p.name=='delivery_dependencies.json':continue
            if p.suffix in TEXT:inv.add(p,'source_code' if p.suffix=='.py' else 'training_log')
            elif study==root and p.name=='model.npz' and (p.parent/'result.json').exists():inv.model(p)
    add_tree(inv,root/'figures','figure');add_tree(inv,root/'website','website')
    # Small V5 QA documents, source and screenshots, excluding staging/fixtures.
    for p in (root/'qa').rglob('*'):
        rel=p.relative_to(root/'qa')
        if p.is_file() and not any(x in EXCLUDED or x.startswith('synthetic') for x in rel.parts) and p.suffix in TEXT|{'.png','.jpg'}:
            inv.add(p,'qa')
    recipe=read(root/'figures/figure_recipe.json')
    for ref in recipe.get('scientific_inputs',[]):inv.add(root/'figures'/ref['path'],'evaluation',ref['sha256'],'figure exact scientific input')
    # Byte-exact restoration of old V2 heads instead of duplicate full base weights.
    restore=work/'new60_opt_v4/code/restore_original_fit_bases.py';inv.add(restore,'source_code')
    for region in ['rotor','stator','airgap']:
        payload=work/'new60_opt_v4/provenance_restore'/region/'base_head_restore.npz';inv.add(payload,'provenance_restore')
        with np.load(payload,allow_pickle=False) as z:meta=json.loads(str(z['restore_metadata_json']))
        original=meta['original_model_relative_path']
        if original in inv.rows:raise ValueError('Restorable original fit base unnecessarily retained: '+original)
        sources=[restore.relative_to(work).as_posix(),payload.relative_to(work).as_posix(),f'new60_opt_v3/fitting_full60/{region}/model.npz']
        inv.omitted.append(dict(source_relative_path=original,bytes=meta['original_model_bytes'],sha256=meta['original_model_sha256'],reason='Exact V2 base reconstructed from the retained V3 calibrated model plus original tiny head; verify-only checks byte-exact SHA before optional restoration.',rebuild_source_relative_paths=sources))
    for p in (work/'new60_opt_v4/qa').glob('*restore*'):
        if p.is_file():inv.add(p,'qa')
    # Explicitly record omitted, regenerable development arrays only, not unrelated old experiments.
    rebuild=sorted(p for p,r in inv.rows.items() if r['role']=='source_code')
    for study in keep_dirs:
        for p in study.rglob('*.npz'):
            rel=p.relative_to(work).as_posix();parts=p.relative_to(study).parts
            if rel in inv.rows or any(x in EXCLUDED for x in parts):continue
            if p.name in {'cycles.npz','all_validation_coefficient_predictions.npz','validation_predictions.npz','validation_truth.npz','mean_data.npz','representation.npz','prepared_data.npz'} or 'prepared' in parts:
                inv.omitted.append(dict(source_relative_path=rel,bytes=p.stat().st_size,sha256=sha(p),reason='Regenerable training preparation or duplicated development predictions; frozen inference, source labels, logs and selected scientific arrays are retained.',rebuild_source_relative_paths=rebuild))
    # Resolve every retained scientific source, component and restoration input.
    for row in inv.omitted:
        if any(x not in inv.rows for x in row['rebuild_source_relative_paths']):raise ValueError('Restoration/rebuild input missing')
    plan=dict(schema_version=1,choices_sha256=ch,dependencies_complete=True,root_relative_to_work=root.name,
              files=sorted(inv.rows.values(),key=lambda x:x['source_relative_path']),model_dependencies=inv.model_dependencies,
              required_reviews=REVIEWS,omitted=inv.omitted,numeric_labels_loaded=False,model_matrices_loaded=False,
              provenance_restore=dict(command='python new60_opt_v4/code/restore_original_fit_bases.py --verify-only',restore_command='python new60_opt_v4/code/restore_original_fit_bases.py',inference_requires_restoration=False),
              previous_delivery={'filename':'沈夏鑫_新60双版本优化_20261014.zip','sha256':'74a8309697a67a7d1ea50a90366dffb1bcb0df58778eee03cf4f748d3fafc3fa'},
              omission_scope='Only explicitly selected dependencies and source/log records are retained, not whole V2/V3/V4 trees. Unselected historical model weights are outside this selected package; see old complete delivery.')
    if not preflight:write_new(output or root/'delivery_dependencies.json',plan)
    return {'ready':True,'plan_written':not preflight,'files':len(inv.rows),'bytes':sum(r['bytes'] for r in inv.rows.values()),'models':len(registry['models']),'omitted':len(inv.omitted),'choices_sha256':ch,'numeric_labels_loaded':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.inspect:
        inv,reg,ch=collect_inference(a.root);out=dict(models=len(reg['models']),files=len(inv.rows),bytes=sum(r['bytes'] for r in inv.rows.values()),choices_sha256=ch,numeric_labels_loaded=False,model_dependencies=inv.model_dependencies)
    else:out=preparation(a.root,a.output,a.preflight)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
