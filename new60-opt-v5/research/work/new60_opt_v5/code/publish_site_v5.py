"""Publish verified V5 site in bounded commits without copying its payloads.

Uses a separate Git index; the existing checkout and local branch are preserved.
Only non-force fast-forward pushes are used. A journal permits safe resumption.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, re, subprocess, urllib.request, time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1] / 'outputs/website'
ROUTE = 'new60-opt-v5'
URL = 'https://xin02350-eng.github.io/maxwell-rotor-field-viewer/' + ROUTE + '/'
EXPECTED_PARENT = '770372e735b9abb442ee85761f65e09f3d61962d'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))

def main():
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group();modes.add_argument('--verify-only',action='store_true');modes.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    choices=sha(ROOT/'both_versions_choices_frozen.json')
    journal=ROOT/'qa/publication_journal.json'
    env=os.environ.copy();env['GIT_INDEX_FILE']=str(ROOT/'qa/publication_git_index')
    def git(*a,input=None,original_index=False):
        selected_env=dict(env)
        if original_index:selected_env.pop('GIT_INDEX_FILE',None)
        r=subprocess.run(['git','-C',str(REPO),*a],env=selected_env,input=input,capture_output=True,text=True,encoding='utf8',errors='replace')
        if r.returncode:raise RuntimeError('git '+a[0]+' failed: '+r.stderr[-2500:])
        return r.stdout.strip()
    def remote():return git('ls-remote','origin','refs/heads/main').split()[0]
    def save(j):journal.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if not args.verify_only:
        for name in ('scientific_qa','cli_v5_check','web_payloads_v5_check','browser_v5_check','figures_v5_check'):
            p=ROOT/'qa'/(name+'.json');j=read(p)
            if not any(j.get(k) is True for k in ('all_checks_pass','all_pass','pass','passed','verified')):raise ValueError('Required QA not PASS: '+str(p))
            if j.get('choices_sha256') and j['choices_sha256']!=choices:raise ValueError('QA freeze mismatch '+name)
        if git('status','--porcelain',original_index=True):raise ValueError('Existing checkout has local changes')
        file_manifest=ROOT/'qa/public_files_manifest.json'
        files=[(p,ROUTE+'/'+p.relative_to(ROOT/'website').as_posix()) for p in (ROOT/'website').rglob('*') if p.is_file() and p.name!='deployment.json']
        # Publish inspectable training/inference sources alongside the site.
        # Raw labels and large model weights remain in the standalone ZIP.
        preview=read(ROOT/'qa/dependency_closure_preview.json');research=set()
        for refs in preview['model_dependencies'].values():
            for rel in refs:
                p=ROOT.parent/rel
                if 'diagnostics_train_only' not in p.parts and (p.suffix=='.py' or p.name.startswith('requirements-')):research.add(p)
        for p in ROOT.rglob('*'):
            rel=p.relative_to(ROOT)
            if not p.is_file() or any(x in ('website','figures','qa','source_inputs','__pycache__') for x in rel.parts):continue
            if p.suffix in ('.py','.json','.log','.md','.txt','.sha256') or (p.suffix=='.csv' and 'runs' in rel.parts):research.add(p)
        for p in sorted(research):files.append((p,ROUTE+'/research/work/'+p.relative_to(ROOT.parent).as_posix()))
        public_readme=ROOT/'code/PUBLIC_RESEARCH_README.md'
        public_readme.write_text('# V5 模型研究源代码\n\n这里保留冻结模型选择、实际神经网络微调源代码/日志与NumPy推理源代码。原始数据、大型权重和科学数组在独立完整交付ZIP中；本GitHub源代码目录不单独包含运行所需全部权重。\n\n公共快照中的本机用户名和微信wxid已替换为USER与WXID_REDACTED。SOURCE_HASHES.json逐项保留原文件SHA与公共文件SHA；脱敏快照的SHA可能与原始冻结记录不同。完整交付ZIP保留未修改的原始源码、权重及哈希，重现模型请使用完整包。\n\n转子51工况纯NN复用测试相对L2为2.887704518%，混合51版为2.855952755%。原测试已在前轮查看，51版无独立验证集；全60训练内拟合不能代表预测精度。\n',encoding='utf8')
        files=[(p,d) for p,d in files if p!=public_readme];files.append((public_readme,ROUTE+'/research/README.md'))
        files.sort(key=lambda x:(x[0].suffix not in ('.gz','.bin'),x[1]))
        if file_manifest.exists():
            recorded=read(file_manifest)
            if recorded['choices_sha256']!=choices:raise ValueError('Public file manifest freeze mismatch')
            files=[(ROOT.parent/r['source_relative_to_work'],r['public_path']) for r in recorded['files']]
            for (p,dest),r in zip(files,recorded['files']):
                if sha(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('Frozen public file changed: '+dest)
        else:
            public_sources=[];mapped=[]
            for p,dest in files:
                if '/research/' in dest:
                    original=p;content=p.read_text('utf-8-sig')
                    public=re.sub(r'wxid_[A-Za-z0-9_]+','WXID_REDACTED',content)
                    public=re.sub(r'\bUSER\b','USER',public,flags=re.IGNORECASE)
                    if public!=content:
                        p=ROOT/'qa/publication_source_snapshots'/original.relative_to(ROOT.parent)
                        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(public,encoding='utf8')
                    public_sources.append(dict(original_relative_to_work=original.relative_to(ROOT.parent).as_posix(),public_path=dest,original_sha256=sha(original),public_sha256=sha(p),redacted=public!=content))
                mapped.append((p,dest))
            source_map=ROOT/'qa/public_source_hashes.json'
            source_map.write_text(json.dumps(dict(choices_sha256=choices,redaction='local username and wxid only; original sources remain immutable in the complete delivery',files=public_sources),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            files=mapped+[(source_map,ROUTE+'/research/SOURCE_HASHES.json')]
            recorded=dict(choices_sha256=choices,files=[dict(source_relative_to_work=p.relative_to(ROOT.parent).as_posix(),public_path=d,sha256=sha(p),bytes=p.stat().st_size) for p,d in files])
            file_manifest.write_text(json.dumps(recorded,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        if args.prepare_only:
            print(json.dumps(dict(prepared=True,choices_sha256=choices,manifest=str(file_manifest),files=len(files),bytes=sum(p.stat().st_size for p,d in files),published=False)),flush=True)
            return
        pending=[];batches=[];size=0
        for p,dest in files:
            if pending and size+p.stat().st_size>20*1024*1024:batches.append(pending);pending=[];size=0
            pending.append((p,dest));size+=p.stat().st_size
        if pending:batches.append(pending)
        source_digest=hashlib.sha256('\n'.join(dest+':'+sha(p) for p,dest in files).encode()).hexdigest()
        if journal.exists():
            j=read(journal)
            if j['choices_sha256']!=choices or j['source_digest']!=source_digest:raise ValueError('Publication source changed after journal')
        else:
            if remote()!=EXPECTED_PARENT:raise ValueError('Remote main changed; inspect before publication')
            j=dict(choices_sha256=choices,source_digest=source_digest,parent=EXPECTED_PARENT,batches=[],file_count=len(files),total_bytes=sum(p.stat().st_size for p,dest in files),url=URL)
            save(j)
        parent=j['batches'][-1]['commit'] if j['batches'] else j['parent']
        for i,batch in enumerate(batches):
            if i<len(j['batches']):continue
            if remote()!=parent:raise ValueError('Remote main no longer matches recorded publication parent')
            git('read-tree',parent)
            for p,dest in batch:
                obj=git('hash-object','-w','--no-filters',str(p))
                git('update-index','--add','--cacheinfo','100644',obj,dest)
            git('-c','core.whitespace=cr-at-eol','diff','--cached','--check')
            tree=git('write-tree')
            commit=git('commit-tree',tree,'-p',parent,input=f'Publish V5 verified rotor site ({i+1}/{len(batches)})\n')
            try:git('push','origin',commit+':refs/heads/main')
            except RuntimeError:
                if remote()!=commit:raise
            if remote()!=commit:raise ValueError('Push did not produce expected remote commit')
            j['batches'].append(dict(index=i,commit=commit,files=len(batch),bytes=sum(p.stat().st_size for p,dest in batch)))
            save(j);parent=commit
            print(json.dumps(dict(published_batch=i+1,total_batches=len(batches),commit=commit)),flush=True)
        j['completed_utc']=datetime.now(timezone.utc).isoformat();j['commit']=parent;save(j)
    j=read(journal);commit=j['commit']
    if remote()!=commit:raise ValueError('Public main changed after final publication')
    checks=[]
    for rel in ('','manifest.json','results.json','app.js','display_precision.json'):
        request=urllib.request.Request(URL+rel,headers={'Cache-Control':'no-cache','User-Agent':'Maxwell V5 artifact verifier'})
        with urllib.request.urlopen(request,timeout=30) as r:data=r.read();status=r.status
        local=ROOT/'website'/('index.html' if rel=='' else rel)
        if status!=200 or hashlib.sha256(data).hexdigest()!=sha(local):raise ValueError('Public content mismatch '+rel)
        checks.append(dict(path=rel or 'index.html',http_status=status,sha256=sha(local)))
    manifest=read(ROOT/'website/manifest.json')
    # Resolve a real gzip cloud through the verified manifest structure.
    def strings(v):
        if isinstance(v,str):yield v
        elif isinstance(v,dict):
            for x in v.values():yield from strings(x)
        elif isinstance(v,list):
            for x in v:yield from strings(x)
    payload=next((s for s in strings(manifest) if s.endswith('.gz') and (ROOT/'website'/s).is_file()),None)
    if payload is None:raise ValueError('No real cloud payload in manifest')
    with urllib.request.urlopen(URL+payload,timeout=30) as r:data=r.read()
    if hashlib.sha256(data).hexdigest()!=sha(ROOT/'website'/payload):raise ValueError('Public cloud SHA mismatch')
    checks.append(dict(path=payload,sha256=sha(ROOT/'website'/payload),http_status=200))
    report=dict(verified=True,all_checks_pass=True,choices_sha256=choices,url=URL,commit=commit,verified_utc=datetime.now(timezone.utc).isoformat(),checks=checks)
    (ROOT/'website/deployment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (ROOT/'qa/public_deployment_v5_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
