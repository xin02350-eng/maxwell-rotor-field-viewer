"""Publish own previously-unpublished change in bounded, non-force push batches."""
import subprocess,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT.parents[1]/'outputs/website'
BASE='ef23c9a17763a65db910f5a581d5339e05e02826'
OWN='3a41adeff74fd989e79170050e22c4c78f6ced2d'
LOG=ROOT/'qa/publish_batches_progress.json'
def git(*args,timeout=180):
 r=subprocess.run(['git',*args],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stdout.decode('utf8','replace')[-2000:])
 return r.stdout.decode('utf8','replace').strip()
def record(d):LOG.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def push():
 # Small content-length requests avoid the timed-out318MB chunked upload.
 return git('-c','http.postBuffer=67108864','-c','http.version=HTTP/1.1','push','--porcelain','origin','main',timeout=300)
def main():
 if LOG.exists():
  d=json.loads(LOG.read_text(encoding='utf8'));print('Resuming existing bounded publication',flush=True)
  # A failed push can be resumed without moving any branch/ref again.
  if d.get('pending_push'):
   print(push(),flush=True);d['pending_push']=False
   if d.get('pending_batch') is not None:d['completed_batches'].append(d.pop('pending_batch'))
   if d.get('phase')=='final':
    d['finished_utc']=datetime.now(timezone.utc).isoformat();d['final_commit']=git('rev-parse','HEAD');d['passed']=True;record(d);print('Final publication resumed successfully',flush=True);return
   record(d)
 else:
  assert git('rev-parse','HEAD')==OWN,'Unexpected local commit; refuse ref changes'
  remote=git('ls-remote','origin','refs/heads/main').split()[0]
  assert remote==BASE,'Remote changed; refuse reset'
  changes=git('status','--porcelain').splitlines()
  assert all(x.lstrip().split(' ',1)[1].startswith('new60-opt-v2/research/') for x in changes),changes
  git('branch','codex/new60-v2-unpublished-20261014',OWN)
  git('reset','--soft',BASE) # Own unpushed commit backed up; leaves all files intact.
  git('restore','--staged','--','.')
  d=dict(start_utc=datetime.now(timezone.utc).isoformat(),original_unpublished_commit=OWN,backup_branch='codex/new60-v2-unpublished-20261014',remote_base=BASE,completed_batches=[],pending_push=False)
  record(d)
 files=sorted((REPO/'new60-opt-v2/data').rglob('*.json.gz'))
 batches=[];group=[];size=0
 for p in files:
  if group and size+p.stat().st_size>24*1024**2:batches.append(group);group=[];size=0
  group.append(p.relative_to(REPO).as_posix());size+=p.stat().st_size
 if group:batches.append(group)
 done={x['batch'] for x in d['completed_batches']}
 for i,paths in enumerate(batches):
  if i in done:continue
  git('add','--',*paths)
  git('commit','--quiet','-m',f'Add NEW60 V2 field display data batch{i+1:02d}/{len(batches)}')
  sha=git('rev-parse','HEAD');d['pending_push']=True;d['pending_batch']=dict(batch=i,files=len(paths),commit=sha);record(d)
  print(f'Uploading batch{i+1}/{len(batches)} ({sum((REPO/p).stat().st_size for p in paths)/1024**2:.1f}MiB)',flush=True)
  msg=push();d['pending_push']=False;d['completed_batches'].append(d.pop('pending_batch'));record(d)
  print(f'Uploaded batch{i+1}/{len(batches)}',flush=True)
 # The manifest and navigation become public only after all data are present.
 git('add','--','new60-opt-v2','index.html','README.md','new60/guide.html')
 git('-c','core.whitespace=cr-at-eol','diff','--cached','--check')
 git('commit','--quiet','-m','Publish optimized NEW60 dual-mode viewer and reproducible source')
 d['pending_push']=True;d['phase']='final';record(d);print('Uploading final viewer, manifest and source',flush=True);print(push(),flush=True)
 d['pending_push']=False;d['finished_utc']=datetime.now(timezone.utc).isoformat();d['final_commit']=git('rev-parse','HEAD');d['passed']=True;record(d)
 print(json.dumps(dict(passed=True,batches=len(batches),final_commit=d['final_commit'])),flush=True)
if __name__=='__main__':main()
