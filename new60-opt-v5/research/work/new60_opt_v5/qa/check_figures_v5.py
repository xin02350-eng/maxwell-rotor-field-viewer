"""Read-only numeric figure audit; no inference or model modification."""
from pathlib import Path
import argparse,csv,json,hashlib,shutil,subprocess,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'figures';WORK=ROOT.parent
parser=argparse.ArgumentParser();parser.add_argument('--visual-review-json',type=Path,required=True);args=parser.parse_args()
checks=[]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def check(name,ok,detail=None):
 checks.append(dict(name=name,passed=bool(ok),detail=detail))
 if not ok:raise AssertionError(name)
def metrics(y,p,axes):
 e=p-y;s=np.sum(e*e,axis=axes);n=np.prod([y.shape[a] for a in axes]);den=np.sum(y*y,axis=axes)
 return dict(rmse_T=np.sqrt(s/n),mae_T=np.mean(abs(e),axis=axes),max_abs_T=np.max(abs(e),axis=axes),relative_L2_pct=100*np.sqrt(s/den))
visual=read(args.visual_review_json);assert visual['all_figures_visually_reviewed'] is True and len(visual['figures'])==8
recipe=read(FIG/'figure_recipe.json');qa=read(ROOT/'qa/scientific_qa.json');freeze=read(ROOT/'both_versions_choices_frozen.json')
check('current freeze identity and scientific QA',qa['all_checks_pass'] and qa['choices_sha256']==recipe['choices_sha256']==sha(ROOT/'both_versions_choices_frozen.json'))
check('actual visual review bound to this recipe',visual['choices_sha256']==recipe['choices_sha256'] and visual['figure_recipe_sha256']==sha(FIG/'figure_recipe.json') and sorted(visual['figures'])==sorted(s['name'] for s in recipe['figures']))
for item in visual['images']:check('reviewed PNG identity '+item['path'],sha(FIG/item['path'])==item['sha256'])
check('all8 actual PNGs reviewed',len(visual['images'])==8)
for category in ('scientific_inputs','plotting_sources','csv_inventory'):
 for r in recipe[category]:
  p=(FIG/r['path']).resolve();check(category+': '+r['path'],p.exists() and sha(p)==r['sha256'])
  if 'bytes' in r:check('bytes '+r['path'],p.stat().st_size==r['bytes'])
  if 'rows' in r:check('row count '+r['path'],len(csvread(p))==r['rows'])
allplot={r['name']:csvread(FIG/r['input_csv']) for r in recipe['figures']};conditionmap={};rotorgeo=None
for ch in freeze['models']:
 mid=ch['id'];part='training_fit' if ch['branch']=='fitting_full60' else 'reused_test';proof=next(r for r in qa['results'] if r['id']==mid and r['partition']==part)
 p=WORK/proof['scientific_source_relative_root']/proof['scientific_npz_relative_path']
 with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in ('truth','prediction','case_ids','point_id','xy','theta','conditions')}
 if mid=='fitting_rotor_magnitude':conditionmap={int(cid):condition for cid,condition in zip(a['case_ids'],a['conditions'])}
 if mid=='prediction_rotor_magnitude':rotorgeo=(a['point_id'].copy(),a['xy'].copy())
 for group,axes in (('case',(1,2)),('frame',(2,)),('point',(0,1))):
  rr=csvread(FIG/'data'/f'{mid}_{group}_metrics.csv');expected=metrics(a['truth'],a['prediction'],axes)
  for k,v in expected.items():check(mid+'/'+group+'/'+k,np.allclose(np.array([float(r[k]) for r in rr]),v.reshape(-1),rtol=3e-12,atol=1e-23))
  if group=='case':check(mid+' case identities',[int(r['case_id']) for r in rr]==a['case_ids'].tolist())
  elif group=='point':check(mid+' point identities',[int(r['point_id']) for r in rr]==a['point_id'].tolist())
  else:check(mid+' frame identities',[(int(r['case_id']),int(r['frame_index'])) for r in rr]==[(int(c),t) for c in a['case_ids'] for t in range(60)])
 for name,rr in allplot.items():
  field=[r for r in rr if r.get('model_id')==mid and 'truth_T' in r]
  if not field:continue
  for r in field:
   ci=a['case_ids'].tolist().index(int(r['case_id']));j=a['point_id'].tolist().index(int(r['point_id']));t=int(r['frame_index']);y=float(a['truth'][ci,t,j]);pred=float(a['prediction'][ci,t,j])
   if float(r['truth_T'])!=y or float(r['prediction_T'])!=pred or float(r['absolute_error_T'])!=abs(pred-y):raise AssertionError('CSV raw-value mismatch')
   theta=np.deg2rad(float(a['theta'][t]));xy=a['xy'][j]@np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
   if not np.allclose([float(r['x_world_mm']),float(r['y_world_mm'])],xy,rtol=1e-14,atol=1e-13):raise AssertionError('world coordinate mismatch')
  check(name+'/'+mid+' all field CSV values exact and world rotation correct',True,dict(rows=len(field)))
oldqa=read(WORK/'new60_opt_v3/qa/scientific_qa.json')
v4qa=read(WORK/'new60_opt_v4/qa/scientific_qa.json')
labelmap={'V2 / kernel42':(oldqa,'prediction_rotor_v2_kernel'),'V3 / mixed42':(oldqa,'prediction_rotor_magnitude'),'V4=V5 / mixed42':(qa,'prediction_rotor_magnitude'),'V4=V5 / mixed51':(qa,'prediction_rotor_final51_magnitude'),'V5 pure / NN42':(qa,'prediction_rotor_pure_nn'),'42 / mixed':(qa,'prediction_rotor_magnitude'),'V5 pure / NN51':(qa,'prediction_rotor_final51_pure_nn'),'51 / mixed':(qa,'prediction_rotor_final51_magnitude')}
labelmap.update({'V4 pure / NN42':(v4qa,'prediction_rotor_pure_nn'),'V4 pure / NN51':(v4qa,'prediction_rotor_final51_pure_nn')})
for name in ('04_same_quantity_v2_v3_v4_v5','05_pure_neural_and_mixed'):
 for r in allplot[name]:
  q,mid=labelmap[r['label']];m=next(v['metrics'] for v in q['results'] if v['id']==mid and v['partition']=='reused_test')
  check(name+'/'+r['label']+' all4 metrics',all(np.isclose(float(r[k]),m[k],rtol=2e-12,atol=1e-15) for k in ('rmse_T','mae_T','max_abs_T','relative_L2_pct')))
from scipy.spatial import ConvexHull
def xy(c):
 c=np.asarray(c);angle=np.deg2rad(c[:,1]);return c[:,0,None]*np.stack((np.cos(angle),np.sin(angle)),axis=1)
chmap={c['id']:c for c in freeze['models']}
for r in allplot['07_case_errors_and_input_coverage']:
 if r['type']=='input':check('coverage input '+r['case_id'],np.allclose([float(r['xI_A']),float(r['yI_A'])],xy([conditionmap[int(r['case_id'])]])[0],rtol=1e-14,atol=1e-13))
 if r['type']=='case':
  mid='prediction_rotor_magnitude' if r['label']=='42 mixed' else 'prediction_rotor_final51_magnitude';ids=chmap[mid]['training_case_ids'];tx=xy([conditionmap[i] for i in ids]);x=xy([conditionmap[int(r['case_id'])]])[0];h=ConvexHull(tx);dd=np.linalg.norm(tx-x,axis=1)
  inside=np.all(h.equations[:,:2]@x+h.equations[:,2]<=1e-8);cm=next(z for z in csvread(FIG/'data'/(mid+'_case_metrics.csv')) if z['case_id']==r['case_id'])
  check('coverage nearest/hull/error '+r['label']+'/'+r['case_id'],np.isclose(float(r['nearest_train_distance_A']),dd.min(),rtol=1e-14) and int(r['nearest_train_case_id'])==ids[int(dd.argmin())] and (r['inside_training_convex_hull']=='True')==inside and all(float(r[k])==float(cm[k]) for k in ('rmse_T','mae_T','max_abs_T','relative_L2_pct')))
hot=csvread(WORK/'new60_opt_v4/diagnostics_train_only/train_hotspot_response.csv');requests=csvread(WORK/'new60_opt_v3/additional_simulation_requests/建议新增工况.csv')
for r in allplot['08_training_risk_and_new_simulation_requests']:
 if r['type']=='risk':
  rr=[h for h in hot if h['point_id']==r['point_id']];rms=np.sqrt(np.mean([float(h['loo_unsigned_mean_residual_T'])**2 for h in rr]));check('training risk '+r['point_id'],len(rr)==42 and float(r['loo_mean_rms_T'])==rms)
 elif r['type']=='request':
  q=next(q for q in requests if q['request_id']==r['request_id']);x=xy([[float(q['I_RMS_A']),float(q['Thet_deg'])]])[0];check('input-only request '+r['request_id'],np.allclose([float(r['xI_A']),float(r['yI_A'])],x,rtol=1e-14,atol=1e-13))
 elif r['type']=='geometry':
  idx=rotorgeo[0].tolist().index(int(r['point_id']));assert np.array_equal(np.array([float(r['x_ref_mm']),float(r['y_ref_mm'])]),rotorgeo[1][idx])
check('all1800 risk background geometry points exact',len([r for r in allplot['08_training_risk_and_new_simulation_requests'] if r['type']=='geometry'])==1800)
check('eight figure groups',len(recipe['figures'])==8)
for s in recipe['figures']:
 name=s['name'];check(name+' source CSV binding',sha(FIG/s['input_csv'])==s['input_sha256'])
 for ext in ('png','pdf','svg'):check(name+'.'+ext,(FIG/(name+'.'+ext)).stat().st_size>1000)
 with Image.open(FIG/(name+'.png')) as im:check(name+' PNG 300dpi',all(abs(d-300)<.1 for d in im.info['dpi']),dict(size=im.size,dpi=im.info['dpi']))
 check(name+' editable SVG text','<text' in (FIG/(name+'.svg')).read_text('utf8'))
name='02_known60_fitting_height_3d';rel=ROOT/'qa/figure_relocation_check';rel.mkdir(exist_ok=True);(rel/'data').mkdir(exist_ok=True)
for n in ('build_figures_v5.py','v2_plot_primitives.py','figure_recipe.json'):shutil.copy2(FIG/n,rel/n)
spec=next(s for s in recipe['figures'] if s['name']==name);shutil.copy2(FIG/spec['input_csv'],rel/spec['input_csv'])
run=subprocess.run([sys.executable,'-B','build_figures_v5.py','--replay-recipe','figure_recipe.json','--output','regenerated','--only',name],cwd=rel,capture_output=True,text=True)
check('isolated CSV-only redraw exit0',run.returncode==0,run.stderr[-1000:]);check('isolated CSV-only PNG is bitexact',sha(rel/'regenerated'/(name+'.png'))==sha(FIG/(name+'.png')))
size=sum(p.stat().st_size for p in FIG.rglob('*') if p.is_file());check('figures under30MB',size<30_000_000,dict(bytes=size))
report=dict(all_checks_pass=True,choices_sha256=recipe['choices_sha256'],figure_recipe_sha256=sha(FIG/'figure_recipe.json'),figure_count=8,format_counts={ext:len(list(FIG.glob('*.'+ext))) for ext in ('png','pdf','svg')},csv_count=len(list((FIG/'data').glob('*.csv'))),total_bytes=size,checks=checks,check_count=len(checks),visual_review=visual,relocation=dict(path='qa/figure_relocation_check',only_csv_and_plot_sources=True,png_bitexact=True),scope='Rendering and independent numeric checks only; no retraining or reselection')
(FIG/'figure_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'qa/figures_v5_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
(FIG/'figure_check.md').write_text('# V5科学图件核验\n\n全部通过。8组PNG/PDF/SVG、35个CSV；各PNG为300dpi，SVG保留可编辑文字。\n\n来源SHA、原始科学QA和冻结SHA一致；9模型的逐case/frame/point四指标独立向量化复算，空间绘图CSV逐行与原精度数组比较，世界坐标旋转核对通过。所有8组图已实际打开检查，修复图注留白和3D残差数量级标注。完整45°测量扇区和最大误差点保留。\n\n仅两份绘图源码、recipe与02号图CSV复制到独立目录重绘，无模型/完整NPZ，PNG逐字节一致。详见figure_check.json。\n',encoding='utf8')
print(json.dumps({k:report[k] for k in ('all_checks_pass','check_count','figure_count','csv_count','total_bytes')},ensure_ascii=False))
