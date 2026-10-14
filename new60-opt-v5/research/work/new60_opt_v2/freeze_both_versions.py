"""Bind fitting models to the immutable prediction choices; no labels read."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;V1=ROOT.parent/'new60_20261014'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=ROOT/'prediction_choices_frozen.json';pred=json.loads(p.read_text(encoding='utf8'))
    if sha(p)!=p.with_suffix('.sha256').read_text().strip():raise ValueError('Prediction freeze hash mismatch')
    combined=dict(pred);combined['version']='NEW60_OPT_V2_BOTH_VERSIONS';combined['frozen']=True
    combined['frozen_utc']=datetime.now(timezone.utc).isoformat();combined['full60_fitting_authorized']=True
    combined['prediction_freeze_sha256']=sha(p);combined['source_study_relative_path']='../new60_20261014'
    combined['models']=list(pred['models'])
    for region in ('rotor','stator','airgap'):
        base=f'neural_route/fitting_full60/{region}';model=ROOT/base/'model.npz';mod='neural_route/fitting_full60.py'
        entry=dict(id=f'fitting_{region}',branch='fitting_full60',region=region,
          quantity='Br' if region=='airgap' else 'planar_magnitude',role='primary',method='pure_neural_all60_condition_POD',
          model_relative_path=f'{base}/model.npz',model_sha256=sha(model),inference_module_relative_path=mod,
          inference_module_sha256=sha(ROOT/mod),inference_function='predict',predict_kwargs={'nonnegative':True},
          cache_sha256=sha(V1/'cache_v1'/f'{region}.npz'),split_sha256=sha(V1/'cache_v1/split.json'),train_case_ids=list(range(60)),validation_case_ids=[],
          score_scope='training_fit only; all60 source conditions, no holdout score',
          saved_predictions=[dict(partition='training_fit',relative_path=f'{base}/fitting_predictions.npz',sha256=sha(ROOT/base/'fitting_predictions.npz'),prediction_key='prediction',case_ids_key='case_ids')],
          provenance_files=[dict(relative_path=f'{base}/{name}',sha256=sha(ROOT/base/name)) for name in ('protocol.json','representation.npz','result.json')],
          source_bindings=[dict(source_relative_path=mod,record='model_metadata',pointer='/source_sha256')])
        combined['models'].append(entry)
    out=ROOT/'both_versions_choices_frozen.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(combined,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    out.with_suffix('.sha256').write_text(sha(out)+'\n',encoding='ascii')
    print('BOTH VERSIONS FROZEN',sha(out),len(combined['models']))
if __name__=='__main__':main()
