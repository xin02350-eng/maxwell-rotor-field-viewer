"""Freeze finished validation choices before any V2 reused-test evaluation."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;V1=ROOT.parent/'new60_20261014'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bound(p):return dict(relative_path=str(Path(p).relative_to(ROOT)).replace('\\','/'),sha256=sha(p))
def main():
    split=read(V1/'cache_v1/split.json');k=read(ROOT/'kernel_route/route_choices.json');n=read(ROOT/'neural_route/prediction_choice_frozen.json');models=[]
    def entry(id,region,qty,model,module,role='primary',sourceqty=None,transform='identity',predict_kwargs=None):
        m=dict(id=id,branch='prediction',region=region,quantity=qty,role=role,
          model_relative_path=model,model_sha256=sha(ROOT/model),inference_module_relative_path=module,
          inference_module_sha256=sha(ROOT/module),inference_function='predict',predict_kwargs=predict_kwargs or {},
          cache_sha256=sha(V1/'cache_v1'/f'{region}.npz'),split_sha256=sha(V1/'cache_v1/split.json'),
          train_case_ids=split['original_ids']['train'],validation_case_ids=split['original_ids']['validation'],
          previous_test_already_seen=True,test_labels_used_for_training_or_selection=False,
          source_quantity=sourceqty or qty,output_transform=transform,
          provenance_files=[bound(ROOT/'protocol.json')])
        models.append(m);return m
    for row in k['selected']:
        if row['region']=='rotor' and row['quantity']=='Br,Bt':continue
        qty='planar_magnitude' if row['quantity']=='|B|' else row['quantity']
        m=entry(f"prediction_{row['region']}_{'magnitude' if qty=='planar_magnitude' else 'vector' if qty=='Br,Bt' else 'Br'}",row['region'],qty,
            'kernel_route/'+row['model_relative_path'],'kernel_route/'+row['predict_module'],sourceqty=row.get('source_quantity'),transform=row.get('output_transform','identity'))
        m['validation_selection_metrics']=row['validation'];m['method']='condition_kernel_POD'
        m['model_source_module_relative_path']='kernel_route/kernel_model.py';m['model_source_module_sha256']=sha(ROOT/'kernel_route/kernel_model.py')
        m['provenance_files'] += [bound(ROOT/'kernel_route/route_choices.json'),bound((ROOT/m['model_relative_path']).parent/'provenance.json')]
    vector=next(x for x in n['candidates'] if x['family']=='rotor')
    m=entry('prediction_rotor_vector','rotor','Br,Bt','neural_route/'+vector['model_relative_path'],'neural_route/cycle_neural.py')
    m['validation_selection_metrics']=vector['validation_raw'];m['method']='pure_neural_condition_POD'
    m['provenance_files'] +=[bound(ROOT/'neural_route/prediction_choice_frozen.json')]
    m=entry('prediction_rotor_magnitude_nn','rotor','planar_magnitude','neural_route/'+vector['model_relative_path'],'neural_route/neural_magnitude_from_vector.py','neural_control','Br,Bt','hypot(Br,Bt)')
    m['method']='pure_neural_condition_POD_derived_magnitude';m['model_source_module_relative_path']='neural_route/cycle_neural.py';m['model_source_module_sha256']=sha(ROOT/'neural_route/cycle_neural.py');m['provenance_files'] +=[bound(ROOT/'neural_route/prediction_choice_frozen.json')]
    m=entry('prediction_rotor_magnitude_direct_nn','rotor','planar_magnitude','neural_route/'+n['selected_model_relative_path'],'neural_route/magnitude_cycle_neural.py','neural_control',predict_kwargs={'nonnegative':False})
    m['validation_selection_metrics']=n['validation_raw'];m['method']='pure_neural_condition_POD_direct_magnitude';m['provenance_files'] +=[bound(ROOT/'neural_route/prediction_choice_frozen.json')]
    r=read(ROOT/'neural_kernel_route/rotor_magnitude/ranking.json')[0]
    m=entry('prediction_rotor_magnitude_neural_kernel','rotor','planar_magnitude','neural_kernel_route/rotor_magnitude/'+r['model'].replace('\\','/'),'neural_frame_kernel.py','neural_control')
    m['validation_selection_metrics']=r['validation'];m['method']='learned_neural_coordinate_embedding_kernel'
    m['provenance_files'] += [bound(ROOT/'neural_kernel_route/rotor_magnitude/protocol.json'),bound(ROOT/'neural_kernel_route/rotor_magnitude/representation.npz')]
    p=ROOT/'prediction_choices_frozen.json'
    if p.exists():raise FileExistsError(p)
    frozen=dict(version='NEW60_OPT_V2_PREDICTION_CHOICES',frozen_utc=datetime.now(timezone.utc).isoformat(),
      test_metrics_computed=False,previous_test_already_seen=True,test_scope='Reused nine-condition test; not a fresh blind evaluation',
      final_source_full60_fitting_allowed=True,full60_fitting_scores_used_in_prediction_choice=False,
      protocol_sha256=sha(ROOT/'protocol.json'),split_relative_path='../new60_20261014/cache_v1/split.json',split_sha256=sha(V1/'cache_v1/split.json'),
      selection='Best validation within each quantity; pureNN vector rotor and best kernel cloud alongside independent neural controls. No new test values used.',models=models)
    p.write_text(json.dumps(frozen,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');p.with_suffix('.sha256').write_text(sha(p)+'\n',encoding='ascii')
    print('FROZEN',sha(p),[(m['id'],m['model_relative_path']) for m in models])
if __name__=='__main__':main()
