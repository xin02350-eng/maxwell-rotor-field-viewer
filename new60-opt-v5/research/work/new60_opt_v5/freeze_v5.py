"""Freeze one final V5 delivery. No magnetic label arrays are read here."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;V4=WORK/'new60_opt_v4'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    p=Path(p)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for s in [p.with_suffix(p.suffix+'.sha256'),p.with_suffix('.sha256')]:s.write_text(sha(p)+'\n',encoding='ascii')
def binding(p):return dict(relative_path=p,sha256=sha(ROOT/p))
def modelbind(row):
    for k,h in [('model_relative_path','model_sha256'),('inference_module_relative_path','inference_module_sha256')]:row[h]=sha(ROOT/row[k])
    return row
def main():
    for p in ['code/inference_numpy_v5.py','code/predict_v5.py','code/scope_utils_v5.py','magnitude_objective_nn/final_manifest.json','final51_magnitude_refit/final_manifest.json','selection_validation_only.json']:
        if not (ROOT/p).exists():raise FileNotFoundError(p)
    if (ROOT/'prediction_choices_frozen.json').exists():raise FileExistsError('Already frozen')
    prior=read(V4/'both_versions_choices_frozen.json');selection=read(ROOT/'selection_validation_only.json')
    if selection['selected_new_primary_neural_ids']:raise ValueError('Primary unexpectedlychanged; final51primary plan wouldneedits prespecifiedrefit')
    def oldrow(cid):
        row=copy.deepcopy(next(r for r in prior['models'] if r['id']==cid))
        for key in ['model_relative_path','inference_module_relative_path','model_source_module_relative_path']:
            if key in row:row[key]='../new60_opt_v4/'+row[key]
        for p in row.get('provenance_files',[]):p['relative_path']='../new60_opt_v4/'+p['relative_path']
        if 'scientific_source_relative_root' not in row:
            row.update(scientific_source_relative_root='new60_opt_v4',scientific_model_id=cid,expected_scientific_choices_sha256=sha(V4/'both_versions_choices_frozen.json'))
        row['reuse_existing_frozen_model']=True
        return modelbind(row)
    primary=oldrow('prediction_rotor_magnitude');primary['display_label']='转子 42训练工况混合预测'
    finalprimary=oldrow('prediction_rotor_final51_magnitude');finalprimary['display_label']='转子 51训练工况混合预测'
    pure=oldrow('prediction_rotor_pure_nn')
    for key in ['scientific_source_relative_root','scientific_model_id','expected_scientific_choices_sha256']:pure.pop(key,None)
    pure.update(model_relative_path=selection['pure_nn']['model_relative_path'],inference_module_relative_path='select_magnitude_recipes.py',
        validation_selection_metrics=selection['pure_nn']['validation'],display_label='转子 42训练工况直接磁密目标纯神经网络',
        display_note='75%直接磁密目标微调两门NN+25%旧纯NN；同一全局比例；真实梯度微调，没有核模型。',reuse_existing_frozen_model=False,
        provenance_files=[binding('magnitude_objective_nn/candidate_protocol.json'),binding('magnitude_objective_nn/final_manifest.json'),binding('magnitude_recipes/protocol.json'),binding('magnitude_recipes/results.json'),binding('selection_validation_only.json')])
    pure=modelbind(pure)
    refit=read(ROOT/'final51_magnitude_refit/final_manifest.json')
    if len(refit['models'])!=1:raise ValueError('Exactlyone51fine-tune permitted')
    new51=refit['models'][0]
    fpure=oldrow('prediction_rotor_final51_pure_nn')
    out=ROOT/'final51_fixed_pure_recipe';out.mkdir(exist_ok=False)
    original=next(r for r in prior['models'] if r['id']=='prediction_rotor_final51_pure_nn')
    parts=[dict(name='V4pure51',weight=.25,quantity='|B|',kwargs={},model_relative_path=os.path.relpath(V4/original['model_relative_path'],out).replace('\\','/'),module_relative_path=os.path.relpath(V4/original['inference_module_relative_path'],out).replace('\\','/'),model_sha256=original['model_sha256'],module_sha256=original['inference_module_sha256']),
        dict(name='direct_magnitude_nn51',weight=.75,quantity='Br,Bt',kwargs=new51.get('signed_output_kwargs',{'return_signed':True}),
            model_relative_path=os.path.relpath(ROOT/'final51_magnitude_refit'/new51['model_relative_path'],out).replace('\\','/'),module_relative_path=os.path.relpath(ROOT/'final51_magnitude_refit'/new51['inference_module_relative_path'],out).replace('\\','/'),model_sha256=new51['model_sha256'],module_sha256=new51['inference_module_sha256'])]
    mp=out/'model.npz';np.savez_compressed(mp,metadata_json=np.array(json.dumps(dict(format='V5fixedpure51recipe',components=parts,blend_route='scalar_magnitude',training_scope='final51',selection='Same25/75pure42 validation-only recipe, no51validation',test_labels_used=False))))
    mp.with_suffix('.npz.sha256').write_text(sha(mp)+'\n',encoding='ascii')
    for key in ['scientific_source_relative_root','scientific_model_id','expected_scientific_choices_sha256']:fpure.pop(key,None)
    fpure.update(model_relative_path='final51_fixed_pure_recipe/model.npz',inference_module_relative_path='select_magnitude_recipes.py',
        display_label='转子 51训练工况直接磁密目标纯神经网络',display_note='原验证9也参加训练；75%直接磁密目标NN51+25%旧纯NN51；25/75由原42验证选定，无独立验证集。',
        reuse_existing_frozen_model=False,provenance_files=[binding('final51_pure_recipe_extension_pretest.json'),binding('final51_magnitude_refit/final_manifest.json')])
    fpure=modelbind(fpure)
    rows=[primary,pure,finalprimary,fpure,oldrow('prediction_stator_magnitude'),oldrow('prediction_airgap_Br')]
    support=[binding(p) for p in ['code/inference_numpy_v5.py','code/predict_v5.py','code/scope_utils_v5.py','select_magnitude_recipes.py','magnitude_objective_nn/magnitude_point_nn.py',
        'final51_magnitude_refit/'+new51['inference_module_relative_path'],'../new60_opt_v4/radial_combo.py','../new60_opt_v4/global_field_blend.py','../new60_opt_v4/point_regime_nn/regime_point_nn.py',
        '../new60_opt_v4/final51_refit/regime_final51.py','../new60_opt_v4/final51_refit/kernel_final51.py','../new60_opt_v2/kernel_route/kernel_model.py','../new60_opt_v2/kernel_route/kernel_magnitude.py','../new60_opt_v3/calibrate_fitting_head.py']]
    doc=dict(version='NEW60_OPT_V5_PREDICTION',frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),
        previous_test_already_seen=True,test_scope='Retrospective reused original9 test; not freshblind',test_labels_used_for_training=False,test_labels_used_for_selection=False,test_metrics_computed=False,
        previous_V4_choices_sha256=sha(V4/'both_versions_choices_frozen.json'),split_relative_path='../new60_20261014/cache_v1/split.json',split_sha256=prior['split_sha256'],
        completed_new_gradient_finetuning_runs=5,completed_new_fromscratch_neural_trainings=0,failed_finetuning_attempts=0,
        completed_original42_finetunes=4,completed_final51_finetunes=1,new_globalfield_combination_recipes=60,
        final51_recipe_selected_before_test=True,final51_has_no_independent_validation=True,support_files=support,models=rows)
    write(ROOT/'prediction_choices_frozen.json',doc)
    both=copy.deepcopy(doc)
    for cid in ['fitting_rotor_magnitude','fitting_stator_magnitude','fitting_airgap_Br']:both['models'].append(oldrow(cid))
    both.update(version='NEW60_OPT_V5_BOTH_VERSIONS',prediction_freeze_sha256=sha(ROOT/'prediction_choices_frozen.json'),
        final_source_full60_fitting_allowed=True,full60_fitting_scores_used_in_prediction_choice=False,reused_full60_fitting_updates=3,new_full60_fitting_updates=0)
    write(ROOT/'both_versions_choices_frozen.json',both)
    print(json.dumps(dict(models=len(both['models']),choices_sha256=sha(ROOT/'both_versions_choices_frozen.json'),test_metrics_computed=False)),flush=True)
if __name__=='__main__':main()
