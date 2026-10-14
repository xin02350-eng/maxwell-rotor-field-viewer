"""Bind all V4 recipes before the one final reused-test replay.

This program hashes weights and reads development JSON only, never magnetic
labels. Existing V3 fitted models/scientific arrays are immutable references.
"""
from pathlib import Path
from datetime import datetime,timezone
import copy, hashlib, json, os
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
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
    for side in [p.with_suffix(p.suffix+'.sha256'),p.with_suffix('.sha256')]:side.write_text(sha(p)+'\n',encoding='ascii')
def binding(p):return dict(relative_path=p,sha256=sha(ROOT/p))
def model_binding(row):
    for k,h in [('model_relative_path','model_sha256'),('inference_module_relative_path','inference_module_sha256')]:row[h]=sha(ROOT/row[k])
    return row
def rebuild51():
    """Rebind two previously selected global recipes, without label access."""
    parent=ROOT/'final51_refit';out=parent/'fixed_global_blends';out.mkdir(exist_ok=False)
    parts={
        'gp':(parent/'kernel_rank24/model.npz',parent/'kernel_final51.py',{'return_signed':True}),
        'nn2':(parent/'final51_regime_sigmoid_affine2_principal_seed42/model.npz',parent/'regime_final51.py',{'return_signed':True}),
        'nn3':(parent/'final51_regime_sigmoid_affine3_principal_seed42/model.npz',parent/'regime_final51.py',{'return_signed':True})}
    results=read(ROOT/'global_field_blend/results.json')
    for label,key in [('primary','final51_fixed_recipe'),('pure_nn','peak_best')]:
        recipe=results['chosen'][key]
        if label=='pure_nn' and recipe['weights']!={'nn2':.25,'nn3':.75}:raise ValueError('Predeclared51 pure recipe changed')
        components=[]
        for name,w in recipe['weights'].items():
            mp,sp,kw=parts[name]
            components.append(dict(name=name,weight=w,kwargs=kw,model_relative_path=os.path.relpath(mp,out).replace('\\','/'),module_relative_path=os.path.relpath(sp,out).replace('\\','/'),model_sha256=sha(mp),module_sha256=sha(sp)))
        meta=dict(format='v4_whole_field_finite_blend',training_scope='final51',training_case_count=51,blend_route=recipe['blend_route'],components=components,
            source_sha256=sha(ROOT/'global_field_blend.py'),recipe_selected_on_original42_validation=True,
            recipe_selection_metrics_from42_model=recipe['validation'],recipe_source_sha256=sha(ROOT/'global_field_blend/results.json'),
            no_independent_validation_for51=True,test_labels_used=False)
        mp=out/(label+'_model.npz');np.savez_compressed(mp,metadata_json=np.array(json.dumps(meta)))
        mp.with_suffix('.npz.sha256').write_text(sha(mp)+'\n',encoding='ascii')
    write(out/'pretest_recipe_binding.json',dict(selected_original42_recipe_source_sha256=sha(ROOT/'global_field_blend/results.json'),
        training_scope='final51',no_test_labels_read=True,primary_weights={'gp':.5,'nn2':.5},pure_weights={'nn2':.25,'nn3':.75}))
def main():
    if (ROOT/'both_versions_choices_frozen.json').exists():raise FileExistsError('V4 already frozen')
    required=['code/inference_numpy_v4.py','code/predict_v4.py','code/scope_utils_v4.py','point_regime_nn/frozen_directory_manifest.json','radial_saturation_nn/final_manifest.json','final51_refit/final51_models_frozen.json','final_selection_validation_only.json']
    for p in required:
        if not (ROOT/p).exists():raise FileNotFoundError(p)
    selection=read(ROOT/'final_selection_validation_only.json')
    rebuild51()
    old=read(WORK/'new60_opt_v3/both_versions_choices_frozen.json');split=read(WORK/'new60_20261014/cache_v1/split.json')
    split_sha=sha(WORK/'new60_20261014/cache_v1/split.json');original51=split['train']+split['validation']
    def oldrow(cid,scope):
        row=copy.deepcopy(next(r for r in old['models'] if r['id']==cid))
        for key in ['model_relative_path','inference_module_relative_path','model_source_module_relative_path']:
            if key in row:row[key]='../new60_opt_v3/'+row[key]
        for p in row.get('provenance_files',[]):p['relative_path']='../new60_opt_v3/'+p['relative_path']
        row.update(training_scope=scope,scientific_source_relative_root='new60_opt_v3',expected_scientific_choices_sha256=sha(WORK/'new60_opt_v3/both_versions_choices_frozen.json'),scientific_model_id=cid,
            reused_test_case_ids=split['test'],validation_case_ids=split['validation'] if scope=='original42' else [],
            scientific_partitions=['validation','reused_test'] if scope=='original42' else ['training_fit'])
        return row
    base=oldrow('prediction_rotor_v2_kernel','original42')
    def newrow(cid,label,scope,mp,family,note,role='primary',source='global_field_blend.py'):
        row=copy.deepcopy(base)
        for key in ['model_source_module_relative_path','model_source_module_sha256','source_quantity','output_transform','scientific_source_relative_root','expected_scientific_choices_sha256','scientific_model_id']:row.pop(key,None)
        if note.startswith('100%×oldmixed42'):note='50%核模型磁密大小+50%两门真实神经网络磁密大小；完整验证9选择同一个全局比例。'
        if note.startswith('100%×oldpure42'):note='25%两门+75%三门真实神经网络磁密大小；完整验证9选择同一个全局比例。'
        row.update(id=cid,display_label=label,training_scope=scope,model_relative_path=mp,inference_module_relative_path=source,
            inference_function='predict',predict_kwargs={},family=family,method='genuine_neural_ensemble' if family=='neural' else 'mixed_kernel_neural',display_note=note,role=role,
            train_case_ids=split['train'] if scope=='original42' else original51,training_case_ids=split['train'] if scope=='original42' else original51,
            validation_case_ids=split['validation'] if scope=='original42' else [],reused_test_case_ids=split['test'],
            scientific_partitions=['validation','reused_test'] if scope=='original42' else ['reused_test'],
            cache_sha256=sha(WORK/'new60_20261014/cache_v1/rotor.npz'),split_sha256=split_sha)
        row['provenance_files']=[binding('global_field_blend/protocol.json'),binding('global_field_blend/results.json'),binding('final_selection_validation_only.json')]
        if scope=='original42':row['validation_selection_metrics']=selection['primary' if role=='primary' else 'pure_nn']['validation']
        else:
            row.pop('validation_selection_metrics',None);row['no_independent_validation']=True
            row['recipe_selected_on_original42_validation']=True;row['provenance_files'] += [binding('final51_refit/protocol.json'),binding('final51_refit/final51_models_frozen.json')]
        return model_binding(row)
    rows=[newrow('prediction_rotor_magnitude','转子 42训练工况优化预测','original42',selection['primary']['model_relative_path'],selection['primary']['family'],selection['primary']['note'],source=selection['primary']['inference_module_relative_path']),
        newrow('prediction_rotor_pure_nn','转子 42训练工况纯神经网络','original42',selection['pure_nn']['model_relative_path'],'neural',selection['pure_nn']['note'],'control',source=selection['pure_nn']['inference_module_relative_path']),
        newrow('prediction_rotor_final51_magnitude','转子 51训练工况最终预测','final51','final51_refit/fixed_global_blends/primary_model.npz','mixed','原42训练+9验证全部参加训练；50%核模型+50%真实神经网络磁密大小；仅原9测试工况可作预测误差评估。','control'),
        newrow('prediction_rotor_final51_pure_nn','转子 51训练工况纯神经网络','final51','final51_refit/fixed_global_blends/pure_nn_model.npz','neural','原42训练+9验证全部参加训练；固定25%两门+75%三门NN磁密大小；没有独立验证集。','control')]
    for cid in ['prediction_stator_magnitude','prediction_airgap_Br']:rows.append(model_binding(oldrow(cid,'original42')))
    support=[binding(p) for p in ['code/inference_numpy_v4.py','code/predict_v4.py','code/scope_utils_v4.py','global_field_blend.py','point_regime_nn/regime_point_nn.py','final51_refit/regime_final51.py','final51_refit/kernel_final51.py','../new60_opt_v2/kernel_route/kernel_model.py','../new60_opt_v2/kernel_route/kernel_magnitude.py','../new60_opt_v3/calibrate_fitting_head.py']]
    for part in selection.get('support_files',[]):
        if part['relative_path'] not in [r['relative_path'] for r in support]:support.append(binding(part['relative_path']))
    common=dict(version='NEW60_OPT_V4_PREDICTION',frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(ROOT/'protocol.json'),split_relative_path='../new60_20261014/cache_v1/split.json',split_sha256=split_sha,
        previous_test_already_seen=True,test_scope='Reused original nine-condition test; not fresh blind validation',test_labels_used_for_selection=False,test_labels_used_for_training=False,test_metrics_computed=False,
        completed_new_neural_trainings=12,failed_gradient_attempts_excluded=2,completed_new_analytic_model_configurations=404,
        completed_global_field_blend_recipes=135,completed_training_LOO_shrink_weights=5,
        final51_recipe_selected_before_test=True,final51_has_no_independent_validation=True,support_files=support,models=rows)
    write(ROOT/'prediction_choices_frozen.json',common)
    doc=copy.deepcopy(common)
    for cid in ['fitting_rotor_magnitude','fitting_stator_magnitude','fitting_airgap_Br']:
        row=oldrow(cid,'full60');row.update(train_case_ids=list(range(60)),training_case_ids=list(range(60)),reused_test_case_ids=[],reuse_existing_frozen_full60_fit=True)
        doc['models'].append(model_binding(row))
    doc.update(version='NEW60_OPT_V4_BOTH_VERSIONS',prediction_freeze_sha256=sha(ROOT/'prediction_choices_frozen.json'),
        final_source_full60_fitting_allowed=True,full60_fitting_scores_used_in_prediction_choice=False,
        reused_full60_fitting_updates=3,new_full60_fitting_updates=0)
    write(ROOT/'both_versions_choices_frozen.json',doc)
    print(json.dumps(dict(frozen_models=len(doc['models']),choices_sha256=sha(ROOT/'both_versions_choices_frozen.json'),test_metrics_computed=False)),flush=True)
if __name__=='__main__':main()
