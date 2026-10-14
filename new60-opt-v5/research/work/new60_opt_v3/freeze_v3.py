"""Create the immutable V3 registry before evaluation; never reads numeric labels."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, json

ROOT=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for side in [p.with_suffix(p.suffix+'.sha256'),p.with_suffix('.sha256')]:side.write_text(digest(p)+'\n',encoding='ascii')
def binding(path):return dict(relative_path=path,sha256=digest(ROOT/path))
def bind_model(row):
    for k,h in [('model_relative_path','model_sha256'),('inference_module_relative_path','inference_module_sha256')]:row[h]=digest(ROOT/row[k])
    return row
def main(include_fit):
    if not include_fit:
        protocol=read(ROOT/'protocol.json')
        protocol['version']='new60-opt-v3-20261015'
        protocol['fitting_branch']='After the V3 prediction freeze only, independently optimize the existing all60-trained MLP linear output layer by supervised least squares on all60 fitting targets. Hidden neural layers retained; no fitting labels or scores feed prediction selection.'
        (ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        old=read(ROOT/'../new60_opt_v2/both_versions_choices_frozen.json')
        def old_row(old_id,new_id,label,family):
            row=copy.deepcopy(next(r for r in old['models'] if r['id']==old_id))
            for k in ['model_relative_path','inference_module_relative_path','model_source_module_relative_path']:
                if k in row:row[k]='../new60_opt_v2/'+row[k]
            for item in row.get('provenance_files',[]):item['relative_path']='../new60_opt_v2/'+item['relative_path']
            row.update(id=new_id,display_label=label,family=family)
            return row
        baseline=old_row('prediction_rotor_magnitude','prediction_rotor_v2_kernel','转子 V2 核模型对照','kernel')
        baseline['role']='control'
        hybrid=copy.deepcopy(baseline)
        hybrid.update(id='prediction_rotor_magnitude',role='primary',method='mixed_kernel_neural',family='mixed',display_label='转子 V3 神经网络与核模型组合',
            model_relative_path='hybrid_gated_control/selected_model.npz',inference_module_relative_path='hybrid_gated_control.py',
            display_note='75% V3纯神经网络 + 25% V2核模型；验证集选择固定全局比例；不是纯NN。',
            validation_selection_metrics=read(ROOT/'hybrid_gated_control/best.json')['validation'])
        for k in ['model_source_module_relative_path','model_source_module_sha256','source_quantity','output_transform']:hybrid.pop(k,None)
        hybrid['provenance_files']=[binding('hybrid_gated_control/protocol.json'),binding('hybrid_gated_control/results.json')]
        pure=copy.deepcopy(hybrid)
        pure.update(id='prediction_rotor_gated_nn',role='control',method='genuine_neural_ensemble',family='neural',
            display_label='转子 V3 纯神经网络',display_note='已训练的多个神经网络组合；固定训练LOO风险掩码；不是查询真值。',
            model_relative_path='gated_neural_ensemble/selected_model.npz',inference_module_relative_path='gated_neural_ensemble.py')
        sel=read(ROOT/'gated_neural_ensemble/selection_validation_only.json')
        pure['validation_selection_metrics']=sel
        pure['provenance_files']=[binding('gated_neural_ensemble/candidate_protocol.json'),binding('gated_neural_ensemble/selection_validation_only.json'),binding('gated_neural_ensemble/train_risk_masks_frozen.json')]
        stator=old_row('prediction_stator_magnitude','prediction_stator_magnitude','定子 V2 核模型沿用','kernel')
        gap=old_row('prediction_airgap_Br','prediction_airgap_Br','气隙 V2 核模型沿用','kernel')
        rows=[bind_model(r) for r in [hybrid,stator,gap,pure,baseline]]
        support=[binding(p) for p in ['code/inference_numpy_v3.py','code/predict_v3.py','calibrate_fitting_head.py','../new60_opt_v2/kernel_route/kernel_model.py','../new60_opt_v2/kernel_route/kernel_magnitude.py','gated_neural_ensemble.py']]
        doc=dict(version='NEW60_OPT_V3_PREDICTION',frozen_utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=digest(ROOT/'protocol.json'),test_labels_used_for_selection=False,test_labels_used_for_training=False,test_metrics_computed=False,
            previous_test_already_seen=True,test_scope='Reused nine-condition test, not a fresh blind evaluation',
            split_relative_path='../new60_20261014/cache_v1/split.json',split_sha256=old['split_sha256'],
            selection='Original-T validation9 complete-field RMSE; all selected recipe choices completed before test replay. V2 stator and gap retained. No test-based reselection.',
            completed_new_neural_trainings=13,interrupted_new_neural_trainings_excluded=1,
            output_layer_closed_form_updates_after_this_freeze=3,
            support_files=support,models=rows)
        write(ROOT/'prediction_choices_frozen.json',doc)
    else:
        doc=read(ROOT/'prediction_choices_frozen.json')
        if digest(ROOT/'prediction_choices_frozen.json')!=(ROOT/'prediction_choices_frozen.json.sha256').read_text().strip():raise ValueError('Prediction freeze mismatch')
        old=read(ROOT/'../new60_opt_v2/both_versions_choices_frozen.json')
        for region,quantity in [('rotor','planar_magnitude'),('stator','planar_magnitude'),('airgap','Br')]:
            result=read(ROOT/f'fitting_full60/{region}/result.json')
            oldfit=next(r for r in old['models'] if r['branch']=='fitting_full60' and r['region']==region)
            row=dict(id=f'fitting_{region}_'+('Br' if region=='airgap' else 'magnitude'),branch='fitting_full60',region=region,
                quantity=quantity,role='primary',method='trained_MLP_linear_head_calibration',family='neural',
                display_label={'rotor':'转子','stator':'定子','airgap':'气隙'}[region]+' V3 全60工况神经网络拟合',
                display_note='全部60工况均参与训练；已有MLP隐藏层+最后线性层监督最小二乘校准；训练内拟合误差不是预测误差。',
                model_relative_path=f'fitting_full60/{region}/model.npz',inference_module_relative_path='calibrate_fitting_head.py',
                inference_function='predict',predict_kwargs={},cache_sha256=oldfit['cache_sha256'],split_sha256=doc['split_sha256'],
                training_case_ids=list(range(60)),train_case_ids=list(range(60)),unseen_accuracy_claim=False,
                fitting_metrics=result['fitting_deployed'],fitting_raw_metrics=result['fitting_raw'],
                prediction_freeze_sha256=digest(ROOT/'prediction_choices_frozen.json'),
                provenance_files=[binding(f'fitting_full60/{region}/result.json')])
            doc['models'].append(bind_model(row))
        doc.update(version='NEW60_OPT_V3_BOTH_VERSIONS',frozen_utc=datetime.now(timezone.utc).isoformat(),
            prediction_freeze_sha256=digest(ROOT/'prediction_choices_frozen.json'),
            final_source_full60_fitting_allowed=True,full60_fitting_scores_used_in_prediction_choice=False)
        write(ROOT/'both_versions_choices_frozen.json',doc)
    print('Frozen: '+('both versions' if include_fit else 'prediction')+'; test metrics not yet computed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-fit',action='store_true');main(p.parse_args().include_fit)
