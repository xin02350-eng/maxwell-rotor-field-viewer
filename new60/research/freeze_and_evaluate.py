"""Freeze all regional validation choices, then explicitly evaluate held cases.

Two separate invocations: --freeze-all and --evaluate-all. No test-based tuning.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
import experiment_core as core
import hybrid_cycle_rbf as hybrid
from stage2_local import acceptance
from run_new60 import group_metrics

ROOT=Path(__file__).resolve().parent.parent
REGIONS=('stator','rotor','airgap')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def frozen(path):
    assert digest(path)==path.with_suffix('.sha256').read_text(encoding='ascii').strip(),path
    return read(path)


def verify_hybrid(directory):
    info=frozen(directory/'selection_frozen.json')
    assert digest(directory/'run_identity.json')==info['identity_sha256']
    for name,expected in info['artifact_sha256'].items():
        assert digest(directory/name)==expected
        hybrid.load_model(directory/name,expected_sha256=expected)
    return info


def neural_parameters(model):
    count=sum(model[f'w{i}'].size+model[f'b{i}'].size for i in range(int(model['layer_count'])))
    if int(model.get('has_base',0)):
        count+=neural_parameters({key[6:]:value for key,value in model.items() if key.startswith('base__')})
    return int(count)


def freeze_all():
    assert not (ROOT/'all_regions_selection_frozen.json').exists()
    plans={}
    for region in REGIONS:
        base=ROOT/'experiments'/region
        first=frozen(base/'stage1/selection_frozen.json')
        second=frozen(base/'stage2/selection_frozen.json')
        for stage,info in (('stage1',first),('stage2',second)):
            assert digest(base/stage/'run_identity.json')==info['identity_sha256']
            for row in info['rows']:
                assert digest(base/stage/row['model_relative_path'])==row['model_sha256']
        assert first['candidate_test_predictions_used_for_selection'] is False
        assert second['candidate_test_predictions_used_for_selection'] is False
        local=next(r for r in second['rows'] if r['name']==second['selected'])
        checks=[]
        for seed in (42,17,73):
            h=verify_hybrid(base/f'hybrid_seed{seed}')
            assert h['test_metrics_computed'] is False and h['seed']==seed
            control=next(r for r in second['rows'] if r['name']==f'uniform_field_control_seed{seed}')
            comparison=acceptance(h['selected_hybrid']['hybrid_validation'],control['validation'])
            checks.append(dict(seed=seed,**comparison))
        candidate=frozen(base/'hybrid_seed42/selection_frozen.json')
        stable=all(c['passed'] for c in checks)
        # Hybrid validation also must satisfy safeguards versus any stable local NN.
        local_gate=acceptance(candidate['selected_hybrid']['hybrid_validation'],local['validation'])
        choose_hybrid=stable and local_gate['passed']
        chosen=(base/'hybrid_seed42/model.npz') if choose_hybrid else (base/'stage2'/local['model_relative_path'])
        control_path=base/'stage2'/second['control_model_relative_path']
        plan=dict(dataset='REAL_NEW60_20261010',region=region,quantity=second['quantity'],
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            selected_name='mlp_cycle_rbf_seed42' if choose_hybrid else local['name'],
            selected_family='POD_MLP_condition_RBF_residual' if choose_hybrid else local['family'],
            selected_model_relative_path=chosen.relative_to(ROOT).as_posix(),selected_model_sha256=digest(chosen),
            neural_model_relative_path=(base/'stage2/uniform_field_control_seed42/best_model.npz').relative_to(ROOT).as_posix()
                if choose_hybrid else chosen.relative_to(ROOT).as_posix(),
            control_model_relative_path=control_path.relative_to(ROOT).as_posix(),control_model_sha256=digest(control_path),
            selected_validation=candidate['selected_hybrid']['hybrid_validation'] if choose_hybrid else local['validation'],
            hybrid_matched_seed_checks=checks,hybrid_stable_vs_uniform=stable,hybrid_vs_local_gate=local_gate,
            local_recipe_stable=second['stable_validation_improvement'],
            neural_contribution_vs_RBF_only=candidate['neural_contribution'],
            selected_by='validation only; all tests pending',candidate_test_predictions_used_for_selection=False,
            new_splitIDs=read(ROOT/'cache_v1/split.json')['original_ids'],
            historical_round4_reproduced=False,source_split_not_original_round4=True,
            comparison_scope='new 42/9/9 interpolation benchmark, fixed 60 time frames and fixed sampled spatial points',
            cache_sha256=digest(ROOT/'cache_v1'/f'{region}.npz'),split_sha256=digest(ROOT/'cache_v1/split.json'),
            prerequisite_freezes={str(p.relative_to(ROOT)):digest(p) for p in (
                base/'stage1/selection_frozen.json',base/'stage2/selection_frozen.json',
                *(base/f'hybrid_seed{s}/selection_frozen.json' for s in (42,17,73)))})
        plans[region]=plan
    # All regions were decided before any final test is computed.
    for region,plan in plans.items():
        base=ROOT/'experiments'/region
        write_new(base/'final_choice.json',plan)
        write_new(base/'selection_frozen.json',plan)
        (base/'selection_frozen.sha256').write_text(digest(base/'selection_frozen.json')+'\n',encoding='ascii')
        identity=read(base/'stage1/run_identity.json')
        identity['consolidated_final_choice_sha256']=digest(base/'final_choice.json')
        write_new(base/'run_identity.json',identity)
    write_new(ROOT/'all_regions_selection_frozen.json',dict(plans=plans,test_metrics_computed=False,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_code_sha256={name:digest(Path(__file__).parent/name) for name in (
            'freeze_and_evaluate.py','experiment_core.py','hybrid_cycle_rbf.py','stage2_local.py','run_new60.py')},
        protocol_sha256=digest(ROOT/'study_protocol.json'),
        choices_sha256={r:digest(ROOT/'experiments'/r/'final_choice.json') for r in REGIONS}))
    (ROOT/'all_regions_selection_frozen.sha256').write_text(digest(ROOT/'all_regions_selection_frozen.json')+'\n',encoding='ascii')
    print(json.dumps({r:p['selected_name'] for r,p in plans.items()},ensure_ascii=False),flush=True)


def model_descriptors(region):
    base=ROOT/'experiments'/region
    rows=[]
    for stage in ('stage1','stage2'):
        info=frozen(base/stage/'selection_frozen.json')
        for raw in info['rows']:
            path=base/stage/raw['model_relative_path']
            assert digest(path)==raw['model_sha256']
            row=dict(raw,name=f'{stage}/{raw["name"]}',family=raw.get('family','architecture'),kind='mlp',
                model_relative_path=path.relative_to(ROOT).as_posix(),model_sha256=digest(path),
                history_relative_path=f'{stage}/{raw["name"]}/history.csv')
            rows.append(row)
    for seed in (42,17,73):
        info=verify_hybrid(base/f'hybrid_seed{seed}')
        path=base/f'hybrid_seed{seed}/model.npz'
        neural=read(base/f'stage2/uniform_field_control_seed{seed}/summary.json')
        rows.append(dict(name=f'mlp_cycle_rbf_seed{seed}',family='POD_MLP_condition_RBF_residual',kind='hybrid',seed=seed,
            model_relative_path=path.relative_to(ROOT).as_posix(),model_sha256=digest(path),analytic_epochs=0,
            validation=info['selected_hybrid']['hybrid_validation'],
            rbf_parameters={key:info['selected_hybrid'][key] for key in ('lengthscale','ridge')},
            neural_parameter_count=neural['parameter_count'],neural_best_epoch=neural['best_epoch'],
            neural_final_epoch=neural['final_epoch'],neural_training_seconds=neural['elapsed_seconds'],
            analytic_fit_seconds=None,analytic_time_note='Per-run analytic wall time was not recorded; null is not zero.',
            neural_contribution=info['neural_contribution']))
    info=verify_hybrid(base/'hybrid_seed42')
    path=base/'hybrid_seed42/rbf_only_model.npz'
    rows.append(dict(name='best_independent_RBF_only',family='POD_RBF_only',kind='rbf_only',seed=42,
        model_relative_path=path.relative_to(ROOT).as_posix(),model_sha256=digest(path),analytic_epochs=0,
        rbf_parameters={key:info['best_rbf_only'][key] for key in ('lengthscale','ridge')},
        analytic_fit_seconds=None,
        validation=info['best_rbf_only']['rbf_only_validation']))
    return rows


def evaluate_all():
    whole=frozen(ROOT/'all_regions_selection_frozen.json')
    assert digest(ROOT/'study_protocol.json')==whole['protocol_sha256']
    for name,h in whole['source_code_sha256'].items():assert digest(Path(__file__).parent/name)==h
    for r in REGIONS:
        p=ROOT/'experiments'/r/'final_choice.json'
        assert digest(p)==whole['choices_sha256'][r]
        plan=read(p)
        assert digest(ROOT/plan['selected_model_relative_path'])==plan['selected_model_sha256']
        assert digest(ROOT/plan['control_model_relative_path'])==plan['control_model_sha256']
        assert digest(ROOT/'cache_v1'/f'{r}.npz')==plan['cache_sha256']
        assert digest(ROOT/'cache_v1/split.json')==plan['split_sha256']
        for name,h in plan['prerequisite_freezes'].items():assert digest(ROOT/name)==h
        assert not (ROOT/'experiments'/r/'final_evaluation').exists()
    with threadpool_limits(limits=4):
        for region in REGIONS:
            base=ROOT/'experiments'/region;plan=read(base/'final_choice.json')
            cache=core.load_cache(ROOT/'cache_v1'/f'{region}.npz');split=read(ROOT/'cache_v1/split.json')
            meta=json.loads(str(cache['metadata_json']))
            tr=np.asarray(split['test']);va=np.asarray(split['validation'])
            truth=cache['field'][tr].astype(np.float64)
            stage2=frozen(base/'stage2/selection_frozen.json')
            assert digest(base/'stage2/fixed_training_masks.npz')==stage2['hotspot_sha256']
            with np.load(base/'stage2/fixed_training_masks.npz',allow_pickle=False) as data:
                hotspot=data['hotspot_mask']
                assert np.array_equal(data['point_id'],cache['point_id'])
            output=base/'final_evaluation';output.mkdir(exist_ok=False)
            rows=model_descriptors(region)
            for row in rows:
                model_path=ROOT/row['model_relative_path']
                assert digest(model_path)==row['model_sha256']
                if row['kind']=='mlp':
                    pred=core.predict(model_path,cache['features'][tr])
                    valpred=core.predict(model_path,cache['features'][va])
                else:
                    pred=hybrid.predict(model_path,cache['conditions'][tr])
                    valpred=hybrid.predict(model_path,cache['conditions'][va])
                old_validation=(row['validation']['global_metrics']['rmse_T'] if 'validation' in row
                    else row['validation_rmse_T'])
                row['validation']=group_metrics(valpred,cache['field'][va],cache['case_ids'][va],hotspot,meta['quantity'])
                assert np.isclose(old_validation,row['validation']['global_metrics']['rmse_T'],rtol=1e-6,atol=1e-9)
                row['test']=group_metrics(pred,truth,cache['case_ids'][tr],hotspot,meta['quantity'])
                row['test']['geometric_hotspot']=group_metrics(pred,truth,cache['case_ids'][tr],cache['dataset_hotspot'],meta['quantity'])
                # Keep neural/RBF compute and coefficients separate in the report.
                with np.load(model_path,allow_pickle=False) as z:
                    row['saved_model_bytes']=model_path.stat().st_size
                    model={k:z[k] for k in z.files}
                    if row['kind']=='mlp':
                        row['total_neural_parameter_count']=neural_parameters(model)
                        row['active_RBF_weight_count']=0
                    elif row['kind']=='hybrid':
                        row['total_neural_parameter_count']=neural_parameters({k[5:]:v for k,v in model.items() if k.startswith('mlp__')})
                        row['active_RBF_weight_count']=int(model['residual_weights'].size)
                    else:
                        row['total_neural_parameter_count']=0
                        row['active_RBF_weight_count']=int(model['rbf_only_weights'].size)
                    row['stored_RBF_weight_count']=sum(int(model[k].size) for k in ('residual_weights','rbf_only_weights') if k in model)
                    row['total_active_fitted_parameter_count']=row['total_neural_parameter_count']+row['active_RBF_weight_count']
                    row['parameter_count_note']='Active neural weights and Gaussian RBF target weights; POD basis and fixed transforms are separately stored. Pure-RBF artifact also stores inactive comparison branches.'
                selected= row['model_relative_path']==plan['selected_model_relative_path']
                control= row['model_relative_path']==plan['control_model_relative_path']
                if selected or control:
                    for role in (['selected'] if selected else [])+(['control'] if control else []):
                        save_prediction(output/f'{role}_predictions.npz',pred,truth,cache,tr,hotspot,region,row['family'],row['model_sha256'])
                print(json.dumps(dict(region=region,name=row['name'],test_rmse_T=row['test']['global_metrics']['rmse_T']),ensure_ascii=False),flush=True)
            oracle=cache['mean']+cache['coeff'][tr].astype(np.float64)@cache['basis'].astype(np.float64).T
            pod=group_metrics(oracle,truth,cache['case_ids'][tr],hotspot,meta['quantity'])
            save_prediction(output/'pod_predictions.npz',oracle,truth,cache,tr,hotspot,region,'POD_truth_projection_diagnostic',None)
            selected_row=next(row for row in rows if row['model_relative_path']==plan['selected_model_relative_path'])
            report=dict(dataset='REAL_NEW60_20261010',region=region,quantity=meta['quantity'],selected_name=plan['selected_name'],
                selected_row_name=selected_row['name'],rows=rows,pod=pod,
                pod_metrics=dict(validation=meta['oracle']['validation'],test=pod),new_splitIDs=plan['new_splitIDs'],
                artifact_sha256={p.name:digest(p) for p in output.glob('*_predictions.npz')},
                final_choice_sha256=digest(base/'final_choice.json'),test_not_used_for_selection=True,
                historical_round4_reproduced=False,protocol=read(ROOT/'study_protocol.json'))
            write_new(output/'metrics.json',report)


def save_prediction(path,prediction,truth,cache,cases,hotspot,region,version,model_hash):
    meta=json.loads(str(cache['metadata_json']))
    meta.update(split_role='test',model_version=version,model_sha256=model_hash)
    np.savez_compressed(path,truth=truth,prediction=prediction,case_ids=cache['case_ids'][cases],
        point_id=cache['point_id'],xy=cache['xy'],theta=cache['theta'],time_s=cache['time_s'],
        solver_index=cache['solver_index'],theta_m_relative_deg=cache['theta_m_relative_deg'],
        hotspot_mask=hotspot,dataset_hotspot=cache['dataset_hotspot'],quantity=np.asarray(meta['quantity']),
        model_version=np.asarray(version),metadata_json=json.dumps(meta,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-all',action='store_true');group.add_argument('--evaluate-all',action='store_true')
    args=parser.parse_args()
    freeze_all() if args.freeze_all else evaluate_all()
