"""Freeze protocol before fitting any real NEW60 model, then prepare caches."""
from pathlib import Path
import hashlib
import json
from prepare_new60 import prepare

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    protocol_path = ROOT / 'study_protocol.json'
    protocol = dict(dataset='REAL_NEW60_20261010', version='NEW60_STUDY_20261014_V1',
        original_round4_available=False, historical_metric_replication=False,
        split=json.loads((ROOT/'inputs_v1/split_source.json').read_text(encoding='utf-8')),
        ranks=dict(stator=192, rotor=192, airgap=64),
        harmonics=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
        angle_kind='absolute mechanical', angle_units='degree',
        features='I_RMS, native Thet, sin(h*absolute_mechanical_angle), cos(h*angle); first-period h1/h2 prevent enforced electrical endpoint equality',
        input_definition='native model RMS, four parallel branches, no /2; Thet is not dq beta',
        basis_and_scalers='training42 only',
        budgets=dict(max_epochs=600, patience=80, learning_rate=1e-3, batch_size=256,
                     activation='tanh', optimizer='Adam', threads_per_process=4),
        architecture_candidates=[[64,64],[128,128],[64,64,64],[128,128,128]],
        controlled_reference=[32,32], seeds=[42,17,73],
        stages=['architecture standardized coefficient MSE',
                'learning rate 0.0003 changes only LR',
                'uniform physical field MSE changes only loss',
                'training top10pct error-point weight4',
                'training top10pct difficult-frame sampling4',
                'frozen uniform model + same-weight neural residual branch',
                'validation-selected MLP + train-condition RBF coefficient residual; pure RBF control'],
        local_acceptance='validation hotspot RMSE decreases; overall/ordinary/worst-case RMSE each <=1.02*uniform; max_abs reported independently',
        hotspot_definition='training errors of seed42 uniform field model; distinct from supplied geometric hotspot',
        final_evaluation='only after all validation selection and seed checks are frozen; no test-based tuning',
        claims=['new operating-condition interpolation, shared 60 frames, fixed sampled spatial points'],
        not_claimed=['original PPT score reproduced', 'unseen time', 'unseen space', 'certified steady state', 'volume 3D field'],
        metrics='float64 RMSE, MAE, max_abs and 100*||error||2/||truth||2 over sampled vector components; no FEM area weights available')
    protocol['code_sha256'] = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')
                               if p.name in ('experiment_core.py','prepare_new60.py','run_new60.py','convert_merged_data.py','prepare_real_study.py')}
    with protocol_path.open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
    for region in ('rotor', 'stator', 'airgap'):
        result=prepare(ROOT/'inputs_v1'/f'{region}.npz', ROOT/'inputs_v1/split_source.json',
            region, protocol['ranks'][region], protocol['harmonics'], 'absolute',
            ROOT/'cache_v1'/f'{region}.npz', quantity='Br' if region=='airgap' else 'Br,Bt')
        print(json.dumps(dict(region=region, rank=result['rank'], oracle=result['oracle']),ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
