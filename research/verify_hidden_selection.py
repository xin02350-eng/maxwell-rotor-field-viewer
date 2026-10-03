"""Audit hyperparameter selection using visible conditions only.

This does not evaluate hidden labels, modify the delivered models, or rewrite
metrics.json. Candidate checkpoints are intermediate files under work/.
"""
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(OUT))
from train_rotor import train_candidate
from rotor_model import metrics


def main():
    with np.load(ROOT / 'work/rotor_dataset.npz', allow_pickle=False) as data:
        y = data['Bmag']
        cond = data['conditions']
        theta = data['theta']
        xy = data['xy']
        point_id = data['point_id']
    train = np.arange(300) % 5 < 3
    validation = np.arange(300) % 5 == 3
    intermediate = ROOT / 'work/hidden_selection_audit'
    intermediate.mkdir(exist_ok=True)
    audit = {
        'purpose': 'Select rank and Fourier order using visible-condition validation only.',
        'hidden_labels_evaluated': False,
        'original_models_modified': False,
        'selection_metric': 'visible-condition angle-validation RMSE in T',
        'candidate_grid': [[32, 3], [32, 5], [64, 5], [64, 6]],
        'previously_selected': {'rank': 64, 'fourier_order': 6},
        'protocol_note': 'This is a retrospective internal audit, not a new blinded external test.',
        'experiments': {},
    }
    for name, held in [('held_current', [3, 4, 5]), ('held_phase', [1, 4, 7, 10])]:
        visible = np.array([c for c in range(12) if c not in held])
        # Passing sliced data prevents candidate inference/scoring over hidden labels.
        yy = y[visible]
        cc = cond[visible]
        rows = []
        for rank, order in audit['candidate_grid']:
            path = intermediate / f'{name}_r{rank}_k{order}.npz'
            pred, model, history = train_candidate(
                yy, cc, theta, np.arange(len(visible)), train, rank, order,
                xy, point_id, path,
            )
            result = metrics(yy[:, validation], pred[:, validation])
            row = {'rank': rank, 'fourier_order': order, 'validation': result}
            rows.append(row)
            print(name, rank, order, result['rmse_T'], flush=True)
        best = min(rows, key=lambda row: row['validation']['rmse_T'])
        audit['experiments'][name] = {
            'visible_case_ids': visible.tolist(),
            'hidden_case_ids': held,
            'training_frames_per_visible_case': int(train.sum()),
            'validation_frames_per_visible_case': int(validation.sum()),
            'independent_training_phases': 15,
            'independent_validation_phases': 5,
            'candidates': rows,
            'selected': {'rank': best['rank'], 'fourier_order': best['fourier_order']},
            'matches_original_choice': best['rank'] == 64 and best['fourier_order'] == 6,
        }
    output = OUT / 'hidden_selection_audit.json'
    output.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('Saved', output, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
