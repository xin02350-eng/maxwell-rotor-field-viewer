"""Convert the supplied real merged dataset and freeze a feature-only case split.

No model fitting or test-label evaluation. Input geometry alone selects the
training convex hull, so the new holdouts evaluate parameter interpolation.
This split does not claim to reproduce the absent historical Round4 split.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    source = HERE / 'source/combined/flux60_s60_merged.npz'
    output = HERE / 'inputs_v1'
    output.mkdir(exist_ok=False)
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids = data['case_id'].astype(int)
    assert np.array_equal(ids, np.arange(60))
    conditions = np.column_stack((data['I_A'], data['Thet_deg']))
    # Different physical scales are normalized for numerically stable hulls.
    plane = (conditions - conditions.min(0)) / np.ptp(conditions, axis=0)
    hull_indices = np.sort(ConvexHull(plane).vertices)
    seed = 42
    remaining = np.setdiff1d(np.arange(60), hull_indices)
    shuffled = np.random.default_rng(seed).permutation(remaining)
    train = np.sort(np.concatenate((hull_indices, shuffled[:42-len(hull_indices)])))
    validation = np.sort(shuffled[42-len(hull_indices):51-len(hull_indices)])
    test = np.sort(shuffled[51-len(hull_indices):])
    assert tuple(map(len, (train, validation, test))) == (42, 9, 9)
    split = {key: ids[group].tolist() for key, group in (
        ('train', train), ('validation', validation), ('test', test))}
    split.update(seed=seed, historical_round4_split=False,
                 design='Training includes all input-space convex hull vertices; remaining IDs RNG42',
                 input_hull_case_ids=ids[hull_indices].tolist(),
                 selection_uses='I_A and Thet_deg only; no target field/error values',
                 evaluation_scope='new operating conditions inside training parameter convex hull; shared time/space')
    (output/'split_source.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = dict(source=source.relative_to(HERE).as_posix(), source_sha256=digest(source),
                    split=split, quantities={}, case_count=60, time_count=60,
                    physical=dict(pole_pairs=4, speed_rpm=5500, initial_mechanical_deg=7.5,
                                  current='native model RMS; I_peak=sqrt(2)*I; 4 parallel branches, no /2',
                                  angle='native Thet; not dq beta', steady_state_verified=False))
    for region in ('stator', 'rotor', 'airgap'):
        if region == 'airgap':
            field = data['airgap_Br'].astype(np.float32)
            quantity = 'Br'
        else:
            field = np.stack((data[region+'_Br'], data[region+'_Bt']), axis=-1)
            field = field.reshape(60, 60, -1)
            quantity = 'Br,Bt'
        p = len(data[region+'_point_id'])
        assert field.shape == (60, 60, p * (1 if quantity == 'Br' else 2))
        # Absolute mechanical angle is an injective coordinate on this first
        # electrical period. Harmonic features need not impose endpoint equality.
        path = output/(region+'.npz')
        np.savez_compressed(path, field=field, conditions=conditions, case_ids=ids,
                            theta=data['theta_m_absolute_deg'], point_id=data[region+'_point_id'],
                            xy=np.column_stack((data[region+'_x_ref_mm'], data[region+'_y_ref_mm'])),
                            dataset_hotspot=data[region+'_hotspot'], solver_index=data['solver_index'],
                            time_s=data['time_s'], theta_m_relative_deg=data['theta_m_relative_deg'],
                            theta_e_deg=data['theta_e_deg'], initial_mechanical_deg=data['initial_mechanical_deg'],
                            quantity=np.asarray(quantity))
        manifest['quantities'][region] = dict(quantity=quantity, shape=list(field.shape),
                                              points=p, path=path.name, sha256=digest(path),
                                              coordinate_frame='rotor_reference' if region=='rotor' else 'stator_fixed')
    (output/'conversion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
