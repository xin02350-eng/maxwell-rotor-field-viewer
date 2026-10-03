"""POD + conditional neural Fourier field model. Units: A RMS, mechanical degrees, T."""
import json
from pathlib import Path
import numpy as np
try:
    import torch
    from torch import nn
except ImportError:  # Prediction from NPZ needs NumPy only.
    torch = None
    nn = None


class CoefficientNet(nn.Module if nn is not None else object):
    def __init__(self, outputs, width=32):
        if nn is None:
            raise ImportError('Training requires PyTorch; NPZ inference needs only NumPy.')
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, outputs))

    def forward(self, x):
        return self.net(x)


def condition_features(conditions):
    c = np.asarray(conditions, dtype=np.float64).reshape(-1, 2)
    return np.column_stack(((c[:, 0] - 22.) / 5.3, (c[:, 1] - 30.) / 30.))


def fourier_features(theta, harmonics):
    rad = np.deg2rad(np.asarray(theta).reshape(-1))
    a = rad[:, None] * np.asarray(harmonics)[None, :]
    return np.column_stack((np.ones(len(rad)), np.sin(a), np.cos(a)))


def predict_numpy(model, conditions, theta):
    """One field for each pair (condition row, relative mechanical angle)."""
    x = condition_features(conditions)
    for layer in model['layers']:
        x = x @ np.asarray(layer['weight']).T + np.asarray(layer['bias'])
        if layer['activation'] == 'tanh':
            x = np.tanh(x)
    cf = x * model['coef_scale'] + model['coef_mean']
    cf = cf.reshape(len(x), 1 + 2 * len(model['harmonics']), len(model['basis']))
    score = np.einsum('nh,nhr->nr', fourier_features(theta, model['harmonics']), cf)
    return np.maximum(model['mean'] + score @ model['basis'], 0.)


def load_numpy(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        layers = []
        for i in range(3):
            layers.append({'weight': z[f'w{i}'], 'bias': z[f'b{i}'],
                           'activation': 'tanh' if i < 2 else 'linear'})
        return {**meta, 'layers': layers,
                **{k: z[k] for k in ['mean', 'basis', 'coef_mean', 'coef_scale', 'xy', 'point_id']}}


def export_model(path, net, mean, basis, coef_mean, coef_scale, harmonics, xy, point_id, metadata):
    meta = {**metadata, 'harmonics': list(map(int, harmonics)), 'network': '2-32-32-FourierCoefficients',
            'input_units': ['A RMS', 'phase angle degrees', 'relative mechanical angle degrees'],
            'output_units': 'T', 'initial_absolute_angle_deg': 36.,
            'sampled_region': '180 degree rotor iron, 2500 fixed rotor-reference points'}
    arrays = dict(mean=mean, basis=basis, coef_mean=coef_mean, coef_scale=coef_scale,
                  xy=xy, point_id=point_id, metadata=json.dumps(meta))
    for i, layer in enumerate([net.net[0], net.net[2], net.net[4]]):
        arrays[f'w{i}'] = layer.weight.detach().numpy()
        arrays[f'b{i}'] = layer.bias.detach().numpy()
    np.savez_compressed(path, **arrays)
    torch.save({'state_dict': net.state_dict(), 'metadata': meta,
                'mean': torch.tensor(mean), 'basis': torch.tensor(basis),
                'coef_mean': torch.tensor(coef_mean), 'coef_scale': torch.tensor(coef_scale)},
               str(Path(path).with_suffix('.pt')))


def metrics(truth, pred):
    y = np.asarray(truth, dtype=np.float64).reshape(-1, truth.shape[-1])
    p = np.asarray(pred, dtype=np.float64).reshape(y.shape)
    e = p-y
    rel = np.linalg.norm(e, axis=1) / np.maximum(np.linalg.norm(y, axis=1), 1e-12)
    return dict(rmse_T=float(np.sqrt(np.mean(e**2))), mae_T=float(np.mean(abs(e))),
                r2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),
                relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(y)),
                frame_relative_L2_median_pct=float(100*np.median(rel)),
                frame_relative_L2_p95_pct=float(100*np.quantile(rel, .95)),
                abs_error_p95_T=float(np.quantile(abs(e), .95)),
                abs_error_p99_T=float(np.quantile(abs(e), .99)),
                max_abs_error_T=float(abs(e).max()),
                peak_abs_error_mean_T=float(np.mean(abs(p.max(1)-y.max(1)))))


if __name__ == '__main__':
    import argparse, csv
    ap = argparse.ArgumentParser(description='Predict the sampled rotor iron field (not a new geometry).')
    ap.add_argument('--model', default=str(Path(__file__).with_name('rotor_production.npz')))
    ap.add_argument('--current', type=float, required=True, help='A RMS, not peak current')
    ap.add_argument('--phase', type=float, required=True)
    ap.add_argument('--theta', type=float, required=True, help='Mechanical angle relative to first frame')
    ap.add_argument('--output', default='prediction.csv')
    args = ap.parse_args()
    if not 16.7 <= args.current <= 27.3 or not 0 <= args.phase <= 60:
        ap.error('Outside the supported current/phase range; no validated extrapolation.')
    m = load_numpy(args.model)
    field = predict_numpy(m, [[args.current, args.phase]], [args.theta])[0]
    xy = m['xy']; a = np.deg2rad(args.theta + 36.)
    world = xy @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['point_id', 'x_ref_mm', 'y_ref_mm', 'x_world_mm', 'y_world_mm', 'B_magnitude_T'])
        w.writerows(zip(m['point_id'], xy[:, 0], xy[:, 1], world[:, 0], world[:, 1], field))
    print(f'Saved {len(field)} predictions to {args.output}; range {field.min():.6f}–{field.max():.6f} T')
