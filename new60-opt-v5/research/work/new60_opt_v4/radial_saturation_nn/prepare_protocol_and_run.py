"""Pre-register and execute exactly two separately approved radial NN models."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREP=ROOT/'work/new60_opt_v3/joint_neural/point_prepared'
TRUTH=ROOT/'work/new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared/validation_truth.npz'
SOURCE=HERE/'radial_point_nn.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ast.parse(SOURCE.read_text(encoding='utf-8'))
    identity=json.loads((PREP/'identity.json').read_text(encoding='utf-8'))
    configs=[dict(name='radial_tanh_isotropic_seed42',radial_kind='isotropic',seed=42,adam_steps=5000,lbfgs_blocks=20,learning_rate=.005,radial_prior=1e-6,minimum_saturation=1e-3,max_log_stretch=1.3862943611198906,max_shear=2.0),
             dict(name='radial_tanh_elliptic_seed42',radial_kind='elliptic',seed=42,adam_steps=5000,lbfgs_blocks=20,learning_rate=.005,radial_prior=1e-6,minimum_saturation=1e-3,max_log_stretch=1.3862943611198906,max_shear=2.0)]
    protocol=dict(version='v4-extra2-radial-saturation-predeclared-before-results',authorization='Root explicitly approved two additional genuinely coupled radial NN candidates, separate from frozen point_regime_nn8; cumulative completed NN42 budget10. No extra seeds/hyperparameter search.',
        candidates=configs,new_parameters_per_point=[7,9],source_sha256=sha(SOURCE),prepared_identity_sha256=sha(PREP/'identity.json'),prepared_means_sha256=sha(PREP/'mean_data.npz'),geometry_sha256=sha(PREP/'geometry.npz'),baseline_model_sha256=sha(PREP/'baseline_model.npz'),validation_truth_sha256=sha(TRUTH),
        train_case_ids=identity['train_case_ids'],validation_case_ids=identity['validation_case_ids'],excluded_case_ids=identity['excluded_case_ids'],frames=60,points=1800,CPU_threads=2,
        checkpoint='minimum full9x60x1800 raw planar magnetic magnitude RMSE; report fullmax and allcases, no cropping or validation point weights',
        loss='uniform original-T signed mean squared error scaled by fulltrain42 magnetic energy, plus1e-6 train-initialization prior on positive saturation and ellipse raw parameters',
        initialization='train42 perpoint signed mean scalar energy; train42 input/current scalers; saturation1.25xmaximum train radial mean norm; inverse tanh target followed by affine ridgeOLS; elliptic metric fromtrain signed covariance, det1, initial eigenvalues0.5..2',
        model_isotropic='h_p=b_p+A_p*x; B_p=tanh(||h_p||/Bs_p)/(||h_p||/Bs_p)*h_p; Bs positive softplus+1e-3; 6affine+1saturation parameters',
        model_elliptic='same affine drive; radial norm||U_p h_p|| where U=[[exp(s),a],[0,exp(-s)]], det1, s=ln4*tanh(raw_s),a=2*tanh(raw_a); 7+2parameters',
        physical_limitation='Effective drive proxy h is not observed H. Structural B-H analogy is only a low-capacity inductive bias; anisotropy is learned response geometry, not measured material tensor. Smooth saturation cannot by itself prove explanations of sharp nearby-condition jumps.',
        dynamic='Immutable original train42 V2 signed pureNN, remove eachpoint sixty-frame mean; fixedtrain-only scale1.0. No all60/51 model used.',
        unchanged_frozen_directory='work/new60_opt_v4/point_regime_nn',frozen_directory_manifest_sha256=sha(ROOT/'work/new60_opt_v4/point_regime_nn/frozen_directory_manifest.json'),
        no_automatic_retry_or_budget_expansion=True,test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False)
    protocol_path=HERE/'candidate_protocol.json'
    if protocol_path.exists():raise FileExistsError('Protocol must be written once before training')
    protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(HERE/'candidate_protocol.sha256').write_text(sha(protocol_path)+'\n')
    results=[]
    for config in configs:
        name=config['name'];config_path=HERE/(name+'_config.json');config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
        (HERE/'progress.json').write_text(json.dumps(dict(active=name,completed=[x['name'] for x in results],protocol_sha256=sha(protocol_path)))+'\n',encoding='utf-8')
        cmd=[sys.executable,str(SOURCE),'--prepared',str(PREP),'--truth',str(TRUTH),'--output',str(HERE/'runs'/name),'--config',str(config_path),'--protocol',str(protocol_path)]
        with (HERE/(name+'_stdout.log')).open('w',encoding='utf-8') as stdout:
            completed=subprocess.run(cmd,stdout=stdout,stderr=subprocess.STDOUT,check=False,cwd=ROOT)
        if completed.returncode:
            evidence=dict(name=name,returncode=completed.returncode,log=name+'_stdout.log',source_sha256=sha(SOURCE),protocol_sha256=sha(protocol_path),test_labels_loaded=False)
            (HERE/'runs'/name/'failed.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8');raise RuntimeError('Failed candidate; budget does not automatically retry: '+name)
        result=json.loads((HERE/'runs'/name/'result.json').read_text(encoding='utf-8'));results.append(result)
        print(json.dumps(dict(name=name,validation=result['validation']['relative_L2_pct'],max_abs=result['validation']['max_abs_T'],seconds=result['elapsed_seconds'])),flush=True)
    (HERE/'all_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'progress.json').write_text(json.dumps(dict(active=None,completed=[x['name'] for x in results],protocol_sha256=sha(protocol_path)))+'\n',encoding='utf-8')

if __name__=='__main__':main()
