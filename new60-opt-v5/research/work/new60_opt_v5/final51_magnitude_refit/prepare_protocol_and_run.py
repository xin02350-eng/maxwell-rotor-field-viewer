"""Bind and run exactly one additional authorized train51 NN fine-tune."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREP=ROOT/'work/new60_opt_v3/joint_neural/point_prepared'
CACHE=ROOT/'work/new60_20261014/cache_v1/rotor.npz'
WARM=ROOT/'work/new60_opt_v4/final51_refit/final51_regime_sigmoid_affine2_principal_seed42/model.npz'
WARMSOURCE=ROOT/'work/new60_opt_v4/final51_refit/regime_final51.py'
INFER=HERE/'magnitude_final51.py'
TRAIN=HERE/'fine_tune_train51.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    for p in (INFER,TRAIN):ast.parse(p.read_text(encoding='utf-8'))
    identity=json.loads((PREP/'identity.json').read_text(encoding='utf-8'));training=identity['train_case_ids']+identity['validation_case_ids']
    config=dict(name='final51_magnitude_ft_gate2_lambda0p02_seed42',gates=2,lambda_signed=.02,seed=42,adam_steps=2000,lbfgs_blocks=10,learning_rate=.001,gate_learning_rate=.0002,warmstart_model=str(WARM),warmstart_model_sha256=sha(WARM),warmstart_source_sha256=sha(WARMSOURCE))
    protocol=dict(version='v5-extra1-final51-fixed-magnitude-finetune-afterpure42-validation',authorization='Root explicitly approved exactlyone train51 additionalgradientfinetune after fourV5 train42 models frozen and fixedpure42 scalarblend selected. This is an addon protocolchain, not mutation of oldprotocol. No extra tuning/seed/structure or automaticretry.',
        candidate=config,inference_source_sha256=sha(INFER),training_source_sha256=sha(TRAIN),prepared_identity_sha256=sha(PREP/'identity.json'),prepared_means_sha256=sha(PREP/'mean_data.npz'),raw_cache_sha256=sha(CACHE),
        training_case_ids=training,validation_case_ids=[],former_validation_now_training_case_ids=identity['validation_case_ids'],excluded_test_case_ids=identity['excluded_case_ids'],frames=60,points=1800,CPU_threads=2,
        warmstart_model_sha256=sha(WARM),warmstart_source_sha256=sha(WARMSOURCE),warmstart_manifest_sha256=sha(ROOT/'work/new60_opt_v4/final51_refit/final51_models_frozen.json'),
        selection_origin='Previously selected onlyon original42/9 validation: 25%V4pure42norm+75%gate2lambda.02norm gaveval2.509153708%/max.7017403844T. No51result feedsback to42 selection. Primarymixed42 unchangedV4; primarymixed51 oldfixed50/50 unchanged.',
        fixed_final51_recipe=dict(old_V4_pure51_scalar_norm_weight=.25,new_this51_scalar_norm_weight=.75,old_V4_pure51_model_sha256=sha(ROOT/'work/new60_opt_v4/final51_refit/fixed_global_blends/pure_nn_model.npz')),
        objective='uniform51×sampledframes×1800 originalTesla |B| MSE +.02uniform51×1800×2 originalTesla signedmean component MSE; no outputstd lossnormalization; no hotspots or pointweights',
        frame_schedule='2000Adamsteps, frames=(step-1)%10+10*k k0..5, all60 every10steps;100full60 LBFGSsteps',
        checkpoint='51 records: step0, every50Adamsteps, every10LBFGSsteps. Select minimumactualfull51×60×1800 trainobjective magMSE+.02signedmeanMSE, nevervalidation/test. Reporttrain51 fittingmetrics only.',
        scalers='Exactlyimmutabletrain51 V4 warmstart input/output scalers',dynamic='Exactlyimmutable originaltrain42 NN signed output minusits60frame mean, fixedtrain42scale1.0 embeddedwarmmodel; no 42finetuned weights used',
        numeric_label_access='Interpret only51 named rows ofrawZIPfield NPY. Originaltest9 byte-skipped andneverdecoded; no full60fit or independentvaltruthfile loaded.',
        original_V5_four_protocol_sha256=sha(ROOT/'work/new60_opt_v5/magnitude_objective_nn/candidate_protocol.json'),original_V5_four_frozen_manifest_sha256=sha(ROOT/'work/new60_opt_v5/magnitude_objective_nn/frozen_directory_manifest.json'),
        test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,train51_labels_loaded=True)
    pp=HERE/'candidate_protocol.json'
    if pp.exists():raise FileExistsError('Prespecifiedone-run protocolwrittenonce')
    pp.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(HERE/'candidate_protocol.sha256').write_text(sha(pp)+'\n');cp=HERE/(config['name']+'_config.json');cp.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(protocol_sha256=sha(pp),inference_source_sha256=sha(INFER),training_source_sha256=sha(TRAIN),candidates=1)),flush=True)
    (HERE/'progress.json').write_text(json.dumps(dict(active=config['name'],completed=[],protocol_sha256=sha(pp)))+'\n',encoding='utf-8')
    command=[sys.executable,str(TRAIN),'--prepared',str(PREP),'--cache',str(CACHE),'--output',str(HERE/'runs'/config['name']),'--config',str(cp),'--protocol',str(pp)]
    with (HERE/(config['name']+'_stdout.log')).open('w',encoding='utf-8') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    if result.returncode:
        (HERE/'runs'/config['name']/'failed.json').write_text(json.dumps(dict(name=config['name'],returncode=result.returncode,completed_verified=False,protocol_sha256=sha(pp),test_labels_loaded=False),indent=2)+'\n',encoding='utf-8');raise RuntimeError('Recordedfailedattempt; noautomaticretry')
    r=json.loads((HERE/'runs'/config['name']/'result.json').read_text(encoding='utf-8'));(HERE/'progress.json').write_text(json.dumps(dict(active=None,completed=[config['name']],protocol_sha256=sha(pp)))+'\n',encoding='utf-8')
    print(json.dumps(dict(name=r['name'],train51_L2=r['train51_fitting_magnitude']['relative_L2_pct'],train51_max_T=r['train51_fitting_magnitude']['max_abs_T'],best_step=r['best_step'],seconds=r['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
