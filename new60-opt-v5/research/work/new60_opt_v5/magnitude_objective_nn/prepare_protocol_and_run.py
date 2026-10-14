"""Pre-register four actual magnitude-objective neural fine-tuning runs."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PREP=ROOT/'work/new60_opt_v3/joint_neural/point_prepared'
TRUTH=ROOT/'work/new60_opt_v2/neural_route/rotor_spatial_magnitude/prepared/validation_truth.npz'
CACHE=ROOT/'work/new60_20261014/cache_v1/rotor.npz'
SOURCE=HERE/'magnitude_point_nn.py'
OLD=ROOT/'work/new60_opt_v4/point_regime_nn'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ast.parse(SOURCE.read_text(encoding='utf-8'))
    identity=json.loads((PREP/'identity.json').read_text(encoding='utf-8'))
    if sha(CACHE)!=identity['cache_sha256']:raise ValueError('Selectedrows raw cache is not previously audited source')
    starts=[(2,OLD/'runs/regime_sigmoid_affine2_principal_retry1_seed42/model.npz'),(3,OLD/'runs/regime_sigmoid_affine3_principal_seed42/model.npz')]
    configs=[]
    for gates,path in starts:
        for penalty in (.02,.2):
            configs.append(dict(name=f'magnitude_ft_gate{gates}_lambda{str(penalty).replace(".","p")}_seed42',gates=gates,lambda_signed=penalty,seed=42,adam_steps=2000,lbfgs_blocks=10,learning_rate=.001,gate_learning_rate=.0002,warmstart_model=str(path),warmstart_model_sha256=sha(path),warmstart_source_sha256=sha(OLD/'regime_point_nn.py')))
    protocol=dict(version='v5-four-predeclared-fullfield-magnitude-warmstart-NN-finetunes',authorization='Root approved four actual gradient fine-tuning runs: actualtrain42 two/three-gate NN × originalTesla signed regularizerlambda .02/.2. Notfromscratch; no automatic retry, extra architecture, seed or hyperparameter sweep.',
        candidates=configs,source_sha256=sha(SOURCE),prepared_identity_sha256=sha(PREP/'identity.json'),prepared_means_sha256=sha(PREP/'mean_data.npz'),raw_cache_sha256=sha(CACHE),validation_truth_sha256=sha(TRUTH),
        train_case_ids=identity['train_case_ids'],validation_case_ids=identity['validation_case_ids'],excluded_case_ids=identity['excluded_case_ids'],frames=60,points=1800,CPU_threads=2,
        objective='mean across train42×sampledframes×1800 of (hypot(signedmeanNN+immutableoldneuraldynamic)-trueframe|B|)^2 +lambda_signed mean across train42×1800×2 of (predsignedmean-trueBrBtmean)^2, both unnormalizedTesla²; allpointsuniform',
        true_train_label_access='zip NPY selectedrows onlytrain42 interpreted; all other18 rows byte-skipped. No np.load(field) full60. Validation norm truth from existingval9-only NPZ; preparedstatic targets train42+val9 previously audited.',
        frame_schedule=dict(frames_per_Adam_step=6,offset='(step-1)%10',indices='offset +10*k, k=0..5',coverage='every10steps all60frames exactlyonce; deterministic predeclared, never based on val',Adam_steps=2000,LBFGS_all60_steps=100),
        checkpoint='every50Adamsteps and10LBFGSsteps, plusstep0; actual full42×60×1800 train magnitude and all9×60×1800 val magnitude. Select minimumval fullmagRMSE; report allmax/percases.',
        warmstart_scope='V4 raw signedmean-trained 2/3gate actualtrain42 models; oldval9 already used repeatedly for checkpoint selection; no train51 or fit60 model/scalers.',
        frozen_scalers='native currentmax, Cartesian2 inputmean/std, perpoint signedmean outputmean/std exactly copied fromtrain42 warmstart; no updating withval',
        frozen_dynamic='exact original V2 train42 pureNN fullsigned output minusits60frame mean, fixedtrain-onlyscale1.0; no dynamicfine-tuning',
        new_parameters_per_point=[24,24,33,33],local_combination_evaluations=0,from_scratch_training=False,
        original_regime8_frozen_manifest_sha256=sha(OLD/'frozen_directory_manifest.json'),radial2_frozen_manifest_sha256=sha(ROOT/'work/new60_opt_v4/radial_saturation_nn/frozen_directory_manifest.json'),
        units='T',quantity='planar sqrt(Br²+Bt²), no Bz',frame_scope='onlyall60suppliedfirst-cycleframes; no arbitrarytime/secondcycleclaim',test_scope='Oldtest9 already seen in projectV1-V4; any future evaluation is reusedtest, notblind. Thisroute neverloadstest.',
        test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,train51_labels_loaded=False)
    protocol_path=HERE/'candidate_protocol.json'
    if protocol_path.exists():raise FileExistsError('Write protocol once before results')
    protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(HERE/'candidate_protocol.sha256').write_text(sha(protocol_path)+'\n')
    print(json.dumps(dict(protocol_sha256=sha(protocol_path),source_sha256=sha(SOURCE),configs=len(configs))),flush=True)
    results=[]
    for config in configs:
        name=config['name'];config_path=HERE/(name+'_config.json');config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (HERE/'progress.json').write_text(json.dumps(dict(active=name,completed=[x['name'] for x in results],protocol_sha256=sha(protocol_path)))+'\n',encoding='utf-8')
        command=[sys.executable,str(SOURCE),'--prepared',str(PREP),'--truth',str(TRUTH),'--cache',str(CACHE),'--output',str(HERE/'runs'/name),'--config',str(config_path),'--protocol',str(protocol_path)]
        with (HERE/(name+'_stdout.log')).open('w',encoding='utf-8') as log:completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        if completed.returncode:
            failed=dict(name=name,returncode=completed.returncode,source_sha256=sha(SOURCE),protocol_sha256=sha(protocol_path),log=name+'_stdout.log',completed_verified=False,test_labels_loaded=False)
            (HERE/'runs'/name/'failed.json').write_text(json.dumps(failed,indent=2)+'\n',encoding='utf-8');raise RuntimeError('Failure recorded, no automatic replacement: '+name)
        result=json.loads((HERE/'runs'/name/'result.json').read_text(encoding='utf-8'));results.append(result);print(json.dumps(dict(name=name,val_L2=result['validation']['relative_L2_pct'],max_T=result['validation']['max_abs_T'],train_L2=result['train_full60_magnitude']['relative_L2_pct'],seconds=result['elapsed_seconds'])),flush=True)
    (HERE/'all_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(HERE/'progress.json').write_text(json.dumps(dict(active=None,completed=[x['name'] for x in results],protocol_sha256=sha(protocol_path)))+'\n',encoding='utf-8')

if __name__=='__main__':main()
