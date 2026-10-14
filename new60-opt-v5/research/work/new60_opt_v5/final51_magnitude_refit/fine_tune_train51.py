"""Exactly one prespecified train51 magnitude NN fine-tune; no validation/test."""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
import magnitude_final51 as inference

def train(prepared,cache,out,config,protocol_path):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);torch.manual_seed(config['seed']);prepared=Path(prepared);cache=Path(cache);out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    sha=inference.sha;protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text(encoding='utf-8'));identity=json.loads((prepared/'identity.json').read_text(encoding='utf-8'));training_ids=identity['train_case_ids']+identity['validation_case_ids']
    if sha(__file__)!=protocol['training_source_sha256'] or sha(inference.__file__)!=protocol['inference_source_sha256'] or config!=protocol['candidate']:raise ValueError('Predeclared both-source/config binding')
    if len(training_ids)!=51 or len(set(training_ids))!=51 or set(training_ids)&set(identity['excluded_case_ids']) or identity['test_labels_loaded'] or identity['full60_fit_labels_loaded']:raise ValueError('Allowed51 label guard')
    if sha(prepared/'identity.json')!=protocol['prepared_identity_sha256'] or sha(prepared/'mean_data.npz')!=protocol['prepared_means_sha256'] or sha(cache)!=protocol['raw_cache_sha256']:raise ValueError('Data provenance binding')
    with np.load(prepared/'mean_data.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    warm_path=Path(config['warmstart_model']);m=inference.load_model(warm_path);warmmeta=json.loads(str(m['metadata_json']))
    if sha(warm_path)!=config['warmstart_model_sha256'] or warmmeta['source_sha256']!=config['warmstart_source_sha256'] or warmmeta['train_case_ids']!=training_ids or warmmeta['validation_case_ids'] or warmmeta['test_labels_loaded'] or warmmeta['full60_fit_labels_loaded']:raise ValueError('Actualtrain51 warmstart guard')
    if a['case_ids'].tolist()!=training_ids or warmmeta['excluded_case_ids']!=identity['excluded_case_ids']:raise ValueError('51 partition order guard')
    with np.load(cache,allow_pickle=False) as z:cacheids=z['case_ids'];conditions=z['conditions'];point_ids=z['point_id'];xy=z['xy'];theta=z['theta'];time_s=z['time_s']
    if cacheids.tolist()!=list(range(60)) or not np.array_equal(conditions[training_ids],a['conditions']) or not np.array_equal(point_ids,m['point_id']) or not np.array_equal(xy,m['xy']) or not np.array_equal(theta,m['theta']) or not np.array_equal(time_s,m['time_s']):raise ValueError('Geometry/nativecondition identity')
    true_signed=inference.selected_rows(cache,'field',training_ids).astype(np.float64).reshape(51,60,1800,2);truth=np.hypot(true_signed[...,0],true_signed[...,1]);target=true_signed.mean(1)
    if not np.allclose(target,a['signed_mean'],atol=1e-12,rtol=1e-12) or not np.allclose(np.sum(truth*truth,axis=(1,2)),a['truth_energy'],atol=1e-8,rtol=1e-12):raise ValueError('Actual51 staticmean/energy identity')
    with threadpool_limits(limits=2):
        base={k[9:]:v for k,v in m.items() if k.startswith('baseline_')};c=a['conditions'];old=inference.baseline_predict(base,c).reshape(51,60,1800,2);dynamic=float(m['dynamic_scale'])*(old-old.mean(1,keepdims=True));x=(inference.features(c,'cartesian',float(m['current_scale']))-m['feature_mean'])/m['feature_scale'];keys=('linear_weight','linear_bias','gate_weight','gate_bias','expert_weight')
        params={k:torch.nn.Parameter(torch.tensor(m[k],dtype=torch.float32).contiguous()) for k in keys};xt=torch.tensor(x,dtype=torch.float32);cm=torch.tensor(m['output_mean'][:,None,:],dtype=torch.float32);sc=torch.tensor(m['output_scale'][:,None,:],dtype=torch.float32);yt=torch.tensor(target,dtype=torch.float32);dyn=torch.tensor(dynamic,dtype=torch.float32);y=torch.tensor(truth,dtype=torch.float32)
        def forward(inp):
            v=torch.einsum('cf,pfh->pch',inp,params['linear_weight'])+params['linear_bias'][:,None,:];z=torch.einsum('cf,pfk->pck',inp,params['gate_weight'])+params['gate_bias'][:,None,:];aug=torch.cat((torch.ones((len(inp),1),dtype=inp.dtype),inp),axis=1);expert=torch.einsum('cf,pkfh->pckh',aug,params['expert_weight'])
            return (v+torch.sum(torch.sigmoid(z)[...,None]*expert,axis=2))*sc+cm
        def objective(frames=None):
            mean=forward(xt).transpose(0,1);signed=mean[:,None]+(dyn if frames is None else dyn[:,frames]);tt=y if frames is None else y[:,frames];magnetic=torch.linalg.vector_norm(signed,dim=-1);mag_mse=torch.mean((magnetic-tt)**2);signed_mse=torch.mean((mean-yt)**2)
            return mag_mse+config['lambda_signed']*signed_mse,mag_mse,signed_mse
        history=[];best=float('inf');best_step=0;best_state=None;closures=0
        def checkpoint(step,stage,frames,loss):
            nonlocal best,best_step,best_state
            with torch.no_grad():mean=forward(xt).numpy().transpose(1,0,2)
            signed=mean[:,None]+dynamic;pred=np.hypot(signed[...,0],signed[...,1]);e=pred-truth;mag_mse=float(np.mean(e*e));signed_mse=float(np.mean((mean-target)**2));full=mag_mse+config['lambda_signed']*signed_mse
            row=dict(step=step,stage=stage,frame_batch='' if frames is None else ','.join(map(str,frames)),sampled_objective_T2=float(loss),train51_full60_magnitude_mse_T2=mag_mse,train51_signed_mean_regularizer_T2=signed_mse,train51_full_objective_T2=full,train51_full60_magnitude_relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(truth)),train51_full60_magnitude_rmse_T=float(np.sqrt(mag_mse)),train51_full60_magnitude_max_abs_T=float(np.abs(e).max()),elapsed_seconds=time.perf_counter()-start);history.append(row)
            if full<best:best=full;best_step=step;best_state={k:p.detach().clone() for k,p in params.items()}
            return row
        def restore():
            with torch.no_grad():
                for k,p in params.items():p.copy_(best_state[k])
        opt=torch.optim.Adam([dict(params=[params[k] for k in ('linear_weight','linear_bias','expert_weight')],lr=config['learning_rate']),dict(params=[params[k] for k in ('gate_weight','gate_bias')],lr=config['gate_learning_rate'])]);checkpoint(0,'actualtrain51_V4_warmstart',None,objective()[0].detach())
        for step in range(1,config['adam_steps']+1):
            frames=list(range((step-1)%10,60,10));opt.zero_grad();loss=objective(frames)[0];loss.backward();torch.nn.utils.clip_grad_norm_(list(params.values()),5);opt.step()
            if step%50==0:
                row=checkpoint(step,'Adam_fixed6frame_cycle',frames,loss.detach())
                if step%500==0:print(json.dumps(dict(name=config['name'],progress=row)),flush=True)
        final_adam=history[-1].copy();restore()
        with torch.no_grad():
            for p in params.values():p.data=p.data.double()
        xt=torch.tensor(x,dtype=torch.float64);cm=torch.tensor(m['output_mean'][:,None,:],dtype=torch.float64);sc=torch.tensor(m['output_scale'][:,None,:],dtype=torch.float64);yt=torch.tensor(target,dtype=torch.float64);dyn=torch.tensor(dynamic,dtype=torch.float64);y=torch.tensor(truth,dtype=torch.float64)
        opt=torch.optim.LBFGS(list(params.values()),lr=1,max_iter=10,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
        for outer in range(1,config['lbfgs_blocks']+1):
            def closure():
                nonlocal closures
                opt.zero_grad();loss=objective()[0];loss.backward()
                for p in params.values():
                    if p.grad is not None and not p.grad.is_contiguous():p.grad=p.grad.contiguous()
                closures+=1;return loss
            opt.step(closure);row=checkpoint(config['adam_steps']+outer*10,'LBFGS_all60_trueframes',None,objective()[0].detach())
            if outer%5==0:print(json.dumps(dict(name=config['name'],progress=row)),flush=True)
        final_optimizer=history[-1].copy();restore();arrays=dict(m);arrays.update({k:p.detach().numpy() for k,p in params.items()})
        meta=dict(version=inference.VERSION,configuration=config,train_case_ids=training_ids,validation_case_ids=[],former_validation_now_training_case_ids=identity['validation_case_ids'],excluded_case_ids=identity['excluded_case_ids'],original_train42_case_ids=identity['train_case_ids'],source_sha256=sha(inference.__file__),training_source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),prepared_identity_sha256=sha(prepared/'identity.json'),prepared_means_sha256=sha(prepared/'mean_data.npz'),raw_cache_sha256=sha(cache),raw_label_rows_loaded=training_ids,
            warmstart_model_sha256=config['warmstart_model_sha256'],warmstart_source_sha256=config['warmstart_source_sha256'],warmstart_kind='Actualtrain51 V4 two-gate signedmean NN; not42fine-tuned weights/notfit60',fine_tuning_objective='originalTesla fullframe magnetic magnitude MSE +.02originalTesla signedstaticmean component MSE',
            checkpoint='minimum train51 all60frames complete magnitudeMSE+.02signedmeanMSE; no independentval or test checkpoint',best_step=best_step,frame_sampling='Adam predetermined6 savedframes offset=(step-1)%10 plus10*k; every10steps coversall60; LBFGS all60',point_weighting='uniform all1800',scalers='immutable actualtrain51 V4 warmstart input/output scalers',dynamic='immutable original train42 V2 pureNN zero60frame-mean; fixedtrain42scale1.0',baseline_model_sha256=warmmeta['baseline_model_sha256'],
            units='T',xy_units='mm',quantity='planar sqrt(Br²+Bt²), no Bz',frame_scope='all60supplied savedfirstcycleframes only',selection_origin='51 recipe chosen previously onlyusing42/9 V5 validation;51training results never alter42 selection or fixed25%V4pure51+75%new51scalarblend',
            label_scope='train42+originalval9 nowtrain51, originaltest9 excluded; all metrics here fitting not independentprediction',test_scope='Oldtest already seenprojectV1-V4; any future score reusedtest notblind; thisroute neverdecodes testlabels',test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,train51_labels_loaded=True)
        arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False));np.savez_compressed(out/'model.npz',**arrays);(out/'model.npz.sha256').write_text(sha(out/'model.npz')+'\n')
        actual=inference.predict_mean(arrays,c)
        with torch.no_grad():expected=forward(xt).numpy().transpose(1,0,2)
        replay=float(np.max(np.abs(actual-expected)))
        if replay>1e-10:raise AssertionError('NumPy/Torch64 static signedmean mismatch')
        pred=inference.predict(out/'model.npz',c);mag_mse=float(np.mean((pred-truth)**2));signed_mse=float(np.mean((actual-target)**2));actual_objective=mag_mse+config['lambda_signed']*signed_mse
        if not np.isclose(actual_objective,best,atol=1e-9,rtol=2e-4):raise AssertionError('Saved training objective not selectedcheckpoint')
        train_signed=actual[:,None]+dynamic;e=train_signed-true_signed
        with (out/'history.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        result=dict(name=config['name'],configuration=config,train51_fitting_magnitude=inference.metrics(pred,truth,a['case_ids'],point_ids),train51_static_mean=inference.mean_metrics(actual,target),train51_full60_signed_components=dict(relative_L2_pct=float(100*np.linalg.norm(e)/np.linalg.norm(true_signed)),component_rmse_T=float(np.sqrt(np.mean(e*e))),component_max_abs_T=float(np.abs(e).max())),train51_selected_objective_T2=actual_objective,train51_selected_magnitude_mse_T2=mag_mse,train51_selected_signed_regularizer_T2=signed_mse,
            validation=None,new_gradient_fine_tuning_run=True,from_scratch_training=False,new_trainable_parameters=sum(p.numel() for p in params.values()),new_parameters_per_point=sum(p.numel() for p in params.values())//1800,best_step=best_step,history_checkpoint_count=len(history),final_adam=final_adam,final_optimizer=final_optimizer,lbfgs_closure_evaluations=closures,elapsed_seconds=time.perf_counter()-start,raw_label_rows_loaded=training_ids,numpy_torch_float64_replay_max_abs_difference_T=replay,model_sha256=sha(out/'model.npz'),source_sha256=sha(inference.__file__),training_source_sha256=sha(__file__),protocol_sha256=sha(protocol_path),test_labels_loaded=False,test_metrics_computed=False,full60_fit_labels_loaded=False,train51_labels_loaded=True)
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('prepared','cache','output','config','protocol'):parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args();train(args.prepared,args.cache,args.output,json.loads(args.config.read_text(encoding='utf-8')),args.protocol)
