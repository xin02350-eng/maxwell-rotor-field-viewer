"""Reproducible training. First run prepare_rotor.py on the original archive.
All POD bases and coefficient targets use training data only. Test scores are
computed only after hyperparameters are selected from validation scores.
"""
import argparse, json, time, copy, csv
from pathlib import Path
import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from scipy.interpolate import CubicSpline
from rotor_model import (CoefficientNet, condition_features, fourier_features,
                         export_model, load_numpy, predict_numpy, metrics)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)


def fit_network(target, cond, seed=42):
    torch.manual_seed(seed)
    cm = target.mean(0)
    # Standardize each Fourier coefficient; floor suppresses simulation noise.
    cs = np.maximum(target.std(0), .002)
    x = torch.tensor(condition_features(cond))
    z = torch.tensor((target-cm)/cs)
    net = CoefficientNet(target.shape[1]).double()
    opt = torch.optim.Adam(net.parameters(), lr=.008)
    hist=[]
    for step in range(801):
        opt.zero_grad(); loss=((net(x)-z)**2).mean(); loss.backward();opt.step()
        if step%100==0: hist.append({'step':step, 'loss_standardized':float(loss.detach())})
    opt = torch.optim.LBFGS(net.parameters(), lr=.8, max_iter=600,
                           tolerance_grad=1e-10, tolerance_change=1e-13,
                           history_size=40, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss=((net(x)-z)**2).mean(); loss.backward();return loss
    opt.step(closure)
    hist.append({'step':1400,'loss_standardized':float(((net(x)-z)**2).mean().detach())})
    net.eval()
    return net,cm,cs,hist


def train_candidate(y,cond,theta,train_cases,train_frames,rank,k,xy,ids,path,basis_info=None):
    start=time.time(); harmonics=np.arange(1,k+1)*12
    if basis_info is None:
        a=y[train_cases][:,train_frames].reshape(-1,y.shape[-1]).astype('float64')
        mean=a.mean(0)
        _,_,basis=randomized_svd(a-mean,n_components=rank,random_state=42,n_iter=4)
    else:mean,b=basis_info;basis=b[:rank]
    score=(y[train_cases][:,train_frames].astype('float64')-mean)@basis.T
    phi=fourier_features(theta[train_frames],harmonics)
    cf=np.einsum('ht,ctr->chr',np.linalg.pinv(phi),score)
    target=cf.reshape(len(train_cases),-1)
    net,cm,cs,hist=fit_network(target,cond[train_cases])
    meta={'rank':rank,'fourier_order':k,'train_cases':list(map(int,train_cases)),
          'train_frames':list(map(int,np.where(train_frames)[0])),
          'seed':42,'training_seconds':time.time()-start}
    export_model(path,net,mean,basis,cm,cs,harmonics,xy,ids,meta)
    model=load_numpy(path)
    cp=np.repeat(cond,300,0); tp=np.tile(theta,len(cond))
    pred=predict_numpy(model,cp,tp).reshape(y.shape).astype('float32')
    return pred,model,hist


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'work/rotor_dataset.npz'));args=ap.parse_args()
    d=np.load(args.data);y=d['Bmag'];cond=d['conditions'];theta=d['theta'];xy=d['xy'];ids=d['point_id']
    split=np.where(np.arange(300)%5<3,0,np.where(np.arange(300)%5==3,1,2))
    tr=split==0;va=split==1;te=split==2;cases=np.arange(12)
    # Phase-equivalent frames separated by 25 samples have the same split label.
    assert np.array_equal(split[:-25],split[25:])
    (OUT/'evaluation_split.json').write_text(json.dumps({'angle_split':{'train':np.where(tr)[0].tolist(),'validation':np.where(va)[0].tolist(),'test':np.where(te)[0].tolist()},'held_current_test_cases':[3,4,5],'held_phase_test_cases':[1,4,7,10], 'purpose':'frame and condition holdout; NOT spatial point holdout'},indent=2),encoding='utf-8')
    a=y[:,tr].reshape(-1,2500).astype('float64');mean=a.mean(0)
    _,s,basis=randomized_svd(a-mean,n_components=64,random_state=42,n_iter=5)
    candidates=[];best=None;histories={}
    for rank,k in [(32,3),(32,5),(64,5),(64,6)]:
        name=f'r{rank}_k{k}';path=ROOT/'work'/f'{name}.npz'
        pred,model,hist=train_candidate(y,cond,theta,cases,tr,rank,k,xy,ids,path,(mean,basis))
        vm=metrics(y[:,va],pred[:,va]);row={'name':name,'rank':rank,'fourier_order':k,'validation':vm}
        candidates.append(row);histories[name]=hist
        print('CANDIDATE',json.dumps(row),flush=True)
        if best is None or vm['rmse_T']<best[0]:best=(vm['rmse_T'],rank,k,pred,path)
    _,rank,k,pred,bestpath=best
    import shutil
    shutil.copy2(bestpath,OUT/'rotor_angle_evaluation.npz');shutil.copy2(bestpath.with_suffix('.pt'),OUT/'rotor_angle_evaluation.pt')
    results={'selected':{'rank':rank,'fourier_order':k},'candidates':candidates,
             'angle_test':metrics(y[:,te],pred[:,te]),'angle_validation':metrics(y[:,va],pred[:,va]),
             'angle_train':metrics(y[:,tr],pred[:,tr]),'per_case_angle_test':[], 'baselines':{}}
    for c in cases:
        results['per_case_angle_test'].append({'case_id':int(c),'Irms_A':float(cond[c,0]),'phase_angle_deg':float(cond[c,1]),**metrics(y[c,te],pred[c,te])})
    # Training-only baselines, including a strong periodic spline.
    meanpred=np.repeat(y[:,tr].mean(1)[:,None,:],te.sum(),axis=1)
    results['baselines']['training_case_mean_angle_test']=metrics(y[:,te],meanpred)
    phase=theta%30.; train_phase=np.unique(np.round(phase[tr],6))
    vals=np.stack([y[:,tr][:,np.isclose(phase[tr],v)].mean(1) for v in train_phase],axis=1)
    sp=CubicSpline(np.r_[train_phase,train_phase[0]+30],np.concatenate([vals,vals[:,:1]],axis=1),axis=1,bc_type='periodic')
    spline=sp(phase[te]);results['baselines']['periodic_cubic_spline_angle_test']=metrics(y[:,te],spline)
    # Each known condition independently fits its own Fourier coefficients.
    # This is a strong lookup/interpolation baseline, not a neural network.
    fb=[]
    for fk in [3,5,6]:
        ftr=fourier_features(theta[tr],np.arange(1,fk+1)*12)
        fc=np.einsum('ht,ctp->chp',np.linalg.pinv(ftr),y[:,tr].astype('float64'))
        fv=np.einsum('th,chp->ctp',fourier_features(theta[va],np.arange(1,fk+1)*12),fc)
        fb.append((metrics(y[:,va],fv)['rmse_T'],fk,fc))
    _,fk,fc=min(fb,key=lambda a:a[0])
    ft=np.einsum('th,chp->ctp',fourier_features(theta[te],np.arange(1,fk+1)*12),fc)
    results['baselines']['per_condition_fourier_angle_test']={'selected_order':fk,**metrics(y[:,te],ft)}
    results['angle_test_dynamic_rmse_ratio_pct']=100*results['angle_test']['rmse_T']/results['baselines']['training_case_mean_angle_test']['rmse_T']
    extra={}
    for label,held in [('held_current',[3,4,5]),('held_phase',[1,4,7,10])]:
        tc=np.array([c for c in cases if c not in held])
        # Select each condition-holdout model using visible conditions only.
        # Refit POD separately for every candidate, without hidden-case labels.
        hidden_candidates=[];hidden_best=None
        for hidden_rank,hidden_k in [(32,3),(32,5),(64,5),(64,6)]:
            hidden_path=ROOT/'work'/f'{label}_r{hidden_rank}_k{hidden_k}.npz'
            candidate_pred,_,_=train_candidate(y,cond,theta,tc,tr,hidden_rank,hidden_k,xy,ids,hidden_path)
            visible_validation=metrics(y[tc][:,va],candidate_pred[tc][:,va])
            hidden_candidates.append({'rank':hidden_rank,'fourier_order':hidden_k,'validation':visible_validation})
            if hidden_best is None or visible_validation['rmse_T']<hidden_best[0]:
                hidden_best=(visible_validation['rmse_T'],hidden_rank,hidden_k,candidate_pred,hidden_path)
        _,selected_hidden_rank,selected_hidden_k,pp,selected_hidden_path=hidden_best
        shutil.copy2(selected_hidden_path,OUT/f'rotor_{label}_evaluation.npz')
        shutil.copy2(selected_hidden_path.with_suffix('.pt'),OUT/f'rotor_{label}_evaluation.pt')
        results[label]={'test_cases':held,
                        'selected':{'rank':selected_hidden_rank,'fourier_order':selected_hidden_k},
                        'selection_source':'visible-condition angle-validation only',
                        'candidates':hidden_candidates,
                        'test':metrics(y[held],pp[held]),'validation_known_cases':metrics(y[tc][:,va],pp[tc][:,va]),'per_case':[]}
        for c in held:results[label]['per_case'].append({'case_id':c,**metrics(y[c],pp[c])})
        # Fixed current/phase linear interpolation baseline uses training frames
        # at neighbouring conditions and the same training-only periodic spline.
        baseline=[]
        for c in held:
            if label=='held_current':low=c-3;high=c+3;w=(cond[c,0]-cond[low,0])/(cond[high,0]-cond[low,0])
            else:low=c-1;high=c+1;w=.5
            vv=(1-w)*vals[low]+w*vals[high]
            cspl=CubicSpline(np.r_[train_phase,train_phase[0]+30],np.concatenate([vv,vv[:1]]),axis=0,bc_type='periodic')
            baseline.append(cspl(phase))
        results[label]['linear_condition_spline_baseline']=metrics(y[held],np.stack(baseline))
        extra[label+'_pred']=pp
        print(label,json.dumps(results[label]),flush=True)
    # Freeze assessment first. Production fit uses all labels, is not a test model.
    production,_,hp=train_candidate(y,cond,theta,cases,np.ones(300,dtype=bool),rank,k,xy,ids,OUT/'rotor_production.npz')
    results['production_training_fit']=metrics(y,production)
    results['environment']={'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','threads':4,'seed':42}
    np.savez_compressed(ROOT/'work/predictions.npz',angle_pred=pred,production_pred=production,split=split,**extra)
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (OUT/'training_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    with (OUT/'per_case_test_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=results['per_case_angle_test'][0].keys());w.writeheader();w.writerows(results['per_case_angle_test'])
    print('FINAL',json.dumps({a:results[a] for a in ['selected','angle_test','baselines','production_training_fit']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
