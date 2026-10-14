"""Train-only oracle floors for independent spatial-mode groups and planar |B|.

Uses only V1 train/validation labels via the reviewed selective row loader.
These projections consume validation truth as oracle coefficients, not as model
training; no prediction accuracy claim or final-model choice is made here.
"""
from pathlib import Path
import json
import time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from audit_train_validation import V1,HERE,read,digest,metrics,load_selection_cache


def affine_projection(train,validation):
    mean=train.mean(axis=0);a=train-mean;ev,u=np.linalg.eigh(a@a.T)
    order=np.argsort(ev)[::-1];ev=np.maximum(ev[order],0);u=u[:,order]
    rank=int(np.count_nonzero(ev>ev[0]*1e-12))
    basis=(u[:,:rank].T@a)/np.sqrt(ev[:rank,None])
    return mean+((validation-mean)@basis.T)@basis,rank


def main():
    if (HERE/'grouped_and_magnitude_floors.json').exists():raise FileExistsError('Existing report is immutable')
    split=read(V1/'cache_v1/split.json');tr=split['train'];va=split['validation'];out={}
    with threadpool_limits(limits=2):
      for region in ('rotor','stator'):
        start=time.perf_counter()
        cache,_,meta=load_selection_cache(V1/'cache_v1'/f'{region}.npz',V1/'cache_v1/split.json','Br,Bt')
        assert np.isnan(cache['field'][split['test']]).all()
        p=len(cache['point_id']);truth=cache['field'][va].astype(np.float64)
        # Planar norm of provided signed Br/Bt, not a statement about missing Bz.
        trainmag=np.linalg.norm(cache['field'][tr].astype(np.float64).reshape(42,60,p,2),axis=-1)
        valmag=np.linalg.norm(truth.reshape(9,60,p,2),axis=-1)
        proj,rank=affine_projection(trainmag.reshape(42,-1),valmag.reshape(9,-1))
        cycle=metrics(proj.reshape(9,60,p),valmag,np.asarray(va))
        print(json.dumps(dict(region=region,planar_magnitude_cycle_floor=cycle,rank=rank)),flush=True)
        grouped=[];a=cache['coeff'][tr].astype(np.float64);b=cache['coeff'][va].astype(np.float64);r=a.shape[-1]
        for size in sorted(set([1,4,8,16,32,64,r])):
            predicted=np.empty_like(b);ranks=[]
            for lo in range(0,r,size):
                hi=min(lo+size,r);oracle,group_rank=affine_projection(a[...,lo:hi].reshape(42,-1),b[...,lo:hi].reshape(9,-1))
                predicted[...,lo:hi]=oracle.reshape(9,60,hi-lo);ranks.append(group_rank)
            field=cache['mean']+predicted@cache['basis'].astype(np.float64).T
            grouped.append(dict(spatial_modes_per_group=size,group_count=len(ranks),maximum_affine_rank=max(ranks),
                vector_validation_projection=metrics(field,truth,np.asarray(va))))
        print(json.dumps(dict(region=region,grouped_floors=[dict(group=x['spatial_modes_per_group'],relative_L2_pct=x['vector_validation_projection']['relative_L2_pct']) for x in grouped])),flush=True)
        # One train-only randomized SVD at rank512, same seed/n_iter for all ranks.
        # Prefixes are nested; not refitted or selected using validation targets.
        snapshots=trainmag.reshape(-1,p);mean=snapshots.mean(axis=0);a=snapshots-mean
        svd_start=time.perf_counter();_,singular,vt=randomized_svd(a,n_components=512,n_iter=5,random_state=42)
        floors=[]
        for k in (64,128,192,256,384,512):
            basis=vt[:k].T;prediction=mean+(valmag-mean)@basis@basis.T
            floors.append(dict(rank=k,validation_projection=metrics(prediction,valmag,np.asarray(va)),
                training_centered_energy_fraction=float(np.sum(singular[:k]**2)/np.sum(a*a))))
        out[region]=dict(quantity='sqrt(Br^2+Bt^2): planar magnitude of supplied components',units='T',
            no_Bz_available=True,train_case_ids=tr,validation_case_ids=va,test_labels_materialized=False,
            cache_sha256=digest(V1/'cache_v1'/f'{region}.npz'),whole_cycle_magnitude_affine_rank=rank,
            whole_cycle_magnitude_floor=cycle,spatial_magnitude_POD_floors=floors,
            grouped_vector_POD_oracle_floors=grouped,spatial_svd=dict(training_snapshots=2520,rank=512,n_iter=5,seed=42,
                dtype='float64',seconds=time.perf_counter()-svd_start,validation_used_for_basis=False),
            caveat='All floors use true validation projections, not predicted coefficients. Grouped vectors use V1 rank192 spatial basis and cannot beat its projection floor.',
            elapsed_seconds=time.perf_counter()-start)
        (HERE/f'{region}_magnitude_grouped_floor.json').write_text(json.dumps(out[region],ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(dict(region=region,magnitude_space_floors=[dict(rank=x['rank'],relative_L2_pct=x['validation_projection']['relative_L2_pct']) for x in floors],seconds=out[region]['elapsed_seconds'])),flush=True)
    final=dict(regions=out,diagnostic_code_sha256=digest(Path(__file__)),data_access='Only train42/validation9 field/coeff rows, test rows skipped to NaNs',
        distinction='Planar-magnitude metrics and signed-vector metrics have different quantities; no cross-quantity improvement claim',
        lower_bound_scope='Chosen train-fitted affine/POD subspaces only; not an information-theoretic limit for arbitrary nonlinear decoders or a noise floor')
    (HERE/'grouped_and_magnitude_floors.json').write_text(json.dumps(final,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 分组与二维磁密范数：验证投影下界','','全部基底只由原 train42 拟合，验证真值仅用于 oracle 投影。未读取测试标签，没有重选最终模型。', '',
        'sqrt(Br²+Bt²) 是所给二维分量的范数，和有符号 Br/Bt 联合指标必须分开，不假定缺失的 Bz 为零。','',
        '| 区域 | 二维范数 cycle-span L2/% | 二维范数 space rank192 L2/% | space rank384 | space rank512 |','|---|---:|---:|---:|---:|']
    for r,a in out.items():
        f={v['rank']:v['validation_projection']['relative_L2_pct'] for v in a['spatial_magnitude_POD_floors']}
        lines.append(f"| {r} | {a['whole_cycle_magnitude_floor']['relative_L2_pct']:.6f} | {f[192]:.6f} | {f[384]:.6f} | {f[512]:.6f} |")
    lines+=['','分组赋予不同空间模态各自的工况系数，可以突破“全场共享一组42权重”的 cycle-span 下界，但仍受空间 POD 子空间约束。细分组自由度增加，不等于预测会更准；须用训练内交叉验证和固定验证集检查稳定性。','',
        '固定 train42 的目标POD上的 LOOCV 是 conditional/transductive 诊断。严格外层泛化评价必须在每个训练折内重新拟合 POD、均值、尺度；也可以直接在原输出上用无目标降维的核回归做 fold-safe 控制。','',
        '这些数字是所选仿射/线性子空间内的最佳真值投影，不是任意非线性模型的理论极限，也不是 Maxwell 噪声底。']
    (HERE/'grouped_and_magnitude_floors.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
