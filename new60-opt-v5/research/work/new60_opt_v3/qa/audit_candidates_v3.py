"""Independent numerical/protocol audit; no training and no test label reads."""
from pathlib import Path
import ast,hashlib,importlib.util,io,json,sys,zipfile
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;V2=WORK/'new60_opt_v2';V1=WORK/'new60_20261014';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'joint_neural'))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))

checks=[]
def check(name,condition,detail=None):
    checks.append(dict(check=name,pass_=bool(condition),detail=detail))
    if not condition:raise AssertionError(name)

with threadpool_limits(2):
    point=module(ROOT/'joint_neural/point_mean_nn.py','qa_point_mean')
    cycle=module(V2/'neural_route/cycle_neural.py','qa_original_cycle')
    joint=module(ROOT/'joint_kernel.py','qa_joint_kernel')
    calibrate=module(ROOT/'calibrate_fitting_head.py','qa_calibrate_fit')
    original_fit=module(V2/'neural_route/fitting_full60.py','qa_original_fit')
    sources=[ROOT/'joint_neural/point_mean_nn.py',ROOT/'joint_kernel.py',ROOT/'calibrate_fitting_head.py']
    initial_sha={str(p.relative_to(WORK)):digest(p) for p in sources}
    split=read(V1/'cache_v1/split.json');keep=split['train']+split['validation']
    # An excluded row with NaNs must never become part of the selected numerical tensor.
    synthetic=np.full((60,3,4),np.nan,dtype=np.float32)
    for i in keep:synthetic[i]=i
    raw=io.BytesIO();np.savez_compressed(raw,field=synthetic);raw.seek(0)
    selected=point.selected_rows(raw,'field',keep)
    check('selected_rows skips all excluded numeric NaN label rows',np.array_equal(selected[:,0,0],keep))
    cache=V1/'cache_v1/rotor.npz'
    fields=point.selected_rows(cache,'field',keep).astype(np.float64);truth=np.hypot(fields[...,0::2],fields[...,1::2])
    with np.load(cache,allow_pickle=False) as z:conditions=z['conditions'];ids=z['point_id'];xy=z['xy']
    prep=ROOT/'joint_neural/point_prepared';identity=read(prep/'identity.json');a=load(prep/'mean_data.npz');base=load(prep/'baseline_model.npz')
    check('prepared label identity is train42 + validation9',identity['label_access']==keep)
    check('identity excludes original9 test',identity['excluded_case_ids']==split['test'])
    check('identity explicitly forbids test and fitting labels',not identity['test_labels_loaded'] and not identity['full60_fit_labels_loaded'])
    check('raw cache SHA bound',identity['cache_sha256']==digest(cache))
    check('split SHA bound',identity['split_sha256']==digest(V1/'cache_v1/split.json'))
    check('mean scalar shape',(a['magnitude_mean'].shape==(51,1800,1)))
    check('mean vector shape',(a['signed_mean'].shape==(51,1800,2)))
    check('scalar means exactly source hypot then time mean',np.array_equal(a['magnitude_mean'],truth.mean(1)[...,None]))
    check('vector means exactly source component then time mean',np.array_equal(a['signed_mean'],fields.mean(1).reshape(51,1800,2)))
    check('training/validation geometry uses original point IDs',np.array_equal(load(prep/'geometry.npz')['point_id'],ids))
    c=conditions[keep]
    original_baseline=cycle.predict(V2/'neural_route/rotor/selected_model.npz',c)
    copied_baseline=point.baseline_predict(base,c)
    base_err=float(np.max(np.abs(original_baseline-copied_baseline)))
    check('copied NumPy baseline equals original V2 all51 inputs',base_err<1e-10,base_err)
    norm=np.hypot(copied_baseline[...,0::2],copied_baseline[...,1::2]);dynamic=norm-norm.mean(1,keepdims=True)
    dynamic_truth=truth[:42]-truth[:42].mean(1,keepdims=True)
    alpha=np.clip(np.sum(dynamic[:42]*dynamic_truth)/np.sum(dynamic[:42]**2),0,1)
    check('dynamic scalar is independent train42-only LS',float(a['magnitude_dynamic_scale'])==float(alpha),float(alpha))
    check('prepared val dynamic numerical match',np.max(np.abs(a['validation_dynamic']-dynamic[42:]))<1e-10,float(np.max(np.abs(a['validation_dynamic']-dynamic[42:]))))
    vector=copied_baseline.reshape(51,60,1800,2);vector=vector-vector.mean(1,keepdims=True)
    vectruth=fields[:42].reshape(42,60,1800,2);vectruth-=vectruth.mean(1,keepdims=True)
    alpha_vector=np.clip(np.sum(vector[:42]*vectruth)/np.sum(vector[:42]**2),0,1)
    check('signed scalar is independent train42-only LS',float(a['signed_dynamic_scale'])==float(alpha_vector),float(alpha_vector))
    check('prepared signed val dynamic numerical match',np.max(np.abs(a['validation_signed_dynamic']-vector[42:]))<1e-10,float(np.max(np.abs(a['validation_signed_dynamic']-vector[42:]))))
    for rp in sorted((ROOT/'joint_neural/point_runs').glob('*/result.json')):
        result=read(rp);mp=rp.parent/'model.npz';m=point.load_model(mp)
        check(rp.parent.name+' model SHA',result['model_sha256']==digest(mp))
        check(rp.parent.name+' data scope declared',not result['test_labels_loaded'] and not result['full60_fit_labels_loaded'])
        mf=m['feature_mean'];fs=m['feature_scale'];kind=str(m['feature_kind']);cs=float(m['current_scale'])
        allfeatures=point.fvalues(c,kind,cs)
        check(rp.parent.name+' feature means train42 only',np.array_equal(mf,allfeatures[:42].mean(0)))
        check(rp.parent.name+' feature scales train42 only',np.array_equal(fs,np.maximum(allfeatures[:42].std(0),1e-8)))
        target=a['magnitude_mean'] if str(m['target_kind'])=='magnitude' else a['signed_mean']
        check(rp.parent.name+' output mean train42 only',np.array_equal(m['output_mean'],target[:42].mean(0)))
        s=target[:42].std(0);s=np.maximum(s,s.max()*.001)
        check(rp.parent.name+' output scale train42 only',np.array_equal(m['output_scale'],s))
        # Separate matrix evaluation of selected point heads, independent of einsum/BMM.
        predictions=point.predict_mean(m,c[[0,3,45]])
        for j in [0,300,1359,1751,1799]:
            v=(point.fvalues(c[[0,3,45]],kind,cs)-mf)/fs
            for layer in range(int(m['layer_count'])):
                v=v@m[f'w{layer}'][j]+m[f'b{layer}'][j]
                if layer<int(m['layer_count'])-1:v=np.tanh(v)
            v=v*m['output_scale'][j]+m['output_mean'][j]
            check(rp.parent.name+f' independent point{j} forward',np.max(np.abs(v-predictions[:,j]))<1e-12,float(np.max(np.abs(v-predictions[:,j]))))
        rawpred=point.predict(mp,c[42:]);actual=point.exact_metrics(rawpred,truth[42:],np.asarray(split['validation']))
        for metric in ['rmse_T','mae_T','max_abs_T','relative_L2_pct']:
            check(rp.parent.name+' raw validation '+metric,np.isclose(actual[metric],result['validation'][metric],rtol=2e-12,atol=1e-12),actual[metric])
        positive=point.predict(mp,c[42:],nonnegative=True)
        check(rp.parent.name+' projection equals explicit max0',np.array_equal(positive,np.maximum(rawpred,0)))
        frame_error=float(np.max(np.abs(point.predict(mp,c[42],frame_index=59)-rawpred[0,59])))
        check(rp.parent.name+' frame extraction numerical equivalence',frame_error<1e-10,frame_error)
        check(rp.parent.name+' raw and projected diagnostics separate','validation_nonnegative' in result)
    # Existing joint-kernel pilot is real96 finite candidate configs.
    pilot=ROOT/'joint_kernel_pilot';ps=read(pilot/'summary.json');pp=read(pilot/'protocol.json');ranking=read(pilot/'ranking.json')
    count=len(pp['settings'])*len(pp['lengths'])*len(pp['kinds'])*len(pp['ridges'])
    check('joint kernel pilot exact candidate count',count==ps['completed']==len(ranking)==96)
    check('joint kernel source SHA exact',pp['source_sha256']==ps['source_sha256']==digest(ROOT/'joint_kernel.py'))
    check('joint kernel model SHA exact',ps['model_sha256']==digest(pilot/'best_model.npz'))
    check('joint kernel no test labels',not pp['test_labels_opened'] and not ps['test_labels_opened'])
    check('joint kernel train42 / val9 identities',pp['training_cases']==split['train'] and pp['validation_cases']==split['validation'])
    check('joint inducing inputs are train2520 only',max(pp['train_only_inducing_indices'])<2520)
    jpred=joint.predict(pilot/'best_model.npz',conditions[split['validation']])
    saved=load(pilot/'best_validation_predictions.npz')
    check('joint saved prediction independent replay',np.allclose(jpred,saved['prediction'],rtol=2e-9,atol=2e-9),float(np.max(np.abs(jpred-saved['prediction']))))
    jscore=joint.exact_metrics(jpred,truth[42:])
    for name in ['rmse_T','mae_T','max_abs_T','relative_L2_pct']:
        check('joint original T metric '+name,np.isclose(jscore[name],ps['best']['validation'][name],rtol=2e-12,atol=1e-12),jscore[name])
    check('joint best still worse than V2 norm baseline',jscore['relative_L2_pct']>2.632302901723209,jscore['relative_L2_pct'])
    # No fitting target field is opened here. Only pre-existing model parameters
    # and synthetic input conditions validate the independent fit-forward formula.
    synthetic_conditions=np.asarray([[0.,0.],[121.,41.],[333.,72.]])
    for region in ['rotor','stator','airgap']:
        m=load(V2/f'neural_route/fitting_full60/{region}/model.npz')
        p=calibrate.predict(m,synthetic_conditions,nonnegative=False)
        q=original_fit.predict(m,synthetic_conditions,nonnegative=False)
        check('calibrate fitting '+region+' unchanged MLP formula exact',np.array_equal(p,q),float(np.max(np.abs(p-q))))
        deployed=calibrate.predict(m,synthetic_conditions,nonnegative=True)
        check('calibrate fitting '+region+' correct sign policy',np.array_equal(deployed,np.maximum(p,0) if str(m['quantity'])=='|B|' else p))
        check('calibrate fitting '+region+' fixed-frame shape',calibrate.predict(m,synthetic_conditions[0],frame_index=59).shape==p.shape[-1:])
    source_text=(ROOT/'calibrate_fitting_head.py').read_text('utf-8')
    check('fitting freeze guard precedes any np.load',source_text.index("if f.get('test_labels_used_for_selection')")<source_text.index('with np.load(source'))
    check('fitting head is last existing NN layer',"m[f'w{last}']=head[:-1]" in source_text and "m[f'b{last}']=head[-1]" in source_text)
    check('fitting report distinguishes raw/projected/unseen scope',all(s in source_text for s in ["fitting_raw=", "fitting_deployed=", "unseen_prediction_claim=False"]))
    check('all reviewed sources unchanged during QA',all(digest(p)==initial_sha[str(p.relative_to(WORK))] for p in sources))
    report=dict(status='PASS',pass_count=len(checks),checks=checks,source_sha256=initial_sha,
        scope='No training executed. Test labels not read. No all60 fitting labels read; only fitting source model weights and synthetic inputs used to verify formula.',
        fitting_guard_hardening_suggestion='Before executing, verify prediction freeze sidecar and hashes of all declared sources/models; verify V2 fitting model sidecar and source cache identity. Current guard checks explicit JSON scope before all60 read, but lacks these stronger integrity bindings.',
        assumptions='Prediction model inputs and dynamic maps retain original fixed60 frames and original sample points. Reused validation9 is exploratory; neural mean model scores are not new blind generalization.',
        pilot_summary=dict(candidate_count=count,best_validation_L2_pct=jscore['relative_L2_pct'],new_improvement=False))
    with (OUT/'candidate_code_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (OUT/'candidate_code_audit.md').open('x',encoding='utf-8') as f:
        f.write(f"# V3 候选代码独立复核\n\n{len(checks)} 项数值/身份/协议检查通过。未训练、未读取测试标签、未读取全60拟合标签。\n\n")
        f.write("点均值神经网络：train42 构造输入与输出尺度，只有42个训练工况进入损失；9验证仅选择检查点。标量目标是先 hypot 再时间均值，向量目标是先对 Br/Bt 时间均值，后加有符号动态，再求 hypot。NumPy baseline、逐点独立前向、raw/deployed指标均核实。\n\n")
        f.write(f"联合空间核：96个实际候选，最佳验证 L2={jscore['relative_L2_pct']:.6f}%，未改进V2。采用训练POD和训练诱导点，未从验证拟合基底。\n\n")
        f.write("全60拟合 head 校准尚不运行。其新 forward 与原拟合 MLP 完全一致；LS 更新最后线性层是对既有神经网络的监督拟合，raw与非负投影分报、气隙有符号Br保留。必须先冻结预测，且不得把全60拟合分数反馈预测选型。建议加强冻结和源文件SHA守卫后再执行。\n")
    print(json.dumps(dict(status='PASS',pass_count=len(checks),report=str(OUT/'candidate_code_audit.json')),ensure_ascii=False))
