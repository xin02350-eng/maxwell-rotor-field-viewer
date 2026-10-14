"""Explicit original-case training scope shared by CLI, exporter and delivery."""
from pathlib import Path

LABELS={'original42':'42训练 · 9验证 · 9复用测试','final51':'51训练（含原验证9）· 9复用测试','full60':'全部60训练内拟合 · 继承V3校准模型'}
DEFAULT_PARTITIONS={'original42':['validation','reused_test'],'final51':['reused_test'],'full60':['training_fit']}

def original_split(split):
    raw=split.get('original_ids',split)
    return {key:list(raw[key]) for key in ['train','validation','test']}

def normalize_scope(choice,split):
    key=choice.get('training_scope')
    if key not in LABELS:raise ValueError('Explicit training_scope must be original42/final51/full60: '+choice['id'])
    original=original_split(split);all_ids=original['train']+original['validation']+original['test']
    if len(all_ids)!=60 or len(set(all_ids))!=60:raise ValueError('Original split must contain60 unique caseIDs')
    a=choice.get('train_case_ids');b=choice.get('training_case_ids')
    if a is not None and b is not None and set(a)!=set(b):raise ValueError('Conflicting training caseID aliases')
    train=list(a if a is not None else b if b is not None else all_ids if key=='full60' else [])
    validation=list(choice.get('validation_case_ids',original['validation'] if key=='original42' else []))
    test=list(choice.get('reused_test_case_ids',choice.get('test_case_ids',original['test'] if key!='full60' else [])))
    expect_train=original['train'] if key=='original42' else all_ids if key=='full60' else original['train']+original['validation']
    if set(train)!=set(expect_train) or set(validation)!=set(original['validation'] if key=='original42' else []) or set(test)!=set(original['test'] if key!='full60' else []):
        raise ValueError('Scope and actual original caseIDs disagree: '+choice['id'])
    for values in [train,validation,test]:
        if len(set(values))!=len(values):raise ValueError('Duplicate caseID in training scope')
    if (choice['branch']=='fitting_full60')!=(key=='full60'):raise ValueError('Branch/training scope mismatch')
    return dict(key=key,label=LABELS[key],train_case_ids=train,validation_case_ids=validation,reused_test_case_ids=test,
                original_validation_in_training=key in {'final51','full60'},fresh_blind_test=False)

def case_role(scope,case_id):
    if case_id in scope['train_case_ids']:return 'training_fit' if scope['key']=='full60' else 'train'
    if case_id in scope['validation_case_ids']:return 'validation'
    if case_id in scope['reused_test_case_ids']:return 'reused_test'
    return 'unseen_input'

def scientific_partitions(choice,scope):
    parts=choice.get('scientific_partitions')
    if not isinstance(parts,list) or not parts or len(set(parts))!=len(parts):raise ValueError('Explicit unique scientific_partitions required')
    if any(p not in DEFAULT_PARTITIONS[scope['key']] for p in parts):raise ValueError('Scientific partition not valid for displayed training scope')
    return parts

def partition_case_ids(scope,partition):
    if partition=='training_fit' and scope['key']=='full60':return scope['train_case_ids']
    if partition=='validation':return scope['validation_case_ids']
    if partition=='reused_test':return scope['reused_test_case_ids']
    raise ValueError('Unsupported saved scientific partition')

def scientific_source(root,choice):
    relative=choice.get('scientific_source_relative_root',Path(root).resolve().name)
    expected=choice.get('expected_scientific_choices_sha256','CURRENT')
    if relative!=Path(root).resolve().name and expected=='CURRENT':raise ValueError('Reused science requires its original freeze SHA')
    raw=Path(relative)
    if raw.is_absolute() or '..' in raw.parts or ':' in relative:raise ValueError('Scientific root must be relative to work/')
    source=(Path(root).resolve().parent/raw).resolve()
    if not source.is_relative_to(Path(root).resolve().parent):raise ValueError('Scientific root escaped work/')
    return source,choice.get('scientific_model_id',choice['id']),expected

def select_choice(choices,model_id=None,branch=None,region=None):
    if model_id:
        rows=[m for m in choices if m['id']==model_id]
        if len(rows)!=1:raise ValueError('model-id must uniquely identify one frozen model')
        m=rows[0]
        if branch is not None and branch!=m['branch'] or region is not None and region!=m['region']:raise ValueError('model-id conflicts with explicit region/branch')
        return m
    branch=branch or 'fitting_full60';region=region or 'rotor'
    rows=[m for m in choices if m['branch']==branch and m['region']==region and m.get('role')=='primary']
    if len(rows)!=1:raise ValueError('Default region/branch requires exactly one primary; choose --model-id')
    return rows[0]
