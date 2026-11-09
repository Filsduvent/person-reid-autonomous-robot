"""Approved MGN LODO v2 science. Pure validation; no launch or v1 migration."""
import copy
import hashlib
import json
import re
from pathlib import Path

from reid.utils.lodo_protocol import (_at, _expect, _keys, _descriptor, validate_fold,
                                      _canonical_numbers)

SCIENTIFIC_PROTOCOL = 'mgn_lodo_v2'
VERSION = 2
BUDGET = 85200


def is_mgn_lodo(cfg):
    p = cfg.get('protocol', {})
    return (isinstance(p, dict) and type(p.get('version')) is int and p['version'] == VERSION
            and p.get('name') == 'multi_source_lodo' and p.get('architecture') == 'mgn')


def contract(cfg):
    _keys(_at(cfg, 'protocol'), {'name','version','architecture','fold'}, 'protocol')
    for path, value in {'protocol.name':'multi_source_lodo', 'protocol.version':2,
            'protocol.architecture':'mgn', 'system.device':'cuda', 'system.gpu_id':0,
            'system.amp':False, 'repro.seed':42, 'repro.deterministic':True,
            'repro.benchmark':False, 'repro.cuda_policy':'strict_cuda_deterministic_v1',
            'data.num_workers':0, 'data.pin_memory':True,
            'data.train.dataset.name':'multi_source', 'train.mode':'successful_updates',
            'train.max_updates':BUDGET, 'train.evaluate_during_training':False,
            'train.checkpoint_selection':'terminal', 'train.save.save_best':False,
            'train.save.save_last':True, 'train.save.save_terminal':True,
            'train.recovery.version':1, 'train.recovery.policy':'completed_epoch_boundary',
            'train.recovery.every_epochs':1, 'train.recovery.keep_generations':2,
            'model.name':'mgn', 'model.variant':'paper_v3_global2048_independent_v1',
            'model.pretrained':True, 'model.head.classifier':True,
            'model.initialization.artifact_id':'resnet50-19c8e357',
            'model.initialization.sha256':'19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097',
            'model.initialization.head_init_id':'reference_a_384e820_bias_free_v1',
            'loss.id.enabled':True, 'loss.id.weight':1., 'loss.id.label_smoothing':0.,
            'loss.id.head_aggregation':'mean', 'loss.triplet.enabled':True,
            'loss.triplet.weight':1., 'loss.triplet.margin':1.2,
            'loss.triplet.mining':'batch_hard', 'loss.triplet.feature_source':'model',
            'loss.triplet.feature_aggregation':'mean', 'loss.center.enabled':False,
            'optim.name':'sgd', 'optim.lr':.01, 'optim.momentum':.9, 'optim.nesterov':False,
            'optim.weight_decay':.0005, 'optim.bias_lr_factor':1., 'optim.weight_decay_bias':.0005,
            'sched.name':'warmup_multistep', 'sched.unit':'successful_primary_updates',
            'sched.milestones':[42600,63900], 'sched.gamma':.1, 'sched.warmup_iters':0,
            'sched.warmup_factor':1., 'sched.warmup_method':'linear',
            'data.train.batch.sampler':'pk', 'data.train.batch.P':16,
            'data.train.batch.K':4, 'data.train.batch.batch_size':64,
            'data.test.batch.size':32, 'data.test.loader.shuffle':False,
            'eval.flip_aggregation':'none', 'eval.normalize_feat':True,
            'eval.distance':'euclidean', 'eval.topk':[1,5,10], 'eval.rerank.enabled':False, 'eval.weight':'',
            'eval.rerank.k1':20, 'eval.rerank.k2':6, 'eval.rerank.lambda_value':.3,
            'eval.export':{'format':['json']}}.items():
        _expect(cfg, path, value)
    allowed = {'system':{'device','gpu_id','amp','log_interval'},
        'repro':{'seed','deterministic','benchmark','cuda_policy'},
        'data':{'root','num_workers','pin_memory','train','test'},
        'data.train':{'dataset','images','aug','batch','loader'},
        'data.test':{'dataset','images','aug','batch','loader'},
        'data.train.dataset':{'name','sources'}, 'data.train.batch':{'sampler','P','K','batch_size'},
        'data.test.batch':{'size'}, 'data.train.loader':{'shuffle','final_batch'},
        'data.test.loader':{'shuffle','final_batch'},
        'train':{'mode','max_updates','evaluate_during_training','checkpoint_selection','save','recovery'},
        'train.save':{'save_best','save_last','save_terminal','resume'},
        'train.recovery':{'version','policy','every_epochs','keep_generations'},
        'model':{'name','variant','pretrained','weights_path','head','initialization'},
        'model.head':{'classifier'}, 'model.initialization':{'artifact_id','sha256','head_init_id'},
        'loss':{'id','triplet','center'}, 'loss.id':{'enabled','weight','label_smoothing','head_aggregation'},
        'loss.triplet':{'enabled','weight','margin','mining','feature_source','feature_aggregation'},
        'loss.center':{'enabled'}, 'optim':{'name','lr','momentum','nesterov','weight_decay','bias_lr_factor','weight_decay_bias'},
        'sched':{'name','unit','milestones','gamma','warmup_iters','warmup_factor','warmup_method'},
        'eval':{'weight','normalize_feat','distance','topk','rerank','export','flip_aggregation'},
        'eval.rerank':{'enabled','k1','k2','lambda_value'}}
    for path, keys in allowed.items():
        _keys(_at(cfg, path), keys, path)
    _keys(cfg, {'experiment','system','repro','logging','data','model','loss','optim','sched','train','eval','protocol'}, 'config')
    if not isinstance(cfg['data']['root'], str) or not cfg['data']['root'].strip():
        raise ValueError('Nonempty canonical dataset root required')
    if not isinstance(cfg['train']['save']['resume'], str):
        raise ValueError('Resume must be an explicit same-run checkpoint path')
    if not isinstance(cfg['model']['weights_path'], str) or not cfg['model']['weights_path'].strip():
        raise ValueError('Pinned pretrained artifact path required')
    for role in ('train','test'):
        _keys(_at(cfg,f'data.{role}.images'), {'size','batch_dims','interpolation'}, f'data.{role}.images')
        for key, value in {'size':[384,128], 'interpolation':'bicubic', 'batch_dims':'NCHW'}.items():
            _expect(cfg, f'data.{role}.images.{key}', value)
        aug = dict(mean=[.485,.456,.406], std=[.229,.224,.225], mirror='random' if role=='train' else 'none')
        if role=='train':
            aug.update(padding=dict(enabled=False),random_crop=dict(enabled=False),random_erasing=dict(enabled=False))
        _expect(cfg,f'data.{role}.aug',aug)
    _expect(cfg,'data.train.loader.shuffle',True)
    _expect(cfg,'data.train.loader.final_batch',True)
    _expect(cfg,'data.test.loader.final_batch',True)
    descriptors = cfg['data']['train']['dataset']['sources']
    if not isinstance(descriptors,list):
        raise ValueError('Ordered source descriptor list required')
    sources = [_descriptor(s,'source') for s in descriptors]
    target = _descriptor(cfg['data']['test']['dataset'],'target')
    validate_fold(cfg['protocol']['fold'],[s['name'] for s in sources],target['name'])
    # Require byte identities rather than implicitly upgrading unbound descriptors.
    for item in sources+[target]:
        if not isinstance(item.get('metadata_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',item['metadata_sha256']):
            raise ValueError('Pinned partition/list SHA256 required for each descriptor')
    from reid.utils.lodo_recovery import recovery_policy
    from reid.utils.lodo_cuda_policy import policy_contract
    recovery_policy(cfg); policy_contract(cfg)
    return sources,target


def validate_output_path(cfg, *, project_root=None):
    project=Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    value=_at(cfg,'experiment.output_dir')
    _keys(_at(cfg,'experiment'),{'name','output_dir','notes'},'experiment')
    if not isinstance(value,str) or not value or value!=value.strip() or '\0' in value:
        raise ValueError('Invalid MGN LODO output path')
    p=Path(value).expanduser()
    if '..' in p.parts:raise ValueError('Output traversal prohibited')
    p=p if p.is_absolute() else project/p
    root=project/'exp'/'lodo'/SCIENTIFIC_PROTOCOL/'mgn'/cfg['protocol']['fold']/'seed42'
    if p.parent!=root or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',p.name) or p.resolve()!=p or root.resolve()!=root:
        raise ValueError('Use fresh exp/lodo/mgn_lodo_v2/mgn/<fold>/seed42/<run-id> without aliases')
    if p.exists() or p.is_symlink():
        if not p.is_dir() or not cfg['train']['save'].get('resume'):
            raise ValueError('Existing MGN LODO output requires verified same-run resume')
        from reid.utils.lodo_checkpoint import validate_resume_path
        validate_resume_path(cfg,p)
    return p


def validate_lodo_config(cfg, *, project_root=None):
    contract(cfg);validate_output_path(cfg,project_root=project_root)


def scientific_configuration(cfg):
    sources,target=contract(cfg)
    model=copy.deepcopy(cfg['model']);model.pop('weights_path')
    from reid.utils.lodo_cuda_policy import policy_contract
    return _canonical_numbers(dict(identity_schema_version=1,scientific_protocol=SCIENTIFIC_PROTOCOL,
        protocol=copy.deepcopy(cfg['protocol']), sources=sources,target=target,seed=42,
        model=model,loss=copy.deepcopy(cfg['loss']),optim=copy.deepcopy(cfg['optim']),
        sampling=copy.deepcopy(cfg['data']['train']['batch']),update_budget=BUDGET,
        scheduler=copy.deepcopy(cfg['sched']),
        transforms={r:{'images':copy.deepcopy(cfg['data'][r]['images']),'aug':copy.deepcopy(cfg['data'][r]['aug'])} for r in ('train','test')},
        checkpoint_selection='terminal',evaluate_during_training=False,target_bn_adaptation=False,
        precision='float32_tensors_amp_disabled',execution_policy=policy_contract(cfg),num_workers=0,
        recovery=copy.deepcopy(cfg['train']['recovery']),
        retrieval={k:copy.deepcopy(cfg['eval'][k]) for k in ('normalize_feat','distance','topk','rerank','flip_aggregation')},
        architecture_contract=dict(embedding_dim=2048,classifier_heads=8,metric_features=3),
        output_namespace=f"exp/lodo/{SCIENTIFIC_PROTOCOL}/mgn/{cfg['protocol']['fold']}/seed42"))


def serialize_scientific_configuration(cfg):
    return json.dumps(scientific_configuration(cfg),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def configuration_identity(cfg):
    return hashlib.sha256(serialize_scientific_configuration(cfg)).hexdigest()
