"""MGN-v2 held-out plumbing; common reconstruction/ranking, no target BN adaptation."""
import copy
import re
from pathlib import Path
import torch
from reid.utils import lodo_evaluation as common
from reid.utils import mgn_lodo_protocol as protocol
from reid.utils.lodo_checkpoint import file_digest
from reid.utils.lodo_protocol import FOLDS


def validate_request(request):
    required={'mode','version','architecture','fold','target','checkpoint','checkpoint_sha256','expected','output_dir','synthetic','device'}
    common._require(isinstance(request,dict) and required<=request.keys() and not request.keys()-required-{'data_root','batch_size','num_workers','stage_a_records'},'Invalid MGN held-out request fields')
    common._require(request['mode']=='lodo_held_out' and type(request['version']) is int and request['version']==2 and request['architecture']=='mgn','Explicit MGN v2 evaluation required')
    common._require(type(request['synthetic']) is bool,'Explicit synthetic flag required')
    common._require(request['fold'] in FOLDS and request['target']==FOLDS[request['fold']][1],'Wrong held-out fold/target')
    expected=request['expected']
    common._require(isinstance(expected,dict) and expected.keys()=={'config_digest','map_digest','data_digest','num_classes'},'Exact frozen terminal identities required')
    for key in ('config_digest','map_digest','data_digest'):
        common._require(isinstance(expected[key],str) and re.fullmatch('[0-9a-f]{64}',expected[key]),'Invalid frozen digest')
    common._require(type(expected['num_classes']) is int and expected['num_classes']>1,'Invalid combined source C')
    common._require(isinstance(request['checkpoint_sha256'],str) and re.fullmatch('[0-9a-f]{64}',request['checkpoint_sha256']),'Full terminal SHA required')
    common._require(request.get('batch_size',32)==32 and type(request.get('batch_size',32)) is int and type(request.get('num_workers',0)) is int and request.get('num_workers',0)==0,'Frozen evaluation batch32/zero workers required')
    common._require(isinstance(request.get('stage_a_records',[]),list),'Reference paths must be a list')
    common._require(request['device']=='cuda' or (request['synthetic'] and request['device']=='cpu'),'CUDA required for official evaluation; no fallback')


def validate_terminal(request):
    validate_request(request)
    p=Path(request['checkpoint']).absolute()
    common._require(file_digest(p)==request['checkpoint_sha256'],'Terminal SHA mismatch')
    value=common._validate_terminal(request,request_validator=validate_request,
        science_function=protocol.scientific_configuration,identity_function=protocol.configuration_identity)
    payload=value[0]
    common._require(payload['lodo_schema_version']==2 and payload['identity']['schema_version']==2,'MGN strict terminal schema2 required; no legacy upgrade')
    policy=payload['identity'].get('numerical_policy')
    common._require(policy==dict(tensor_dtype='float32',amp=False,convolution_tf32_permitted=True,
                        matmul_tf32_permitted=False,full_fp32_arithmetic_guaranteed=False),'Missing/different numerical policy')
    for phase in ('attempted','forward','backward','successful'):
        e=payload['progress']['exposure'];slots=e['identity_slots'][phase]
        common._require(sum(slots.values())==85200*16 and all(4*n==e['presentations'][phase].get(s,0) for s,n in slots.items()),'PK identity-slot accounting mismatch')
    return value


def metadata(cfg,identity,root):
    return common.target_metadata(cfg,identity,root,science_function=protocol.scientific_configuration)


def output_path(request, project_root=None):
    root=Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    raw=Path(request['output_dir']).expanduser()
    p=raw if raw.is_absolute() else root/raw
    prefix=root/'exp/lodo_eval/mgn_lodo_v2'/('synthetic' if request['synthetic'] else 'benchmark')/'mgn'/request['fold']/'seed42'
    common._require('..' not in raw.parts and p.parent==prefix and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',p.name),'Invalid MGN result namespace')
    common._require(p.resolve()==p and not p.exists() and not p.is_symlink(),'MGN result collision/alias')
    return p


def representation(cfg):
    value=common.retrieval_contract(cfg)
    value.update(embedding='mgn_concatenated_raw_2048',dimension=2048,
                 flip_aggregation='none',target_bn_adaptation=False)
    return value


def execution(request):
    import os
    from reid.utils.lodo_cuda_policy import STRICT_POLICY
    return dict(device=request['device'],precision='float32_tensors_amp_disabled',ranking_device='cpu',
                gpu_uuid=os.environ.get('LODO_AUTHORIZED_GPU_UUID'),cuda_policy=STRICT_POLICY,
                convolution_tf32_permitted=True,actual_tf32_kernel_use_measured=False)


def select_device(request,cfg,output):
    if request['synthetic'] and request['device']=='cpu':return torch.device('cpu'),None,None
    from reid.utils.lodo_device import select_lodo_device
    selected=copy.deepcopy(cfg);selected['train']['save']['resume']=''
    device,guard,evidence=select_lodo_device(selected)
    common._require(device.type=='cuda','MGN CPU fallback prohibited')
    return device,guard,evidence


def evaluate_frozen_bn(cfg,model,loader,device,**kwargs):
    """Eval/no-grad/AMP-disabled with fixed BN buffers and float32 tensors."""
    if torch.is_autocast_enabled(device.type):
        raise ValueError('AMP execution context is incompatible with the frozen precision policy')
    if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Float32 model tensors required')
    model.eval()
    modules=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    if any(not m.track_running_stats for m in modules):
        raise ValueError('Target batch-statistics adaptation prohibited')
    before={name:buf.detach().cpu().clone() for name,buf in model.named_buffers()
            if name.endswith(('running_mean','running_var','num_batches_tracked'))}
    inherited=kwargs.pop('before_batch',None)
    def guard():
        if model.training or any(m.training for m in modules):raise ValueError('Target BatchNorm adaptation prohibited')
        if inherited is not None:inherited()
    def input_policy(module,args):
        if not args or not torch.is_tensor(args[0]) or args[0].dtype!=torch.float32 or args[0].device.type!=device.type:
            raise ValueError('Target input dtype/device differs from frozen extraction policy')
    hook=model.register_forward_pre_hook(input_policy)
    try:
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            scores=common.evaluate_reid(cfg,model,loader,device,before_batch=guard,**kwargs)
    finally:
        hook.remove()
    after=dict(model.named_buffers())
    if any(not torch.equal(value,after[name].detach().cpu()) for name,value in before.items()):
        raise ValueError('Target inference changed BatchNorm buffers')
    return scores


def evaluate_mgn_lodo(request, *, request_path, project_root=None):
    validate_request(request)
    from reid.utils.mgn_lodo_operational import validate_evaluation_admission
    admission=validate_evaluation_admission(request_path,request,root=project_root or Path(__file__).resolve().parents[2])
    from reid.utils.mgn_lodo_comparison import compare_stage_a
    return common._evaluate_lodo(request,project_root=project_root,terminal_validator=validate_terminal,
        output_validator=output_path,metadata_reader=metadata,retrieval_reader=representation,
        device_selector=select_device,execution_reader=execution,evaluation_function=evaluate_frozen_bn,
        comparison_function=compare_stage_a,protocol='mgn_lodo_v2',protocol_version=2,execution_receipt=admission)
