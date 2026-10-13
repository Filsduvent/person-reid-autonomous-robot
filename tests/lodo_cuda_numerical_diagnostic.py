"""NON-AUTHORITATIVE Phase 9R.1 forensic utilities; no tolerance relaxation.

CPU artifact analysis is separate from explicitly authorized CUDA experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np

STATE_RTOL = .001
STATE_ATOL = .0001
LOSS_RTOL = .001
LOSS_ATOL = .001
ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'exp/lodo_smoke/phase9r-20261009T193908'


def tensor_difference(reference, actual, *, rtol=STATE_RTOL, atol=STATE_ATOL):
    """Reference is second assert_close operand: |actual-reference| <= atol+rtol*|reference|.

    Report zero-reference cases separately; never conceal them with an epsilon.
    """
    assert reference.shape == actual.shape and reference.dtype == actual.dtype
    a, b = reference.detach().cpu(), actual.detach().cpu()
    delta = (a.double()-b.double()).abs()
    magnitude = a.double().abs()
    outside = ~torch.isclose(b,a,rtol=rtol,atol=atol,equal_nan=False) if a.is_floating_point() else a != b
    nonzero = magnitude > 0
    relative = delta[nonzero]/magnitude[nonzero]
    return dict(shape=list(a.shape), dtype=str(a.dtype), reference_device=str(reference.device),
                actual_device=str(actual.device), finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                exact=torch.equal(a,b), elements=a.numel(), outside=int(outside.sum()),
                max_abs=float(delta.max()) if a.numel() else 0.,
                max_relative_nonzero_reference=float(relative.max()) if relative.numel() else None,
                changed_zero_reference=int(((delta>0)&~nonzero).sum()),
                relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.double())) if torch.linalg.vector_norm(a.double()) else None,
                reference_abs_max=float(magnitude.max()) if a.numel() else 0.,
                actual_abs_max=float(b.double().abs().max()) if b.numel() else 0.,
                abs_quantiles=[float(x) for x in torch.quantile(delta.flatten(),torch.tensor([.5,.9,.99,1.],dtype=torch.float64))] if a.numel() else [],
                outside_abs_min=float(delta[outside].min()) if outside.any() else None,
                outside_abs_max=float(delta[outside].max()) if outside.any() else None)


def parameter_associations(payload):
    groups=payload['optimizer']['param_groups']; contract=payload['identity']['optimizer']['groups']
    assert len(groups)==len(contract)
    result={}
    for index,(group,row) in enumerate(zip(groups,contract)):
        assert len(group['params'])==len(row['parameters'])
        assert group.get('param_name')==row['parameters'][0]
        for identifier,name in zip(group['params'],row['parameters']):
            assert identifier not in result
            result[identifier]=dict(parameter=name,group=index)
    assert not set(payload['optimizer']['state'])-set(result)
    return result


def nested_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b


def audit_prior_boundaries(output):
    """Cheap separate pass for RNG, retained-index integrity and recorded flags."""
    from reid.utils.lodo_checkpoint import read_checkpoint
    from reid.utils.lodo_recovery import validate_boundary,select_generation
    output=Path(output)
    if output.exists() or output.is_symlink():raise ValueError('Refusing to overwrite diagnostic evidence')
    items=[]
    for kind in ('continuous','interrupted'):
        paths=list((PRIOR/f'resnet50/LODO-1/{kind}/checkpoints').glob('ckpt_epoch_000001_u000000002_*.pth'))
        assert len(paths)==1
        p=paths[0];payload,_=read_checkpoint(p);validate_boundary(payload)
        cfg=payload['cfg'];cfg['train']['save']['resume']=str(p.resolve())
        selected,invalid=select_generation(cfg,p.parent.parent.resolve())
        assert selected==p.resolve() and not invalid
        items.append(payload)
    a,b=items
    report=dict(rng_fields_equal={k:nested_equal(a['rng'][k],b['rng'][k]) for k in a['rng']},
        environment=a['identity']['recovery']['environment'],identity_equal=a['identity']==b['identity'],
        model_parameter_groups_equal=a['optimizer']['param_groups']==b['optimizer']['param_groups'],
        group_flags={k:a['optimizer']['param_groups'][0].get(k) for k in
            ('foreach','fused','capturable','differentiable','amsgrad','betas','eps','weight_decay')},
        both_boundaries_indexed_validated=True,cuda_initialized=torch.cuda.is_initialized())
    assert not report['cuda_initialized']
    with output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


def analyze_prior(output):
    from reid.utils.lodo_checkpoint import read_checkpoint
    output=Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Refusing to overwrite existing diagnostic evidence')
    a,ma=read_checkpoint(PRIOR/'resnet50/LODO-1/continuous/checkpoints/ckpt_terminal.pth')
    b,mb=read_checkpoint(PRIOR/'resnet50/LODO-1/interrupted/checkpoints/ckpt_terminal.pth')
    names=parameter_associations(a);assert names==parameter_associations(b)
    rows=[]
    for identifier,state in a['optimizer']['state'].items():
        for key,value in state.items():
            diff=tensor_difference(b['optimizer']['state'][identifier][key],value)
            rows.append(dict(state_id=identifier,key=key,**names[identifier],**diff))
    failed=[r for r in rows if r['outside']]
    report=dict(non_authoritative=True,criterion=dict(state_rtol=STATE_RTOL,state_atol=STATE_ATOL,loss_rtol=LOSS_RTOL,loss_atol=LOSS_ATOL),
                comparison_direction='assert_close(continuous, resumed); resumed is reference',
                continuous_sha256=ma['checkpoint_sha256'],resumed_sha256=mb['checkpoint_sha256'],
                parameter_associations_match=True,failed_elements=sum(r['outside'] for r in failed),
                failed_tensors=failed,all_optimizer_tensors=rows,
                first_divergent_step='NOT RECORDED in prior terminal-only artifacts')
    assert report['failed_elements']==521
    assert all(r['key']=='exp_avg' for r in report['failed_tensors'])
    report['by_state_key']={key:dict(tensors=sum(r['key']==key for r in rows),
        nonexact_tensors=sum(r['key']==key and not r['exact'] for r in rows),
        failed_elements=sum(r['outside'] for r in rows if r['key']==key),
        max_abs=max(r['max_abs'] for r in rows if r['key']==key))
        for key in sorted({r['key'] for r in rows})}
    boundary=[]
    for kind in ('continuous','interrupted'):
        paths=list((PRIOR/f'resnet50/LODO-1/{kind}/checkpoints').glob('ckpt_epoch_000001_u000000002_*.pth'))
        assert len(paths)==1
        boundary.append(read_checkpoint(paths[0]))
    ba,bma=boundary[0];bb,bmb=boundary[1]
    assert parameter_associations(ba)==parameter_associations(bb)==names
    def tensors(value,prefix=''):
        if torch.is_tensor(value):yield prefix,value
        elif isinstance(value,dict):
            for key,item in value.items():yield from tensors(item,prefix+'.'+str(key))
        elif isinstance(value,(tuple,list)):
            for i,item in enumerate(value):yield from tensors(item,prefix+'.'+str(i))
    report['before_resume_boundary']=dict(continuous_sha256=bma['checkpoint_sha256'],
        interrupted_sha256=bmb['checkpoint_sha256'],completed_updates=2,states={})
    for key in ('model','center','optimizer','center_optimizer','scheduler'):
        first=dict(tensors(ba[key]));second=dict(tensors(bb[key]));assert first.keys()==second.keys()
        diffs=[dict(path=p,**tensor_difference(second[p],first[p])) for p in first]
        report['before_resume_boundary']['states'][key]=dict(
            tensors=len(diffs),nonexact_tensors=sum(not d['exact'] for d in diffs),
            outside=sum(d['outside'] for d in diffs),max_abs=max([d['max_abs'] for d in diffs] or [0.]),
            failed_tensors=[d for d in diffs if d['outside']])
    ca=json.loads((PRIOR/'resnet50/LODO-1/continuous/continuous-outcome.json').read_text())
    cb=json.loads((PRIOR/'resnet50/LODO-1/interrupted/interrupt-outcome.json').read_text())
    report['independent_fresh_trajectories']=dict(
        first_three_inputs_equal=ca['input_sha256'][:3]==cb['input_sha256'],
        continuous_losses=ca['losses'][:3],interrupted_losses=cb['losses'],
        absolute_differences=[abs(x-y) for x,y in zip(ca['losses'],cb['losses'])])
    report['first_divergent_step']='Observable loss difference at update 2 BEFORE resume; earliest operation unknown'
    report['causal_limit']='Prior reference and resumed paths start update 3 from different update-2 states; cannot isolate recovery defect'
    assert not torch.cuda.is_initialized()
    with output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(dict(output=str(output),failed_elements=report['failed_elements'],
        failed_tensors=len(report['failed_tensors']),by_state_key=report['by_state_key'],
        boundary_states={k:{a:b for a,b in v.items() if a!='failed_tensors'}
            for k,v in report['before_resume_boundary']['states'].items()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--analyze-prior',metavar='OUTPUT_JSON')
    group.add_argument('--audit-boundaries',metavar='OUTPUT_JSON')
    args=parser.parse_args()
    if args.analyze_prior:analyze_prior(args.analyze_prior)
    else:audit_prior_boundaries(args.audit_boundaries)
