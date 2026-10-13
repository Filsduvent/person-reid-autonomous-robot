"""CPU checks for forensic comparison direction and optimizer parameter identity."""
import copy
import pytest
import torch
import numpy as np

from lodo_cuda_numerical_diagnostic import tensor_difference,parameter_associations,nested_equal


@pytest.mark.parametrize('reference,actual',[
    ([0.,1.,-1.],[0.,1.,-1.]),
    ([0.,1.,-1.],[.0002,1.0005,-1.002]),
    ([0.,1000.,.00000001],[.00001,1001.05,.0002]),
])
def test_comparison_matches_original_torch_assert_close(reference,actual):
    reference=torch.tensor(reference);actual=torch.tensor(actual)
    result=tensor_difference(reference,actual)
    assert result['outside']==int((~torch.isclose(actual,reference,rtol=.001,atol=.0001)).sum())
    try:torch.testing.assert_close(actual,reference,rtol=.001,atol=.0001)
    except AssertionError:assert result['outside']>0
    else:assert result['outside']==0


def test_relative_error_zero_reference_reported_separately():
    result=tensor_difference(torch.tensor([0.,1.]),torch.tensor([1.,2.]))
    assert result['changed_zero_reference']==1 and result['max_relative_nonzero_reference']==1
    assert result['outside']==2 and result['max_abs']==1


def test_integers_require_exact_equality():
    result=tensor_difference(torch.tensor([5,6]),torch.tensor([5,7]))
    assert not result['exact'] and result['outside']==1


@pytest.mark.parametrize('change',['name','identifier','contract'])
def test_parameter_associations_reject_ambiguous_ownership(change):
    p={'optimizer':{'param_groups':[{'params':[0],'param_name':'weight'},{'params':[1],'param_name':'bias'}],
                    'state':{0:{},1:{}}},
       'identity':{'optimizer':{'groups':[{'parameters':['weight']},{'parameters':['bias']}]}}}
    assert parameter_associations(p)=={0:{'parameter':'weight','group':0},1:{'parameter':'bias','group':1}}
    p=copy.deepcopy(p)
    if change=='name':p['optimizer']['param_groups'][0]['param_name']='bias'
    if change=='identifier':p['optimizer']['param_groups'][1]['params']=[0]
    if change=='contract':p['identity']['optimizer']['groups'][1]['parameters']=['weight']
    with pytest.raises(AssertionError):parameter_associations(p)


def test_forensic_comparison_does_not_draw_rng():
    value=torch.arange(10,dtype=torch.float32);state=torch.get_rng_state().clone()
    tensor_difference(value,value)
    assert torch.equal(state,torch.get_rng_state()) and not torch.cuda.is_initialized()


def test_nested_rng_arrays_compare_values_without_truth_ambiguity():
    state={'array':np.arange(5),'tensor':torch.arange(5),'sequence':(1,[2,3])}
    assert nested_equal(state,copy.deepcopy(state))
    changed=copy.deepcopy(state);changed['array'][2]=8
    assert not nested_equal(state,changed)
