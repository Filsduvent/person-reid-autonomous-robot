"""CPU-only exact rejected-sample, ownership and durable status checks."""
import json
import os
import pytest
import torch
import lodo_cuda_admission_diagnostic as log
from reid.utils import lodo_device as dev


@pytest.fixture
def inventory(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',log.TITAN_UUID)
    monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID',log.TITAN_UUID)
    data=dict(total='12288',used='5',free='12179',util='0',apps=[])
    def query(fields,kind='gpu'):
        if kind=='compute-apps':return data['apps']
        row=['0',log.TITAN_UUID,'TITAN Xp','570',data['total']]
        if fields==log.FIELDS:row.append(data['used'])
        return [row+[data['free'],data['util'],'Default','[N/A]']]
    monkeypatch.setattr(dev,'_query',query)
    return data


def journal(tmp_path):
    return log.AdmissionJournal(tmp_path/'admissions',owner_lookup=lambda p:f'{p} owner process')


def record(tmp_path):return json.loads((tmp_path/'admissions/0001.json').read_text())


def test_accept_keeps_exact_guard_samples_and_no_cuda(inventory,tmp_path):
    j=journal(tmp_path);r=j.check(log.TITAN_UUID)
    saved=record(tmp_path)
    assert r['free_mib']==12179 and saved['decision']=='ACCEPT'
    assert saved['before']['target']['used_mib']==5 and saved['after']['target']['free_mib']==12179
    assert len(saved['guard_queries'])==2 and saved['guard_queries'][0]['rows'][0][6]=='0'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('change,expected',[('util','activity/headroom'),('free','headroom'),('apps','occupied')])
def test_rejection_persists_original_sample_before_raising(inventory,tmp_path,change,expected):
    if change=='util':inventory['util']='9'
    elif change=='free':inventory['free']='7000'
    else:inventory['apps']=[[log.TITAN_UUID,'1234','other','5000']]
    j=journal(tmp_path)
    with pytest.raises(ValueError,match=expected):j.check(log.TITAN_UUID)
    saved=record(tmp_path)
    assert saved['decision']=='REJECT' and saved['rejection_reason'] and saved['rejection_check']
    assert saved['guard_queries'] and saved['before'] and saved['after']
    assert j.sequence==1  # Never retries the rejected decision.
    if change=='util':assert saved['guard_queries'][0]['rows'][0][6]=='9'
    if change=='apps':assert saved['before']['ownership']['1234']=='1234 owner process'


@pytest.mark.parametrize('field',['free','total','used','util'])
def test_missing_metric_retains_raw_inventory_and_fails_closed(inventory,tmp_path,field):
    inventory[field]='[N/A]';j=journal(tmp_path)
    with pytest.raises(ValueError,match='metrics'):j.check(log.TITAN_UUID)
    saved=record(tmp_path)
    assert saved['decision']=='REJECT' and saved['before']['gpu_rows']
    assert not saved['guard_queries'] and saved['after_error']


@pytest.mark.parametrize('wrong',['wrong_uuid','visibility','authorization'])
def test_wrong_uuid_or_declarations_prevent_guard(inventory,tmp_path,monkeypatch,wrong):
    value=log.TITAN_UUID
    if wrong=='wrong_uuid':value=log.GTX_UUID
    elif wrong=='visibility':monkeypatch.setenv('CUDA_VISIBLE_DEVICES',log.GTX_UUID)
    else:monkeypatch.delenv('LODO_AUTHORIZED_GPU_UUID')
    j=journal(tmp_path)
    with pytest.raises(ValueError):j.check(value)
    assert record(tmp_path)['decision']=='REJECT' and not record(tmp_path)['guard_queries']


def test_no_overwrite_atomic_and_symlink_rejection(tmp_path):
    p=tmp_path/'evidence.json';log.write_once_json(p,{'status':'STOPPED'})
    with pytest.raises(FileExistsError):log.write_once_json(p,{'status':'PASS'})
    assert json.loads(p.read_text())=={'status':'STOPPED'}
    q=tmp_path/'alias.json';q.symlink_to(p)
    with pytest.raises(ValueError):log.write_once_json(q,{})
    assert not list(tmp_path.glob('.admission-*'))


def test_final_audit_cannot_mask_stopped_publication(tmp_path):
    def fail():raise ValueError('occupied')
    summary={'status':'STOPPED','original_error':'primary'}
    log.publish_final_status(tmp_path/'final.json',summary,fail)
    value=json.loads((tmp_path/'final.json').read_text())
    assert value['status']=='STOPPED' and value['original_error']=='primary'
    assert value['final_audit_error']['reason']=='occupied'


def test_self_only_update_guard_and_unknown_owner_fail_closed(inventory,tmp_path):
    inventory.update(util='90',free='4000',apps=[[log.TITAN_UUID,str(os.getpid()),'self','7000']])
    j=journal(tmp_path);assert j.check(log.TITAN_UUID,allow_pid=os.getpid())['processes']
    j.owner_lookup=lambda pid:None
    with pytest.raises(ValueError,match='ownership'):j.check(log.TITAN_UUID,allow_pid=os.getpid())
    assert json.loads((tmp_path/'admissions/0002.json').read_text())['decision']=='REJECT'


def test_threshold_cannot_be_weakened(tmp_path):
    with pytest.raises(ValueError):log.AdmissionJournal(tmp_path/'bad',min_free_mib=1)


@pytest.mark.parametrize('key,value',[('uuid',log.GTX_UUID),('allocated',False),('authorized',False),('phase','old_gtx'),('source','inferred_idle'),('confirmation_text','')])
def test_campaign_needs_separate_exact_allocation_receipt(tmp_path,key,value):
    from lodo_cuda_titan_abc_campaign import require_allocation
    receipt=dict(uuid=log.TITAN_UUID,allocated=True,authorized=True,phase='phase9r1_titan_abc',source='user',confirmation_text='Explicit confirmation')
    path=tmp_path/'receipt.json';path.write_text(json.dumps(receipt))
    assert require_allocation(path)==receipt
    receipt[key]=value;path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='confirmation'):require_allocation(path)
    assert not torch.cuda.is_initialized()


def test_missing_allocation_receipt_stops_before_inventory_or_output(tmp_path,monkeypatch):
    import lodo_cuda_titan_abc_campaign as campaign
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):campaign.run(tmp_path/'absent.json')
    assert not (tmp_path/'exp').exists() and not torch.cuda.is_initialized()
