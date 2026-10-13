from pathlib import Path
import json,hashlib,subprocess,sys,copy,gc,collections,pickle,platform,datetime,shutil
import torch
from importlib.metadata import version
root=Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo').resolve();sys.path[:0]=[str(root),str(root/'scripts')]
from lodo_official_preparation import metadata_only
from reid.utils.lodo_checkpoint import file_digest,digest,read_checkpoint
from reid.data.multi_source import canonical_json
from reid.utils.config import load_config,validate_model_loss_requirements
from reid.models.build import build_model
from reid.losses.build import build_criterion
from reid.optim.build import build_optimizer,build_center_optimizer,build_scheduler
from reid.utils.lodo_comparison import _compatible_reference
p=root/'docs/lodo_phase10_evidence';index=json.loads((p/'candidate-index.json').read_text())
prior=root/'exp/lodo_smoke/phase9r3-titan-20261010T005917'
manifest=json.loads((prior/'final-evidence-manifest.json').read_text())
for name,sha in manifest['sha256'].items():assert file_digest(prior/name)==sha,name
executed=json.loads((prior/'executed-source-sha256.json').read_text());executed.update(json.loads((prior/'additional-source-sha256.json').read_text()))
for name,sha in executed.items():assert file_digest(root/name)==sha,name
qualification=json.loads((prior/'campaign-summary.json').read_text());assert qualification['status']=='COMPLETED'
actual=[]
for arch in ('resnet50','pcb'):
 a=qualification['architectures'][arch];assert a['status']=='PASS' and a['terminal_noop']
 assert all(row['exact'] and row['tolerance_pass'] and row['max_abs']==0 for row in a['abc']+a['replay'])
 common=json.loads((prior/arch/'LODO-1/common.json').read_text());checkpoint=Path(common['immutable_checkpoint'])
 for path in (checkpoint,prior/arch/'LODO-1/branch-A/checkpoints/ckpt_terminal.pth'):
  payload,proof=read_checkpoint(path)
  assert payload['lodo_schema_version']==2 and proof['schema_version']==2
  assert payload['identity']['recovery']['capability']=='epoch_boundary_rollback_cuda_v1'
  assert payload['identity']['scientific_config']['execution_policy']['name']=='strict_cuda_deterministic_v1'
  assert payload['progress']['completed_primary_updates']==(2 if payload['kind']=='recovery' else 5)
  if arch=='resnet50':assert payload['center']['centers'].shape==(2510,2048) and payload['center_optimizer'] is not None
  else:assert payload['center'] is None and payload['center_optimizer'] is None
  if payload['kind']=='recovery':assert proof['checkpoint_sha256']==common['sha256']
  actual.append(dict(architecture=arch,kind=payload['kind'],path=str(path),checkpoint_sha256=proof['checkpoint_sha256'],primary=payload['progress']['completed_primary_updates'],auxiliary=payload['progress']['auxiliary_steps'],schema=2))
  del payload;gc.collect()
(p/'qualification-reverified.json').write_bytes(canonical_json(dict(status='PASS',qualification_scope='LODO-1/five updates/two-batch fixture epochs/zero workers/recorded TITAN runtime',manifest_hashes_verified=len(manifest['sha256']),executed_source_hashes_verified=len(executed),abc_events={a:sum(x['events'] for x in qualification['architectures'][a]['abc']) for a in ('resnet50','pcb')},actual_cpu_reads=actual,cuda_initialized=torch.cuda.is_initialized())))
models=[]
with metadata_only():
 for row in index['runs']:
  cfg=load_config(str(root/row['config']));classes=row['identities']
  model=build_model(cfg,num_classes=classes,initialize_pretrained=False)
  dim=validate_model_loss_requirements(cfg,model);criterion=build_criterion(cfg,num_classes=classes,feat_dim=dim)
  optimizer=build_optimizer(cfg,model);aux=build_center_optimizer(cfg,criterion)
  scheduler=build_scheduler(cfg,optimizer,steps_per_epoch=row['initial_epoch_batches'])
  heads=[model.classifier] if row['architecture']=='resnet50' else list(model.classifiers.fc_list)
  assert len(heads)==(1 if row['architecture']=='resnet50' else 6)
  assert all(head.out_features==classes for head in heads)
  if aux is not None:assert criterion.center_loss.centers.shape==(classes,2048) and aux.param_groups[0]['lr']==.5
  points=[1,500,501,28400,28401,49700,49701,85200] if row['architecture']=='resnet50' else [1,31400,31401,94200]
  rates={str(u):list(scheduler.rates_after(u-1)) for u in points}
  if row['architecture']=='resnet50':
   assert all(abs(x-.000299406)<1e-12 for x in rates['500']);assert all(abs(x-.0003)<1e-12 for x in rates['501'])
  else:assert set(rates['31400'])=={.01,.1} and all(any(abs(x-y)<1e-12 for x in rates['31401']) for y in (.001,.01))
  models.append(dict(architecture=row['architecture'],fold=row['fold'],classes=classes,classifier_shapes=[list(head.weight.shape) for head in heads],center_shape=list(criterion.center_loss.centers.shape) if aux is not None else None,auxiliary_optimizer=type(aux).__name__ if aux is not None else None,main_optimizer=type(optimizer).__name__,lr_used=rates,pretrained_initialization_executed=False,forward_backward_updates=0))
  del model,criterion,optimizer,aux,scheduler,heads;gc.collect()
(p/'model-and-lr-preflight.json').write_bytes(canonical_json(dict(status='PASS',runs=models,cuda_initialized=False)))
original=Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot').resolve()
records=sorted((original/'results').rglob('cross_dataset.json'))+sorted((original/'exp').rglob('cross_dataset.json'))
comparisons=[]
for row in index['runs']:
 m=json.loads((root/row['manifest']).read_text())['immutable'];compatible=[];issues=[];found=[]
 probe=dict(architecture=row['architecture'],variant=m['configuration']['scientific']['model'].get('variant'),sources=row['sources'],target=row['target'],representation=m['evaluation']['representation'],evaluation=dict(configuration=dict(target=m['evaluation']['target'])))
 for path in records:
  data=json.loads(path.read_text());architecture='pcb' if data['architecture']=='pcb' else 'resnet50'
  if architecture!=row['architecture'] or data['target_dataset']!=row['target'] or data['source_dataset'] not in row['sources']:continue
  found.append(str(path));reason=None
  try:_compatible_reference(data,probe);compatible.append(data['source_dataset'])
  except (ValueError,KeyError,TypeError) as error:reason=str(error)
  issues.append(dict(path=str(path),sha256=file_digest(path),source=data['source_dataset'],target=data['target_dataset'],eligible=reason is None,reason=reason,has_historical_target_hashes=bool(data.get('evaluation_provenance',{}).get('target_metadata_hashes')),has_historical_camera_filter=data.get('evaluation_provenance',{}).get('camera_filter'),checkpoint_exists=Path(data['checkpoint_path']).is_file()))
 comparisons.append(dict(architecture=row['architecture'],fold=row['fold'],required_sources=row['sources'],historical_records_found=len(found),eligible_sources=compatible,missing_eligible_sources=[s for s in row['sources'] if s not in compatible],status='complete' if len(compatible)==3 else 'incomplete',records=issues))
(p/'stage-a-reference-eligibility.json').write_bytes(canonical_json(dict(discovered_records=len(records),runs=comparisons,original_records_modified=False,standalone_stage_b_evaluation='independent of missing comparison records')))
data_root=Path('/home/addirakoze/nobackup/Projects/Dataset').resolve();cuhk=data_root/'cuhk03';parts=pickle.loads((cuhk/'detected/partitions.pkl').read_bytes(),encoding='latin1');new=json.loads((cuhk/'splits_new_detected.json').read_text())[0]
signatures=[]
for role,split in [('train','trainval'),('query','test'),('gallery','test')]:
 names=[x.decode() if isinstance(x,bytes) else str(x) for x in parts[split+'_im_names']]
 if role!='train':names=[n for n,mark in zip(names,parts['test_marks']) if mark==(0 if role=='query' else 1)]
 old=collections.Counter(int(n[:8]) for n in names);ref=collections.Counter(r[1] for r in new[role])
 signatures.append(dict(role=role,processed_images=len(names),processed_identities=len(old),reference_images=len(new[role]),reference_identities=len(ref),same_sorted_per_identity_counts=sorted(old.values())==sorted(ref.values()),same_pid_sets=set(old)==set(ref)))
(p/'cuhk03-provenance.json').write_bytes(canonical_json(dict(status='UNVERIFIED_OFFICIAL_NEW',descriptor=dict(image_type='detected',protocol='processed_partition',split_id=None),reference_sha256={str(q):file_digest(q) for q in (cuhk/'detected/partitions.pkl',cuhk/'splits_new_detected.json',cuhk/'cuhk03_new_protocol_config_detected.mat')},membership_signatures=signatures,missing_evidence='No verified raw PNG/MAT identifier to processed JPG/PID conversion manifest or generator; counts/signatures are consistent but do not prove role-by-role sample membership.',advisor_decision='Accept exact hashed detected processed partition with explicit limitation, or require separately reviewed provenance/conversion work.')))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
owners=[]
for line in apps.strip().splitlines():
 pid=line.split(',')[1].strip();owners.append(subprocess.check_output(['ps','-p',pid,'-o','pid=,user=,args='],text=True))
(p/'resource-audit.json').write_bytes(canonical_json(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_inventory_csv=gpu,compute_apps_csv=apps,owners=owners,allocation='NONE_AUTHORIZED_IN_PHASE10',disk=shutil.disk_usage(root)._asdict(),cuda_initialized=torch.cuda.is_initialized(),runtime=dict(host=platform.node(),python=platform.python_version(),torch=str(torch.__version__),torchvision=version('torchvision'),numpy=version('numpy'),cuda_build=torch.version.cuda,cudnn_version=torch.backends.cudnn.version()))))
# Protect current result bytes in addition to all baseline sources/artifacts.
result_before=json.loads(Path('/tmp/lodo-phase10-stage-a-results-before.json').read_text());assert all(file_digest(original/n)==h for n,h in result_before.items())
(p/'original-stage-a-result-hashes.json').write_bytes(canonical_json(result_before))
# Review patch includes all production changes plus new preparation/test sources.
patch=subprocess.check_output(['git','diff','HEAD','--','.gitignore','reid','scripts'],cwd=root)
import difflib
before=json.loads(Path('/tmp/lodo-phase10-before.json').read_text())['roots'][str(root)]
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=root,text=True).split('\0'))
for rel in sorted(set(before['sources'])|{'scripts/lodo_official_preparation.py','tests/test_lodo_official_preparation.py'}):
 if rel in tracked or not rel.endswith('.py'):continue
 text=(root/rel).read_text();patch+=''.join(difflib.unified_diff([],text.splitlines(True),fromfile='/dev/null',tofile='b/'+rel)).encode()
(p/'implementation-review.patch').write_bytes(patch)
(p/'pre-edit-state.json').write_bytes(Path('/tmp/lodo-phase10-before.json').read_bytes())
(p/'preparation-generator.py').write_bytes(Path('/tmp/lodo_phase10_generate.py').read_bytes())
(p/'read-only-audit.py').write_bytes(Path('/tmp/lodo_phase10_audit.py').read_bytes())
print('AUDIT PASS:',len(actual),'actual v2 CPU reads;',len(models),'model/LR dimensions;',len(records),'Stage A records; no CUDA/image/training/evaluation access',flush=True)
