from pathlib import Path
import copy,hashlib,json,subprocess,platform,datetime,sys
import yaml,torch
from importlib.metadata import version
root=Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo').resolve();sys.path[:0]=[str(root),str(root/'scripts')]
from lodo_official_preparation import dataset_evidence,seal
from reid.utils.config import load_config,validate_reid_config
from reid.utils.config_schema import validate_config
from reid.utils.lodo_protocol import FOLDS,scientific_configuration,configuration_identity
from reid.utils.lodo_checkpoint import file_digest,digest
from reid.utils.lodo_evaluation import validate_request,evaluation_output,retrieval_contract,target_metadata
from reid.utils.lodo_comparison import _compatible_reference
base=json.loads(Path('/tmp/lodo-phase10-before.json').read_text())
assert base['roots'][str(root)]['branch']=='feature/lodo-v1'
evidence=root/'docs/lodo_phase10_evidence';evidence.mkdir(exist_ok=False)
(evidence/'maps').mkdir(); (root/'configs/lodo/v1').mkdir(parents=True,exist_ok=False)
(evidence/'manifests').mkdir();(evidence/'evaluation_requests').mkdir()
source_paths={str(p.relative_to(root)) for b in ('reid','scripts','tests','configs') for p in (root/b).rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.py','.yaml','.yml') and not p.is_relative_to(root/'configs/lodo')}
source_hashes={n:file_digest(root/n) for n in sorted(source_paths)}
code=dict(base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),state='dirty_worktree_unfrozen',frozen_commit=None,source_sha256=source_hashes,source_snapshot_sha256=digest(source_hashes))
runtime=dict(host=platform.node(),python=platform.python_version(),torch=str(torch.__version__),torchvision=version('torchvision'),numpy=version('numpy'),cuda_build=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),driver='570.211.01',uuid='GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c',name='NVIDIA TITAN Xp')
def descriptor(n,target=False):
 d=dict(name=n,format='raw' if n=='msmt17' else 'processed',split='test' if target else 'train' if n=='msmt17' else 'trainval')
 if n=='cuhk03':d.update(image_type='detected',protocol='processed_partition',split_id=None,provenance=dict(intended_protocol='detected_new',verification='unverified'))
 if n=='msmt17':d.update(lists=['list_query.txt','list_gallery.txt'] if target else ['list_train.txt'],image_subdir='msmt17/MSMT17_V2/'+('mask_test_v2' if target else 'mask_train_v2'))
 return d
# Read original reference hashes; source/target hashes are subsequently checked by
# the unchanged production preflight including the composite MSMT target hash.
from reid.data.lodo_preflight import _read_inventory
canonical_data=Path('/home/addirakoze/nobackup/Projects/Dataset').resolve(strict=True)
metadata={n:_read_inventory(canonical_data,n)['descriptor_metadata_hashes'] for n in ('market1501','duke','cuhk03','msmt17')}
rows=[];map_files={}
for arch in ('resnet50','pcb'):
 for fold,(sources,target) in FOLDS.items():
  cfg=load_config(str(root/('configs/pcb/market1501.yaml' if arch=='pcb' else 'configs/baseline_market1501_resnet50_triplet.yaml')))
  run_id='official-candidate-20261011'
  cfg['protocol']=dict(name='multi_source_lodo',version=1,architecture=arch,fold=fold)
  cfg['experiment']=dict(name=f'lodo_v1_{arch}_{fold.lower()}_seed42',output_dir=f'exp/lodo/v1/{arch}/{fold}/seed42/{run_id}',notes='PHASE 10 CANDIDATE; DO NOT EXECUTE BEFORE EXPLICIT FREEZE, ALLOCATION AND LAUNCH APPROVAL')
  cfg['system'].update(device='cuda',gpu_id=0,amp=False)
  cfg['repro'].update(seed=42,deterministic=True,benchmark=False,cuda_policy='strict_cuda_deterministic_v1')
  cfg['data'].update(root=str(canonical_data),num_workers=0,pin_memory=True)
  cfg['data'].pop('prefetch_threads',None)
  ds=[descriptor(n) for n in sources]
  for d in ds:d['metadata_sha256']=metadata[d['name']]['source']
  cfg['data']['train']['dataset']=dict(name='multi_source',sources=ds)
  cfg['data']['test']['dataset']=descriptor(target,True);cfg['data']['test']['dataset']['metadata_sha256']=metadata[target]['target']
  cfg['logging']=dict(to_file=False,tensorboard=False) # LODO does not invoke Stage A logging setup.
  cfg['train']=dict(mode='successful_updates',max_updates=94200 if arch=='pcb' else 85200,evaluate_during_training=False,checkpoint_selection='terminal',save=dict(save_best=False,save_last=True,save_terminal=True,resume=''),recovery=dict(version=1,policy='completed_epoch_boundary',every_epochs=1,keep_generations=2))
  cfg['sched'].update(unit='successful_primary_updates',milestones=[31400] if arch=='pcb' else [28400,49700])
  if arch=='pcb':cfg['data']['train']['batch']['drop_last']=True
  else:cfg['loss']['center'].update(enabled=True,feat='raw',reduction='half_mean_squared_distance')
  validate_config(cfg);validate_reid_config(cfg)
  rel=f'configs/lodo/v1/{arch}_{fold.lower()}_seed42.yaml';path=root/rel
  text='# CANDIDATE ONLY. DO NOT EXECUTE BEFORE EXPLICIT APPROVAL.\n'+yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
  # Verify YAML round trip through the same production loader before publication.
  assert yaml.safe_load(text)==cfg
  path.write_text(text)
  actual=load_config(str(path));assert actual==cfg
  obs,mapping=dataset_evidence(cfg)
  map_rel=f'docs/lodo_phase10_evidence/maps/{fold.lower()}.json'
  map_path=root/map_rel
  from reid.data.multi_source import canonical_json
  if map_path.exists():assert map_path.read_bytes()==canonical_json(mapping)
  else:map_path.write_bytes(canonical_json(mapping))
  filename='resnet50-19c8e357.pth' if arch=='pcb' else 'resnet50-11ad3fa6.pth'
  weights=(Path(torch.hub.get_dir())/'checkpoints'/filename).resolve(strict=True)
  evaluation=dict(mode='lodo_held_out',version=1,architecture=arch,fold=fold,target=target,checkpoint=str(root/cfg['experiment']['output_dir']/'checkpoints/ckpt_terminal.pth'),expected=dict(config_digest=configuration_identity(cfg),map_digest=obs['map_digest'],data_digest=obs['data_digest'],num_classes=obs['num_classes']),synthetic=False,output_dir=f'exp/lodo_eval/v1/benchmark/{arch}/{fold}/seed42/{run_id}',data_root=str(canonical_data),num_workers=0,batch_size=32,stage_a_records=[])
  validate_request(evaluation);evaluation_output(evaluation)
  evalrel=f'docs/lodo_phase10_evidence/evaluation_requests/{arch}_{fold.lower()}.yaml'
  (root/evalrel).write_text('# DO NOT EXECUTE BEFORE SEPARATE POST-TRAINING APPROVAL.\n'+yaml.safe_dump(evaluation,sort_keys=False))
  immutable=dict(configuration=dict(path=rel,file_sha256=file_digest(path),scientific_sha256=configuration_identity(cfg),scientific=scientific_configuration(cfg)),code=code,dataset=obs,pretrained=dict(path=str(weights),identifier=filename,sha256=file_digest(weights)),recovery=dict(capability='epoch_boundary_rollback_cuda_v1',checkpoint_policy=cfg['train']['recovery'],checkpoint_schema=2,terminal='ckpt_terminal.pth',partial_terminal_epoch='allowed only at exact completed budget; cannot continue'),mapping_file=map_rel,execution_candidate=dict(cuda_policy='strict_cuda_deterministic_v1',qualified_runtime=runtime,qualification='LODO-1 only, budget5, two-batch fixture epochs, zero workers; NOT all-fold or full-duration proof',workspace=':4096:8',gpu_allocation='separate approval required'),evaluation=dict(request=evalrel,request_sha256=file_digest(root/evalrel),target_free_training=True,checkpoint_selection='terminal',representation=retrieval_contract(cfg),target=target_metadata(cfg,dict(target=target,provenance=obs['provenance']),str(canonical_data))))
  manifest=seal(immutable,dict(approval_status='NOT_APPROVED',allocation_status='NOT_ALLOCATED',prepared_client_date='2026-10-11',proposed_output_dir=str(root/cfg['experiment']['output_dir']),resume='',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  mrel=f'docs/lodo_phase10_evidence/manifests/{arch}_{fold.lower()}.json'
  (root/mrel).write_bytes(canonical_json(manifest))
  rows.append(dict(architecture=arch,fold=fold,sources=list(sources),target=target,images=obs['images'],identities=obs['num_classes'],budget=cfg['train']['max_updates'],config=rel,config_file_sha256=file_digest(path),scientific_sha256=configuration_identity(cfg),map_digest=obs['map_digest'],manifest=mrel,manifest_immutable_sha256=manifest['immutable_sha256'],initial_epoch_batches=obs['initial_epoch_batches'],maximum_epoch_updates=obs['maximum_epoch_updates'],evaluation_request=evalrel))
  print(arch,fold,obs['images'],obs['num_classes'],obs['initial_epoch_batches'],obs['maximum_epoch_updates'],flush=True)
(evidence/'candidate-index.json').write_bytes(canonical_json(dict(schema_version=1,status='CANDIDATE_NOT_FROZEN',runs=rows)))
print('EIGHT CANDIDATES PREPARED; NO CUDA/IMAGES/TRAINING/EVALUATION',flush=True)
