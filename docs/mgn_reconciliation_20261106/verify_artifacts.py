import copy,hashlib,json,math,os,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import torch,yaml
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.mgn_gpu_smoke import file_hash,source_provenance
from scripts.mgn_preflight import environment,evaluation_identity,normalized_training_cfg
from reid.utils.config import load_config
from reid.data.build import build_train_loader,build_test_loader

def portable_cfg(cfg):
 result=normalized_training_cfg(cfg)
 output=result['experiment']['output_dir']
 old='/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot/'
 if output.startswith(old): result['experiment']['output_dir']=str(ROOT/output[len(old):])
 return result

def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def ident(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)}
def check(value,message):
 if not value:raise AssertionError(message)
def verify_mapping(mapping):
 return {p: {'expected':d,'actual':file_hash(p),'matches':file_hash(p)==d} for p,d in mapping.items()}
metrics=('mAP','mINP','Rank1','Rank5','Rank10')
manifest=json.load(open('docs/mgn_experiment_manifest.json'));registry=yaml.safe_load(open('docs/mgn_training_results.yaml'))
report={'created_utc':datetime.now(timezone.utc).isoformat(),'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'starting_status':git('status','--short'),'hostname':subprocess.check_output(['hostname'],text=True).strip(),'commits':{},'sources':{},'datasets':{},'checks':{},'limitations':[]}
protected={p:file_hash(p) for p in git('ls-files','--cached','--others','--exclude-standard').splitlines() if Path(p).is_file()}
Path('/tmp/mgn-protected-before.json').write_text(json.dumps(protected))
for rev in ('b72c4a3','81d356b','f4aea84'):
 report['commits'][rev]={'full_sha':git('rev-parse',rev),'ancestor_of_head':subprocess.run(['git','merge-base','--is-ancestor',rev,'HEAD']).returncode==0}
report['environment']=environment(); report['environment_matches_manifest']=report['environment']==manifest['environment']
report['frozen_files']={section:verify_mapping(manifest[section]) for section in ('runtime_source_sha256','qualification_receipt_sha256')}
for section,items in report['frozen_files'].items():check(all(v['matches'] for v in items.values()),section)
report['pretrained']=ident('/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth');check(report['pretrained']['sha256']==manifest['weights_sha256'],'pretrained hash')
report['historical_manifest']=ident('docs/mgn_freezes/20261105_before_schedule_trial/mgn_experiment_manifest.json');check(report['historical_manifest']['sha256']==registry['frozen_manifest']['sha256'],'historical freeze')
for source,entry in manifest['sources'].items():
 print('Checking dataset',source,flush=True)
 cfg=load_config(entry['config_path'])
 check(file_hash(entry['config_path'])==entry['config_sha256'],'source config')
 for section in ('training','evaluation'):
  for p,h in entry[section]['metadata_sha256'].items():check(file_hash(p)==h,f'metadata {p}')
 train,classes=build_train_loader(cfg);test=build_test_loader(cfg)
 tr=source_provenance(train.dataset,cfg);ev=evaluation_identity(test.dataset,cfg)
 check(tr==entry['training'],f'{source} host training identity');check(ev==entry['evaluation'],f'{source} host evaluation identity')
 old=registry['sources'][source]
 report['datasets'][source]={'classes':classes,'training':tr,'evaluation':ev,'matches_active_host_manifest':True,'portable_matches_original_registry':all(tr[k]==old['training_data_provenance'][k] for k in ('images','num_classes','records_sha256','identity_map_sha256')) and all(ev[k]==old['evaluation_data_provenance'][k] for k in ('images','query','gallery','records_sha256'))}
 check(report['datasets'][source]['portable_matches_original_registry'],'portable identity')
 del train,test
for source,old in registry['sources'].items():
 print('Checking original run',source,flush=True)
 run=Path(old['run_directory']);cfg=load_config(f'configs/mgn/{source}.yaml');resolved=load_config(str(run/'config.resolved.yaml'))
 check(portable_cfg(cfg)==portable_cfg(resolved),f'{source} resolved cfg')
 log=(run/'train.log').read_text();epochs=[int(x) for x in re.findall(r'\[Epoch (\d+)\] avg_loss=',log)]
 check(set(epochs)==set(range(1,121)),f'{source} all epochs');periodic=[json.load(open(run/f'metrics/val_epoch_{i:03}.json')) for i in range(10,121,10)]
 check(len(re.findall(r'\[Eval\] epoch=',log))==12,'12 evaluation log markers')
 best=max(periodic,key=lambda x:x['mAP']);final=json.load(open(run/'metrics/final_test.json'));last=json.load(open(run/'metrics/final_epoch_test.json'))
 check(best['epoch']==final['epoch'] and last['epoch']==120,'selected final identity')
 artifacts=[]
 for p,item in old['artifact_sha256'].items():
  actual=ident(p);check(actual['sha256']==item['sha256'],f'registry artifact {p}');artifacts.append(actual)
 checkpoints={}
 for role,item in old['checkpoints'].items():
  p=Path(item['path']);actual=ident(p);check(actual['sha256']==item['sha256'],f'checkpoint hash {p}')
  c=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
  check(c['epoch']==item['epoch'] and c['scheduler']['completed_epochs']==c['epoch'],'checkpoint epoch')
  check(tuple(c['scheduler']['milestones'])==(60,90),'original schedule');check(portable_cfg(c['cfg'])==portable_cfg(cfg),'checkpoint cfg')
  check(c['selection_state']['best_epoch']==best['epoch'],'selection history');check(c['reconstruction']['num_classes']==old['source_identities'],'source C')
  check(bool(c['optimizer']['state']),'optimizer state')
  finite=all(bool(torch.isfinite(t).all()) for t in c['model'].values() if t.is_floating_point())
  check(finite,'finite model state')
  score=final if role=='ckpt_best.pth' else last
  check(all(abs(c['scores'][k]-score[k])<=1e-5 for k in metrics),'checkpoint metric agreement')
  checkpoints[role]={**actual,'epoch':c['epoch'],'scheduler_completed_epochs':c['scheduler']['completed_epochs'],'milestones':list(c['scheduler']['milestones']),'reconstruction':c['reconstruction'],'selection_state':c['selection_state'],'finite_model_state':finite,'optimizer_state_present':True}
  del c
 for score in periodic+[final,last]:
  check(all(math.isfinite(score[k]) and 0<=score[k]<=1 for k in metrics),'metric bounds');check(score['Rank1']<=score['Rank5']<=score['Rank10'],'ranks')
 check('[Final Eval] checkpoint=ckpt_best.pth epoch='+str(best['epoch']) in log,'final log')
 report['sources'][source]={'status':'COMPLETE_REVERIFIED','completed_epochs':len(set(epochs)),'periodic_evaluations':len(periodic),'checkpoints':checkpoints,'final_best':final,'final_last':last,'registry_artifacts_verified':artifacts,'log':ident(run/'train.log'),'resolved_config':ident(run/'config.resolved.yaml'),'launch_command':(run/'command.txt').read_text().strip(),'launch_commit':(run/'git_commit.txt').read_text().strip(),'resume':old['resume'],'update_count':'NOT_VERIFIED; ordinary checkpoints lack authoritative successful-update counter'}
print('Checking schedule trial',flush=True)
run=Path('exp/mgn/market1501_paper_schedule120');cfg=load_config('configs/mgn_experiments/market1501_paper_schedule120.yaml');original=load_config('configs/mgn/market1501.yaml')
def diffs(a,b,p=''):
 if isinstance(a,dict) and isinstance(b,dict):
  return sum((diffs(a.get(k),b.get(k),p+'.'+k if p else k) for k in sorted(a.keys()|b.keys())),[])
 return [] if a==b else [{'key':p,'original':a,'trial':b}]
delta=diffs(original,cfg);check({d['key'] for d in delta}=={'experiment.name','experiment.output_dir','experiment.notes','train.preflight.name','sched.milestones'},'trial config delta')
check(normalized_training_cfg(load_config(str(run/'config.resolved.yaml')))==normalized_training_cfg(cfg),'trial resolved cfg')
log=(run/'train.log').read_text();epochs=[int(x) for x in re.findall(r'\[Epoch (\d+)\] avg_loss=',log)];check(epochs==list(range(1,121)),'trial epochs')
periodic=[json.load(open(run/f'metrics/val_epoch_{i:03}.json')) for i in range(10,121,10)];best=max(periodic,key=lambda x:x['mAP']);final=json.load(open(run/'metrics/final_test.json'));last=json.load(open(run/'metrics/final_epoch_test.json'))
trial={'status':'COMPLETE_ARTIFACT_VERIFIED_SEPARATE_UNSELECTED_TRIAL','configuration_delta':delta,'completed_epochs':120,'periodic_evaluations':periodic,'final_best':final,'final_last':last,'checkpoints':{},'artifacts':[ident(p) for p in sorted(run.rglob('*')) if p.is_file() and (p.suffix in ('.json','.yaml','.txt') or p.name=='train.log')],'launch_command':(run/'command.txt').read_text().strip(),'launch_commit':(run/'git_commit.txt').read_text().strip()}
for role in ('ckpt_best.pth','ckpt_last.pth'):
 p=run/'checkpoints'/role;c=torch.load(p,map_location='cpu',weights_only=True,mmap=True);score=final if role=='ckpt_best.pth' else last
 check(c['epoch']==score['epoch'] and c['scheduler']['completed_epochs']==c['epoch'],'trial checkpoint epoch');check(tuple(c['scheduler']['milestones'])==(40,60),'trial schedule');check(portable_cfg(c['cfg'])==portable_cfg(cfg),'trial checkpoint config');check(c['selection_state']['best_epoch']==best['epoch'],'trial selection');check(all(abs(c['scores'][k]-score[k])<=1e-5 for k in metrics),'trial scores')
 check(all(bool(torch.isfinite(t).all()) for t in c['model'].values() if t.is_floating_point()),'trial finite')
 trial['checkpoints'][role]={**ident(p),'epoch':c['epoch'],'scheduler_completed_epochs':c['scheduler']['completed_epochs'],'milestones':list(c['scheduler']['milestones']),'selection_state':c['selection_state'],'reconstruction':c['reconstruction'],'finite_model_state':True};del c
check('[Final Eval] checkpoint=ckpt_best.pth epoch=90' in log,'trial final log');check(len(re.findall(r'\[Eval\] epoch=',log))==12,'trial eval markers')
lr_windows={}
for epoch,rate in re.findall(r'Epoch \[(\d+)\] Iter .*? lr=([0-9.e+-]+)',log):
 epoch=int(epoch);expected=.01 if epoch<=40 else .001 if epoch<=60 else .0001;check(abs(float(rate)-expected)<1e-12,'trial LR trace');lr_windows[str(epoch)]=float(rate)
check(len(lr_windows)==120,'trial LR all epochs');trial['observed_lr_by_epoch']=lr_windows
trial['delta_percentage_points']={role:{k:100*(trial['final_'+role][k]-report['sources']['market1501']['final_'+role][k]) for k in metrics} for role in ('best','last')}
report['trial']=trial
report['cuda_initialized']=torch.cuda.is_initialized();check(not report['cuda_initialized'],'no CUDA initialization')
report['status']='CPU_ARTIFACT_AND_DATA_RECONCILIATION_PASS';report['training_launched']=False;report['evaluation_launched']=False
Path('/tmp/mgn-reconciliation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',json.dumps({s:r['final_best']['epoch'] for s,r in report['sources'].items()}),flush=True)
