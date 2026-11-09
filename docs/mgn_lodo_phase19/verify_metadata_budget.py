import hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
import yaml
from scripts.mgn_gpu_smoke import file_hash
from reid.utils.config import load_config
from reid.data.build import build_train_loader
from reid.data.samplers import PKBatchSampler
from reid.data.multi_source import canonical_json
from reid.utils.lodo_protocol import FOLDS
paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],text=True).splitlines()
protected={p:file_hash(p) for p in paths if Path(p).is_file()}
Path('/tmp/mgn-phase19-protected.json').write_text(json.dumps(protected))
manifest=json.load(open('docs/mgn_experiment_manifest.json'));datasets={}
for source,entry in manifest['sources'].items():
 for p,h in entry['training']['metadata_sha256'].items():assert file_hash(p)==h
 loader,c=build_train_loader(load_config(entry['config_path']));datasets[source]=loader.dataset
report={'status':'PROPOSAL_CPU_METADATA_AND_INDEX_CHECKS_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'folds':{},'budget_proposal':85200,'milestones_proposal':[42600,63900],'gpu_executed':False,'training_executed':False}
for fold,definition in FOLDS.items():
 print(fold,definition,flush=True)
 sources=definition['sources'] if isinstance(definition,dict) else definition[0]
 target=definition['target'] if isinstance(definition,dict) else definition[1]
 rows=[];labels=[];offsets={};stats={}
 for source in sources:
  d=datasets[source];pid_local={int(p):int(l) for p,l in zip(d.pids,d.labels)}
  offset=len(rows);pid_global={p:offset+i for i,p in enumerate(sorted(pid_local))}
  rows.extend(dict(source_key=source,source_exposed_pid=p,local_label=pid_local[p],global_label=pid_global[p]) for p in sorted(pid_local))
  labels.extend(pid_global[int(p)] for p in d.pids)
  offsets[source]={'first_global_label':offset,'last_global_label':len(rows)-1};stats[source]={'images':len(d),'identities':d.num_classes}
 mapping=dict(schema_version=1,pid_semantics='source_exposed_integer',ordered_sources=list(sources),num_classes=len(rows),source_class_counts={n:s['identities'] for n,s in stats.items()},identities=rows)
 old=Path(f'docs/lodo_phase10_evidence/maps/{fold.lower()}.json')
 assert mapping==json.load(open(old)),f'{fold} historical map differs'
 digest=hashlib.sha256(canonical_json(mapping)).hexdigest();assert digest==file_hash(old),'historical canonical map bytes'
 sampler=PKBatchSampler(labels,P=16,K=4,seed=42,drop_last=True)
 epoch_counts=[];updates=0;budget=85200;prefix_exposure={s:0 for s in sources};first120=0;terminal_epoch=None;terminal_batch=None;stream=hashlib.sha256()
 while len(epoch_counts)<120 or updates<budget:
  count=0;epoch=len(epoch_counts)+1
  for batch in sampler:
   count+=1;updates+=1
   if updates<=budget:
    stream.update(json.dumps([epoch,batch],separators=(',',':')).encode()+b'\n')
    for index in batch:
     label=labels[index]
     source=next(s for s in sources if offsets[s]['first_global_label']<=label<=offsets[s]['last_global_label'])
     prefix_exposure[source]+=1
    if updates==budget:terminal_epoch=epoch;terminal_batch=count
  epoch_counts.append(count)
  if epoch==120:first120=updates
 assert sum(prefix_exposure.values())==budget*64
 report['folds'][fold]={'ordered_sources':list(sources),'target':target,'images':len(labels),'classes':len(rows),'source_statistics':stats,'label_ranges':offsets,'historical_identity_map_path':str(old),'historical_map_sha256':digest,'mapping_matches_historical_bytes':True,'reference_first_120_epoch_batches':epoch_counts[:120],'reference_120_epoch_updates':first120,'proposed_terminal_epoch':terminal_epoch,'proposed_terminal_batch':terminal_batch,'proposed_budget_image_draws':budget*64,'index_only_source_exposure':prefix_exposure,'index_prefix_sha256':stream.hexdigest(),'limitation':'Metadata/index-only fresh seed42 simulation, no images decoded or updates executed; exact replay only after qualified recovery. Reference120 epochs is not the new stopping rule.'}
report['mean_reference_120_epoch_updates']=sum(f['reference_120_epoch_updates'] for f in report['folds'].values())/4
report['schedule_boundary_examples']=[{'update':u,'completed_before':u-1,'learning_rate':.01 if u<=42600 else .001 if u<=63900 else .0001} for u in (1,42600,42601,63900,63901,85200)]
report['budget_images_per_fold']=85200*64;report['budget_updates_four_folds']=4*85200
Path('docs/mgn_lodo_phase19/metadata-budget-evidence.yaml').write_text(yaml.safe_dump(report,sort_keys=False))
print('PASS mean120 reference',report['mean_reference_120_epoch_updates'],flush=True)
