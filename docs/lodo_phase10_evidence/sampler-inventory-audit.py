from pathlib import Path
import sys,json,collections
root=Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo').resolve();sys.path[:0]=[str(root),str(root/'scripts')]
from lodo_official_preparation import metadata_only
from reid.data import build
from reid.data.samplers import PKBatchSampler
from reid.data.multi_source import canonical_json
p=root/'docs/lodo_phase10_evidence';index=json.loads((p/'candidate-index.json').read_text());datasets={};result=[]
with metadata_only():
 for name in ('market1501','duke','cuhk03','msmt17'):
  factory=getattr(build,f'_build_{name}_train_dataset')
  kwargs=dict(root='/mnt/hd-data/addirakoze/Projects/Dataset',split='train' if name=='msmt17' else 'trainval',dataset_format='raw' if name=='msmt17' else 'processed',transform=None)
  if name=='cuhk03':kwargs['dataset_cfg']=dict(image_type='detected',protocol='processed_partition',split_id=None)
  datasets[name]=factory(**kwargs)
 for row in index['runs']:
  if row['architecture']!='resnet50':continue
  mapping=json.loads((p/'maps'/f"{row['fold'].lower()}.json").read_text())
  lookup={(r['source_key'],r['local_label']):r['global_label'] for r in mapping['identities']}
  labels=[lookup[(name,int(label))] for name in row['sources'] for label in datasets[name].labels]
  assert len(labels)==row['images'] and set(labels)==set(range(row['identities']))
  counts=collections.Counter(labels);upper=sum(max(1,n//4) for n in counts.values())//16;lower=len(counts)//16
  sampler=PKBatchSampler(labels,P=16,K=4,drop_last=True,seed=42);assert len(sampler)==row['initial_epoch_batches']
  lengths=[]
  for epoch in range(20):
   batches=list(sampler);lengths.append(len(batches))
   assert lower<=len(batches)<=upper
   for batch in batches:
    selected=collections.Counter(labels[i] for i in batch)
    assert len(batch)==64 and len(selected)==16 and set(selected.values())=={4}
  result.append(dict(fold=row['fold'],epochs_metadata_only=20,initial=lengths[0],observed_min=min(lengths),observed_max=max(lengths),conservative_minimum=lower,conservative_maximum=upper,lengths=lengths,sampling_algorithm='unchanged PKBatchSampler; private audit instance only'))
 source_stats={name:dict(images=len(ds),identities=ds.num_classes,source_local_cameras=sorted(set(map(int,ds.cams))),camera_count=len(set(map(int,ds.cams)))) for name,ds in datasets.items()}
(p/'sampler-and-source-inventory.json').write_bytes(canonical_json(dict(status='PASS',source_statistics=source_stats,pk_epochs=result,pcb_epoch_batches={row['fold']:row['initial_epoch_batches'] for row in index['runs'] if row['architecture']=='pcb'},images_decoded=0,target_loaders_built=0)))
(p/'sampler-inventory-audit.py').write_bytes(Path('/tmp/lodo_phase10_sampler_audit.py').read_bytes())
print(json.dumps(dict(status='PASS',sources=source_stats,pk=[{k:v for k,v in row.items() if k!='lengths'} for row in result]),indent=2))
