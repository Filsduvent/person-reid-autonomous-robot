from pathlib import Path
import hashlib,json,subprocess,datetime
base=json.loads(Path('/tmp/lodo-phase10-before.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
out={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':{},'scope':'All recorded artifact stats and selected full hashes, not every image or every large original checkpoint'}
for root_s,old in base['roots'].items():
 root=Path(root_s)
 def git(*args):return subprocess.check_output(['git',*args],cwd=root).decode()
 files=[p for p in (root/'exp').rglob('*') if p.is_file() and not p.is_symlink()]
 stats={str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino] for p in files}
 assert stats==old['artifacts'],root_s+' experiment artifact stat/membership changed'
 for i,(rel,digest) in enumerate(old['artifact_sha256'].items()):
  assert sha(root/rel)==digest,rel
  if i%500==0: print(root.name,'hash',i,flush=True)
 r=dict(artifact_stats_matched=len(stats),artifact_full_sha256_matched=len(old['artifact_sha256']),head=git('rev-parse','HEAD').strip(),branch=git('branch','--show-current').strip(),staged=git('diff','--cached','--name-only'),exp_membership_unchanged=True)
 assert r['head']==old['head'] and r['branch']==old['branch'] and r['staged']==old['staged']==''
 if root.name=='person-reid-autonomous-robot':
  assert git('status','--short')==old['status']
  for rel,digest in old['sources'].items():assert sha(root/rel)==digest,rel
  r['original_sources_matched']=len(old['sources']);r['original_git_status_unchanged']=True
 out['roots'][root_s]=r
 print(root.name,'PASS',flush=True)
results=json.loads(Path('/tmp/lodo-phase10-stage-a-results-before.json').read_text())
print('RESULTS_SCHEMA',list(results)[:5],flush=True)
# Result baseline is a path-to-digest mapping.
for rel,digest in results.items():
 if isinstance(digest,str):assert sha(Path('/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot')/rel)==digest,rel
out['original_result_files_matched']=len(results)
out['status']='PASS'
p=Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo/docs/lodo_phase10_evidence/preservation-artifacts.json')
p.write_text(json.dumps(out,indent=2)+'\n')
print('PRESERVATION PASS',flush=True)
