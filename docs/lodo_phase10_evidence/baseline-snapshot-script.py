from pathlib import Path
import hashlib,json,subprocess,datetime
out=Path('/tmp/lodo-phase10-before.json')
assert not out.exists()
roots=[Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot').resolve(),Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo').resolve()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
r={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':{}}
for root in roots:
 def git(*args):return subprocess.check_output(['git',*args],cwd=root).decode()
 names=[x for x in git('ls-files','-co','--exclude-standard','-z').split('\0') if x]
 sources={x:sha(root/x) for x in names if (root/x).is_file()}
 files=[p for p in (root/'exp').rglob('*') if p.is_file() and not p.is_symlink()]
 stats={str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino] for p in files}
 chosen=[p for p in files if p.stat().st_size<2*1024*1024 or root.name.endswith('-lodo') and p.suffix=='.pth' or 'pcb/msmt17/checkpoints/ckpt_' in str(p)]
 hashes={str(p.relative_to(root)):sha(p) for p in chosen}
 r['roots'][str(root)]=dict(head=git('rev-parse','HEAD').strip(),branch=git('branch','--show-current').strip(),status=git('status','--short'),sources=sources,artifacts=stats,artifact_sha256=hashes,staged=git('diff','--cached','--name-only'))
 print(root.name,len(sources),'sources',len(stats),'artifact stats',len(hashes),'full artifact hashes',flush=True)
with out.open('x') as f:json.dump(r,f,indent=2)
print('BASELINE COMPLETE',flush=True)
