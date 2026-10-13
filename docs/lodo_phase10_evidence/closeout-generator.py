from pathlib import Path
import json,hashlib,subprocess,datetime,shutil,re,ast
root=Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo').resolve();e=root/'docs/lodo_phase10_evidence';before=json.loads((e/'pre-edit-state.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True)
# Correct a report-only arithmetic range, retaining original report generator.
r=root/'docs/lodo_phase10_readiness.md';t=r.read_text().replace('roughly1.3–3.5%','roughly1.1–5.3%');r.write_text(t)
# Make resume preflight scope explicit in the runbook as well.
p=root/'docs/lodo_official_launch_runbook.md';t=p.read_text();t=t.replace('The final line is a reviewable command, **not an instruction to execute now**. Resume exports', 'The candidate verifier reads integrity/config/same-run namespace through the existing output validator and verifies live provenance; it does not itself compare all saved tensors against fresh model/optimizer objects. Production performs that full CPU-only compatibility validation before device admission. The final line is a reviewable command, **not an instruction to execute now**. Resume exports');p.write_text(t)
checks=[]
for i,block in enumerate(re.findall(r'```bash\n(.*?)```',t,re.S),1):
 result=subprocess.run(['/bin/bash','-n'],input=block,text=True,capture_output=True);assert result.returncode==0,(i,result.stderr)
 checks.append({'block':i,'shell_syntax':'PASS','executed':False})
 for j,py in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)",block,re.S),1):ast.parse(py);checks[-1].setdefault('embedded_python',[]).append({'block':j,'syntax':'PASS','executed':False})
(e/'runbook-syntax.json').write_text(json.dumps({'status':'PASS','checks':checks,'official_commands_executed':False},indent=2)+'\n')
for src,name in [('/tmp/lodo_phase10_report.py','readiness-report-generator.py'),('/tmp/lodo_phase10_finalize_docs.py','evidence-archive-script.py'),('/tmp/lodo_phase10_snapshot.py','baseline-snapshot-script.py')]:shutil.copyfile(src,e/name)
old=before['roots'][str(root)];changed=[]
for rel,h in old['sources'].items():
 assert (root/rel).is_file(),rel
 if sha(root/rel)!=h:changed.append(rel)
assert changed==['docs/lodo_decision_register.md'],changed
oldregister=old['sources']['docs/lodo_decision_register.md'];current=(root/'docs/lodo_decision_register.md').read_text();original=current.split('\n\n## Phase 10 preparation')[0]
assert hashlib.sha256(original.encode()).hexdigest()==oldregister,'Decision register was not append-only'
# Verify source snapshots and all recorded scientific/file identities one final time.
import sys;sys.path.insert(0,str(root/'scripts'))
from lodo_official_preparation import verify_code,validate_envelope
index=json.loads((e/'candidate-index.json').read_text())
for row in index['runs']:
 m=json.loads((root/row['manifest']).read_text());validate_envelope(m);verify_code(m['immutable']['code'])
 assert row['manifest_immutable_sha256']==m['immutable_sha256']
 assert sha(root/row['config'])==row['config_file_sha256']
 assert not (root/Path(m['operational']['proposed_output_dir'])).exists()
 verify=git(root,'diff','--check');assert not verify
manifest_passes=[json.loads(line) for line in (e/'eight-preflights.log').read_text().splitlines() if line.startswith('{"cuda_initialized"')]
assert len(manifest_passes)==8 and all(x['status']=='PASS_CANDIDATE_ONLY' and x['cuda_initialized'] is False for x in manifest_passes)
(e/'eight-preflight-results.json').write_text(json.dumps({'status':'PASS','runs':manifest_passes,'source_snapshot_sha256':m['immutable']['code']['source_snapshot_sha256']},indent=2)+'\n')
# CPU/source/status closeout; preservation artifact receipt must exist first.
art=json.loads((e/'preservation-artifacts.json').read_text());assert art['status']=='PASS'
shutil.copyfile('/tmp/lodo-phase10-preservation.log',e/'preservation-artifacts.log')
close={'status':'PASS','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'artifact_receipt':'preservation-artifacts.json','original_stage_a_unchanged':True,'feature_preexisting_files':len(old['sources']),'feature_preexisting_intentional_changes':changed,'decision_register_append_only':True,'production_implementation_snapshot_unchanged_after_full_tests':True,'official_output_directories_created':0,'training_evaluation_gpu_jobs_launched':0,'commits_staged_or_pushed':0,'diff_check':'PASS','roots':{}}
for root_s,state in before['roots'].items():
 d=Path(root_s);status=git(d,'status','--short');head=git(d,'rev-parse','HEAD').strip();branch=git(d,'branch','--show-current').strip();staged=git(d,'diff','--cached','--name-only');assert head==state['head'] and branch==state['branch'] and staged==''
 if d!=root:assert status==state['status']
 close['roots'][root_s]=dict(branch=branch,head=head,status=status,staged=staged)
(e/'preservation-closeout.json').write_text(json.dumps(close,indent=2)+'\n')
# Complete delta and freeze proposal list excludes experiment/generated tensor artifacts.
names=[n for n in git(root,'ls-files','-co','--exclude-standard','-z').split('\0') if n]
new=sorted((set(names)|{str(p.relative_to(root)) for p in e.rglob('*') if p.is_file()})-set(old['sources']));nowmod=sorted(n for n in names if n in old['sources'] and sha(root/n)!=old['sources'][n]);trackedmod=[n for n in git(root,'diff','--name-only').splitlines()]
freeze=sorted(set(trackedmod)|{n for n in names if n not in git(root,'ls-files','-z').split('\0')}|{str(p.relative_to(root)) for p in e.rglob('*.json')})
assert all(not n.startswith(('exp/','results/')) and not n.endswith(('.pth','.pt','.jpg','.png','.pkl')) for n in freeze)
# Include the three final evidence inventory files themselves in the explicit proposal.
freeze=sorted(set(freeze)|{'docs/lodo_phase10_evidence/repository-delta.json','docs/lodo_phase10_evidence/proposed-freeze-files.txt','docs/lodo_phase10_evidence/final-evidence-sha256.json','docs/lodo_phase10_evidence/closeout-generator.py'})
(e/'proposed-freeze-files.txt').write_text('\n'.join(freeze)+'\n')
shutil.copyfile('/tmp/lodo_phase10_closeout.py',e/'closeout-generator.py')
delta=dict(schema_version=1,original_git_state=close['roots'][str(Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot').resolve())],feature_git_state=close['roots'][str(root)],preexisting_tracked_modified=trackedmod,phase10_modified_existing=nowmod,phase10_added_files=new,full_current_freeze_proposal=freeze,phase10_file_hashes={n:dict(sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in new+nowmod if (root/n).is_file() and n not in ('docs/lodo_phase10_evidence/repository-delta.json','docs/lodo_phase10_evidence/final-evidence-sha256.json')},inventory_self_exclusions=['repository-delta.json','final-evidence-sha256.json'],staged_files=[],code_snapshot_sha256=m['immutable']['code']['source_snapshot_sha256'])
# Refresh current status after adding inventory/list/generator; then record final manifests separately.
(e/'repository-delta.json').write_text(json.dumps(delta,indent=2)+'\n')
files=sorted(p for p in e.rglob('*') if p.is_file() and p.name!='final-evidence-sha256.json')
proof=dict(schema_version=1,status='READY FOR LAUNCH APPROVAL REVIEW',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),self_excluded='docs/lodo_phase10_evidence/final-evidence-sha256.json',sha256={str(p.relative_to(root)):sha(p) for p in files},phase10_deliverables={str(p.relative_to(root)):sha(p) for p in [root/'docs/lodo_phase10_readiness.md',root/'docs/lodo_official_experiment_protocol.md',root/'docs/lodo_official_launch_runbook.md',root/'docs/lodo_decision_register.md',root/'scripts/lodo_official_preparation.py',root/'tests/test_lodo_official_preparation.py']+sorted((root/'configs/lodo/v1').glob('*.yaml'))},official_execution_authorized=False)
(e/'final-evidence-sha256.json').write_text(json.dumps(proof,indent=2)+'\n')
print('CLOSEOUT PASS',len(new),'new Phase10 files;',len(freeze),'proposed freeze paths;',len(files),'evidence files')
