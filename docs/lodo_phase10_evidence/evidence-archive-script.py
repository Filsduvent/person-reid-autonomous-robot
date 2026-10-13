from pathlib import Path
import json, shutil, subprocess, re, ast, xml.etree.ElementTree as ET, hashlib
root=Path('/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo')
e=root/'docs/lodo_phase10_evidence'
logs={'/tmp/lodo-phase10-full.log':'cpu-full.log','/tmp/lodo-phase10-full.xml':'cpu-full.xml','/tmp/lodo-phase10-focused.log':'cpu-focused.log','/tmp/lodo-phase10-focused.xml':'cpu-focused.xml','/tmp/lodo-phase10-generation.log':'generation.log','/tmp/lodo-phase10-eight-preflights.log':'eight-preflights.log','/tmp/lodo-phase10-independent-audit-final.log':'independent-audit-final.log','/tmp/lodo-phase10-sampler-audit.log':'sampler-audit.log','/tmp/lodo-phase10-train-cli-help.txt':'train-cli-help.txt','/tmp/lodo-phase10-eval-cli-help.txt':'eval-cli-help.txt','/tmp/lodo_phase10_preserve.py':'preservation-audit.py'}
for s,d in logs.items():shutil.copyfile(s,e/d)
# Preserve exact generator refinement if present.
for p in Path('/tmp').glob('lodo_phase10*refin*.py'):shutil.copyfile(p,e/p.name)
with (e/'pip-freeze.txt').open('w') as f:subprocess.run(['/home/addirakoze/environments/Reid/bin/python','-m','pip','freeze','--all'],stdout=f,check=True)
checks=[];text=(root/'docs/lodo_official_launch_runbook.md').read_text()
for i,block in enumerate(re.findall(r'```bash\n(.*?)```',text,re.S),1):
 r=subprocess.run(['/bin/bash','-n'],input=block,text=True,capture_output=True)
 assert r.returncode==0,(i,r.stderr)
 checks.append({'block':i,'shell_syntax':'PASS','executed':False})
 for j,py in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)",block,re.S),1):
  ast.parse(py);checks[-1].setdefault('embedded_python',[]).append({'block':j,'syntax':'PASS','executed':False})
(e/'runbook-syntax.json').write_text(json.dumps({'status':'PASS','checks':checks,'official_commands_executed':False},indent=2)+'\n')
def junit(path):
 tree=ET.parse(path);suite=tree.getroot().find('testsuite');skips=[]
 for tc in suite.findall('testcase'):
  s=tc.find('skipped')
  if s is not None:skips.append({'test':tc.attrib['classname']+'.'+tc.attrib['name'],'reason':s.attrib.get('message','')})
 return dict(tests=int(suite.attrib['tests']),passed=int(suite.attrib['tests'])-int(suite.attrib['failures'])-int(suite.attrib['errors'])-int(suite.attrib['skipped']),failed=int(suite.attrib['failures']),errors=int(suite.attrib['errors']),skipped=int(suite.attrib['skipped']),junit_seconds=float(suite.attrib['time']),skips=skips)
r={'environment':{'interpreter':'/home/addirakoze/environments/Reid/bin/python','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'.:scripts:tests','CUDA_VISIBLE_DEVICES':'','CUBLAS_WORKSPACE_CONFIG':':4096:8'},'full':junit(e/'cpu-full.xml'),'focused':junit(e/'cpu-focused.xml'),'full_pytest_seconds':511.10,'focused_pytest_seconds':6.97,'full_command':'timeout 1800 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests -q --junitxml=/tmp/lodo-phase10-full.xml','focused_command':'timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_official_preparation.py -q --junitxml=/tmp/lodo-phase10-focused.xml','workdir':str(root.resolve()),'regressions_observed':0,'audit_fixture_failure':'Preserved first audit literal .01 equality failure; corrected audit expectation to frozen LR arithmetic, no production/config/tolerance edits.'}
(e/'cpu-tests.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
