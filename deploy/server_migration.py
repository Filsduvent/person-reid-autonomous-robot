#!/usr/bin/env python3
"""Prepare an offline migration kit; never connect, install, or launch training.

Run with the source Reid interpreter. Requires PyYAML for config candidates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '.aws', '.agents', '.ssh', '__pycache__', '.pytest_cache',
             '.venv', 'venv', 'env', '.mypy_cache', '.ruff_cache'}
SKIP_FILES = {'.codex', '.env', '.netrc', '.git-credentials'}
TEXT_SUFFIXES = {'.py', '.yaml', '.yml', '.md', '.txt', '.json', '.csv', '.sh'}


def command(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def files(root, excluded):
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and not (Path(base)/d).resolve().is_relative_to(excluded))
        for name in sorted(names + [d for d in dirs if (Path(base)/d).is_symlink()]):
            if name in SKIP_FILES or name.startswith('.env.') or name.endswith(('.pyc', '.pyo')):
                continue
            yield Path(base)/name


def inventory(root, output, excluded):
    count = total = 0
    links = []
    with output.open('w') as stream:
        for path in files(root, excluded):
            stat = path.lstat()
            row = dict(path=str(path.relative_to(root)), size=stat.st_size,
                       mtime_ns=stat.st_mtime_ns)
            if path.is_symlink():
                row['symlink'] = os.readlink(path)
                row['resolved'] = str(path.resolve())
                row['external'] = not path.resolve().is_relative_to(root.resolve())
                links.append(row)
            elif path.is_file():
                total += stat.st_size
                if path.suffix in TEXT_SUFFIXES and stat.st_size <= 1024*1024:
                    row['sha256'] = digest(path)
            else:
                continue
            count += 1
            stream.write(json.dumps(row) + '\n')
    return dict(files=count, logical_bytes=total, symlinks=links,
                hash_policy='Text <=1 MiB only; verify all transferred bytes with rsync -nrc.')


def relocate(value, mapping, changes, key=''):
    if isinstance(value, dict):
        return {k: relocate(v, mapping, changes, f'{key}.{k}'.strip('.')) for k,v in value.items()}
    if isinstance(value, list):
        return [relocate(v, mapping, changes, f'{key}[{i}]') for i,v in enumerate(value)]
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            if value == old or value.startswith(old + '/'):
                result = new + value[len(old):]
                changes.append(dict(key=key, before=value, after=result))
                return result
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh local kit directory')
    parser.add_argument('--target-parent', default=str(ROOT.parent), help='Absolute target parent directory')
    parser.add_argument('--target-dataset', default='/home/addirakoze/nobackup/Projects/Dataset')
    parser.add_argument('--target-cache', default='/home/addirakoze/.cache/torch/hub/checkpoints')
    parser.add_argument('--destination', default='addirakoze@oneiros.inf.ufpr.br')
    parser.add_argument('--dataset', type=Path, default=Path('/home/addirakoze/nobackup/Projects/Dataset'))
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/torch/hub/checkpoints')
    args = parser.parse_args()
    for value in (args.target_parent, args.target_dataset, args.target_cache):
        if not Path(value).is_absolute() or any(c in value for c in '\n\r'):
            parser.error('Target paths must be absolute single-line paths')
    for path in (args.dataset, args.cache):
        if not path.is_dir():
            parser.error(f'Missing external dependency directory: {path}')
    out = args.output.resolve()
    if out == ROOT or not (out.is_relative_to(ROOT/'.local') or out.is_relative_to(Path('/tmp'))):
        parser.error('Output must be under project .local/ or /tmp/')
    out.mkdir(parents=True, exist_ok=False)
    excluded = ROOT/'.local/server_migration' if out.is_relative_to(ROOT/'.local/server_migration') else out
    worktrees = []
    for block in command('git', 'worktree', 'list', '--porcelain').split('\n\n'):
        entry = dict(line.split(' ',1) for line in block.splitlines() if ' ' in line)
        if 'worktree' in entry:
            worktrees.append(entry)
    if len({Path(w['worktree']).name for w in worktrees}) != len(worktrees):
        raise ValueError('Worktree basenames collide; choose explicit distinct destinations')
    mapping = {}
    for entry in worktrees:
        source = Path(entry['worktree'])
        same_layout = args.target_parent == str(ROOT.parent)
        target = str(source) if same_layout else f'{args.target_parent}/{source.name}'
        mapping[str(source)] = target
        alias = '/home/addirakoze/nobackup/Projects/' + source.name
        if Path(alias).resolve() == source.resolve():
            mapping[alias] = alias if same_layout else target
    mapping.update({str(args.dataset): args.target_dataset, str(args.dataset.resolve()): args.target_dataset,
                    str(args.cache): args.target_cache, str(args.cache.resolve()): args.target_cache})
    if args.target_dataset == str(args.dataset) and args.target_parent == str(ROOT.parent):
        mapping[str(args.dataset.resolve())] = str(args.dataset.resolve())
    dump(out/'path-map.json', mapping)
    (out/'pip-freeze.txt').write_text(command(sys.executable,'-m','pip','freeze','--all')+'\n')
    (out/'pip-list.json').write_text(command(sys.executable,'-m','pip','list','--format=json')+'\n')
    (out/'pip-check.txt').write_text(subprocess.run([sys.executable,'-m','pip','check'], text=True,
                                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout)
    # Keep exact build tags. Install CUDA torch from its matching index first.
    pins = [line for line in (out/'pip-freeze.txt').read_text().splitlines()
            if not line.lower().startswith(('torch==','torchvision==','torchaudio=='))]
    (out/'requirements-without-torch.txt').write_text('\n'.join(pins)+'\n')
    import torch, torchvision
    runtime = dict(python=platform.python_version(), executable=sys.executable,
                   platform=platform.platform(), torch=torch.__version__, torchvision=torchvision.__version__,
                   cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    dump(out/'runtime.json', runtime)
    # Bundle history without copying linked-worktree .git pointers or changing refs.
    subprocess.run(['git','bundle','create',str(out/'repository.bundle'),'--all','HEAD',
                    *[w['HEAD'] for w in worktrees]], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(['git','bundle','verify',str(out/'repository.bundle')], cwd=ROOT,
                   check=True, capture_output=True)
    configs = []
    inventory_dir = out/'inventories'; inventory_dir.mkdir()
    for i, entry in enumerate(worktrees):
        source = Path(entry['worktree'])
        if not source.is_dir():
            raise ValueError(f'Missing worktree: {source}')
        entry['target'] = mapping[str(source)]
        entry['inventory'] = inventory(source, inventory_dir/f'worktree-{i}.jsonl', excluded)
        entry['status'] = command('git','status','--porcelain','--untracked-files=all',cwd=source)
        # Includes tracked changes as evidence; rsync overlay also carries untracked/ignored files.
        (out/f'worktree-{i}.patch').write_text(command('git','diff','HEAD','--binary',cwd=source)+'\n')
        for config in sorted((source/'configs').rglob('*')):
            if config.suffix not in {'.yaml','.yml'}:
                continue
            original = yaml.safe_load(config.read_text())
            changes = []
            candidate = relocate(original, mapping, changes)
            relative = Path(source.name)/config.relative_to(source)
            destination = out/'config-candidates'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text('# Migration candidate; review config-index.json before use.\n' +
                                   yaml.safe_dump(candidate, sort_keys=False))
            guarded = isinstance(original,dict) and bool(original.get('train',{}).get('preflight'))
            configs.append(dict(source=str(config), candidate=str(relative), sha256=digest(config),
                                changes=changes, status='REQUALIFICATION_REQUIRED' if guarded or '/lodo/' in str(config)
                                else 'REVIEW_REQUIRED'))
    dependencies = []
    for name, source, target in [('dataset',args.dataset,args.target_dataset),
                                  ('torch-cache',args.cache,args.target_cache)]:
        dependencies.append(dict(name=name, source=str(source.resolve()), target=target,
                                 inventory=inventory(source.resolve(),inventory_dir/f'{name}.jsonl',out)))
    dump(out/'config-index.json', configs)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='PREPARED_NOT_TRANSFERRED',
                  runtime=runtime, worktrees=worktrees, dependencies=dependencies,
                  configurations=len(configs), training_launched=False,
                  minimum_payload_bytes=sum(w['inventory']['logical_bytes'] for w in worktrees)+
                                        sum(d['inventory']['logical_bytes'] for d in dependencies))
    dump(out/'manifest.json', report)
    # An explicit source-side transfer script, inert until the operator supplies a destination.
    excludes = sorted(SKIP_DIRS | SKIP_FILES) + ['.env.*','*.pyc','*.pyo',str(excluded.relative_to(ROOT))+'/' if excluded.is_relative_to(ROOT) else '__no_migration_kit__']
    exclude_args = [arg for item in excludes for arg in ('--exclude',item)]
    q = shlex.quote
    lines = ['#!/usr/bin/env bash','set -euo pipefail',
             'DEST=${DEST:-'+q(args.destination)+'}',
             '# Optional: export RSYNC_RSH="ssh -i /home/addirakoze/.ssh/id_ed25519_oneiros -o IdentitiesOnly=yes"',
             ': "${MODE:?Set MODE=preview, copy, or verify}"',
             'case "$MODE" in', '  preview) flags=(-an --itemize-changes);;',
             '  copy) flags=(-a --partial --info=progress2);;',
             '  verify) flags=(-anrc --itemize-changes);;', '  *) exit 2;;','esac',
             '# No deletion, permission changes, process control, or training. Destinations must be fresh.',
             '# Create target directories using target-setup.sh first.']
    for source, target in [(Path(w['worktree']),w['target']) for w in worktrees] + [
                           (Path(d['source']),d['target']) for d in dependencies] + [(out,args.target_parent+'/migration-kit')]:
        lines.append('rsync "${flags[@]}" --protect-args ' + shlex.join(exclude_args) +
                     ' -- ' + q(str(source)+'/') + ' "$DEST":' + q(target+'/'))
    (out/'transfer.sh').write_text('\n'.join(lines)+'\n')
    setup = ['#!/usr/bin/env bash','set -euo pipefail',
             '# Run on the TARGET. Git setup refuses an existing .git; transfer never deletes.',
             'case "${1:-}" in', '  directories)']
    alias_parent = Path('/home/addirakoze/nobackup/Projects')
    if args.target_parent == str(ROOT.parent) and alias_parent.resolve() == ROOT.parent:
        alias = q(str(alias_parent)); canonical = q(str(ROOT.parent))
        setup += ['    mkdir -p -- '+canonical,
                  f'    if test -e {alias} || test -L {alias}; then',
                  f'      test "$(readlink -f {alias})" = {canonical} || {{ echo "Existing Projects path has a different layout" >&2; exit 2; }}',
                  '    else', '      mkdir -p -- '+q(str(alias_parent.parent)),
                  f'      ln -s -- {canonical} {alias}', '    fi']
    setup += [
             '    mkdir -p -- '+shlex.join([w['target'] for w in worktrees]+[d['target'] for d in dependencies]+[args.target_parent+'/migration-kit'])+';;',
             '  git)', '    kit='+q(args.target_parent+'/migration-kit')]
    for w in worktrees:
        target = q(w['target'])
        setup += [f'    test ! -e {target}/.git',
                  f'    git -C {target} init',
                  f'    git -C {target} fetch "$kit/repository.bundle" "+refs/heads/*:refs/remotes/source/*" "+refs/tags/*:refs/tags/*"',
                  f'    git -C {target} fetch "$kit/repository.bundle" {q(w["HEAD"])}']
        if w.get('branch'):
            setup.append(f'    git -C {target} symbolic-ref HEAD {q(w["branch"])}')
            setup.append(f'    git -C {target} update-ref HEAD {q(w["HEAD"])}')
        else:
            setup.append(f'    git -C {target} update-ref --no-deref HEAD {q(w["HEAD"])}')
        setup.append(f'    git -C {target} read-tree HEAD')
    setup += ['    ;;','  *) echo "Usage: bash target-setup.sh directories|git" >&2; exit 2;;','esac']
    (out/'target-setup.sh').write_text('\n'.join(setup)+'\n')
    bootstrap = ['#!/usr/bin/env bash','set -euo pipefail',
                 'kit=$(cd -- "$(dirname -- "$0")" && pwd)',
                 'env_dir=/home/addirakoze/environments/Reid',
                 '# Run on Oneiros, after verifying GPU/driver and available disk space.',
                 'test ! -e "$env_dir" || { echo "Environment already exists; inspect rather than overwrite" >&2; exit 2; }',
                 f'test "$(python3.11 -c \'import platform; print(platform.python_version())\')" = {q(runtime["python"])} || {{ echo "Exact source Python {runtime["python"]} required for the frozen MGN environment" >&2; exit 2; }}',
                 'python3.11 -m venv "$env_dir"',
                 '"$env_dir/bin/python" -m pip install '+shlex.join([
                     '--no-deps', 'torch=='+runtime['torch'], 'torchvision=='+runtime['torchvision'],
                     '--index-url','https://download.pytorch.org/whl/cu'+str(runtime['cuda_build']).replace('.','')]),
                 '"$env_dir/bin/python" -m pip install -r "$kit/requirements-without-torch.txt"',
                 '"$env_dir/bin/python" -m pip check',
                 '"$env_dir/bin/python" -m pip freeze --all > "$kit/target-pip-freeze.txt"',
                 '"$env_dir/bin/python" -c \'import sys,torch,torchvision; print(sys.version); print(torch.__version__,torchvision.__version__,torch.version.cuda)\'']
    (out/'target-environment.sh').write_text('\n'.join(bootstrap)+'\n')
    # Hash kit content, excluding the checksum file itself.
    with (out/'SHA256SUMS').open('w') as stream:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name!='SHA256SUMS':
                stream.write(f'{digest(path)}  {path.relative_to(out)}\n')
    print(json.dumps(dict(output=str(out),configurations=len(configs),
                          payload_GiB=round(report['minimum_payload_bytes']/1024**3,2),
                          status=report['status']),indent=2))


if __name__ == '__main__':
    main()
