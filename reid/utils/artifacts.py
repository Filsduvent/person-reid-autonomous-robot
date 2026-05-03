from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
from typing import Sequence

import torch

from reid.utils.io import ensure_dir


def get_git_commit(cwd: str | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_environment_text() -> str:
    lines = [
        f"python: {sys.version.replace(os.linesep, ' ')}",
        f"executable: {sys.executable}",
        f"platform: {platform.platform()}",
        f"cwd: {os.getcwd()}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"cuda_version: {torch.version.cuda}",
        f"cudnn_version: {torch.backends.cudnn.version()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            lines.append(f"cuda_device_{idx}: {torch.cuda.get_device_name(idx)}")
    return "\n".join(lines) + "\n"


def save_run_artifacts(exp_dir: str, argv: Sequence[str] | None = None, cwd: str | None = None) -> dict:
    exp_dir = os.path.abspath(exp_dir)
    ensure_dir(exp_dir)

    if argv is None:
        argv = sys.argv
    command_path = os.path.join(exp_dir, "command.txt")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write(shlex.join([str(arg) for arg in argv]))
        f.write("\n")

    environment_path = os.path.join(exp_dir, "environment.txt")
    with open(environment_path, "w", encoding="utf-8") as f:
        f.write(collect_environment_text())

    paths = {
        "command": command_path,
        "environment": environment_path,
    }

    commit = get_git_commit(cwd=cwd)
    if commit is not None:
        git_commit_path = os.path.join(exp_dir, "git_commit.txt")
        with open(git_commit_path, "w", encoding="utf-8") as f:
            f.write(commit)
            f.write("\n")
        paths["git_commit"] = git_commit_path

    return paths
