import os
import random

import numpy as np
import torch

from reid.utils.artifacts import collect_environment_text, save_run_artifacts
from reid.utils.seed import set_seed


def test_set_seed_controls_python_numpy_torch_and_cudnn(monkeypatch):
    cuda_calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: cuda_calls.append(seed))

    set_seed(seed=123, deterministic=True, benchmark=False)
    py_value = random.random()
    np_value = np.random.rand()
    torch_value = torch.rand(1).item()

    set_seed(seed=123, deterministic=True, benchmark=False)

    assert random.random() == py_value
    assert np.random.rand() == np_value
    assert torch.rand(1).item() == torch_value
    assert len(cuda_calls) >= 2
    assert all(seed == 123 for seed in cuda_calls)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_seed(seed=321, deterministic=False, benchmark=True)

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_save_run_artifacts_writes_command_environment_and_git_commit(tmp_path, monkeypatch):
    monkeypatch.setattr("reid.utils.artifacts.get_git_commit", lambda cwd=None: "abc123")

    paths = save_run_artifacts(
        str(tmp_path),
        argv=["python", "scripts/train.py", "--config", "configs/test.yaml"],
        cwd=str(tmp_path),
    )

    assert set(paths) == {"artifacts_dir", "command", "environment", "git_commit"}
    assert (tmp_path / "artifacts" / "command.txt").read_text(encoding="utf-8").strip() == (
        "python scripts/train.py --config configs/test.yaml"
    )
    assert (tmp_path / "command.txt").read_text(encoding="utf-8").strip() == (
        "python scripts/train.py --config configs/test.yaml"
    )
    environment = (tmp_path / "artifacts" / "environment.txt").read_text(encoding="utf-8")
    assert "python:" in environment
    assert "torch:" in environment
    assert "cuda_available:" in environment
    assert (tmp_path / "artifacts" / "git_commit.txt").read_text(encoding="utf-8").strip() == "abc123"
    assert (tmp_path / "git_commit.txt").read_text(encoding="utf-8").strip() == "abc123"


def test_save_run_artifacts_skips_git_commit_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("reid.utils.artifacts.get_git_commit", lambda cwd=None: None)

    paths = save_run_artifacts(str(tmp_path), argv=["train"])

    assert set(paths) == {"artifacts_dir", "command", "environment"}
    assert not (tmp_path / "artifacts" / "git_commit.txt").exists()
    assert not (tmp_path / "git_commit.txt").exists()


def test_collect_environment_text_contains_runtime_facts():
    text = collect_environment_text()

    assert f"executable: {os.sys.executable}" in text
    assert "platform:" in text
    assert "cwd:" in text
