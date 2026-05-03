from pathlib import Path


REQUIRED_ALLOWED = [
    "reid/models/<model_name>.py",
    "reid/models/build.py",
    "configs/<model>_<dataset>.yaml",
    "reid/losses/",
]


REQUIRED_FORBIDDEN = [
    "reid/data/",
    "reid/engine/evaluator.py",
    "reid/engine/train_loop.py",
    "reid/metrics/",
    "scripts/train.py",
    "scripts/evaluate.py",
]


REQUIRED_TERMS = [
    "PCB",
    "MGN",
    "TransReID",
    "feat_raw",
    "feat_bn",
    "emb",
    "logits",
    "feat_dim",
    "tests/test_model_plugin_contract.py",
]


def test_model_plugin_protocol_documents_allowed_and_forbidden_boundaries():
    path = Path("docs/model_plugin_protocol.md")

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for item in REQUIRED_ALLOWED:
        assert item in text
    for item in REQUIRED_FORBIDDEN:
        assert item in text
    for term in REQUIRED_TERMS:
        assert term in text
