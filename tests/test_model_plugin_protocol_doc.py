from pathlib import Path


REQUIRED_ALLOWED = [
    "reid/models/<model_name>.py",
    "reid/models/build.py",
    "configs/<model>_<dataset>.yaml",
    "reid/losses/",
]


REQUIRED_SHARED_OWNERS = [
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


def test_model_plugin_protocol_documents_generic_boundaries_and_capabilities():
    path = Path("docs/model_plugin_protocol.md")

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for item in REQUIRED_ALLOWED:
        assert item in text
    for item in REQUIRED_SHARED_OWNERS:
        assert item in text
    for term in REQUIRED_TERMS:
        assert term in text

    for term in ("embedding_dim", "nonempty ordered flat", "validate_logits", "feat_dim",
                 "Center loss requires a positive integer", "initialize_pretrained=False",
                 "source `num_classes`", "strictly loads state", "single tensor",
                 "not yet supported"):
        assert term in text
    assert "Every trainable model must expose" not in text
    assert "only core framework file" not in text
