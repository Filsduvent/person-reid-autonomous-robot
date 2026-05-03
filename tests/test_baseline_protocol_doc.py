from pathlib import Path


REQUIRED_SECTIONS = [
    "## Scope",
    "## Config Schema",
    "## Dataset Preparation",
    "## Transforms",
    "## Sampler",
    "## Model Settings",
    "## Losses",
    "## Optimizer",
    "## Scheduler",
    "## Evaluation",
    "## Checkpoint Format",
    "## Artifact Schema",
    "## Smoke Matrix",
    "## Future Model Rule",
]


REQUIRED_TERMS = [
    "Market1501",
    "DukeMTMC-ReID",
    "CUHK03",
    "MSMT17",
    "feat_raw",
    "feat_bn",
    "emb",
    "logits",
    "mAP",
    "mINP",
    "Rank1",
    "Rank5",
    "Rank10",
    "ckpt_last.pth",
    "ckpt_best.pth",
    "final_test.json",
]


def test_baseline_protocol_v1_document_exists_and_freezes_required_sections():
    path = Path("docs/baseline_protocol_v1.md")

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text
    for term in REQUIRED_TERMS:
        assert term in text
