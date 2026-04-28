from reid.models.baseline import ReidBaseline


def build_model(cfg, num_classes: int | None = None):
    mcfg = cfg["model"]
    bcfg = mcfg["backbone"]
    hcfg = mcfg["head"]
    last_conv_stride = int(bcfg["last_conv_stride"])
    bnneck = bool(hcfg["bnneck"])
    metric_feat = str(hcfg.get("metric_feat", "raw")).lower()
    eval_feat = str(hcfg.get("eval_feat", "bn")).lower()
    classifier = bool(hcfg.get("classifier", True))

    if metric_feat not in {"raw", "bn"}:
        raise ValueError(f"Unsupported metric feature '{metric_feat}'. Use 'raw' or 'bn'.")
    if eval_feat not in {"raw", "bn"}:
        raise ValueError(f"Unsupported eval feature '{eval_feat}'. Use 'raw' or 'bn'.")

    if mcfg["name"] != "reid_baseline":
        raise NotImplementedError(mcfg["name"])

    print(f"[Model] backbone=resnet50 last_conv_stride={last_conv_stride}")

    model = ReidBaseline(
        pretrained=bool(bcfg["pretrained"]),
        last_conv_stride=last_conv_stride,
        embedding_dim=int(hcfg["embedding_dim"]),
        bnneck=bnneck,
        normalize=bool(hcfg["normalize"]),
        metric_feat=metric_feat,
        eval_feat=eval_feat,
        classifier_enabled=classifier,
        num_classes=num_classes,
    )
    return model
