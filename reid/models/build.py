from reid.models.baseline import ReidBaseline


def build_model(cfg, num_classes: int | None = None):
    mcfg = cfg["model"]
    bcfg = mcfg["backbone"]
    hcfg = mcfg["head"]

    if mcfg["name"] != "reid_baseline":
        raise NotImplementedError(mcfg["name"])

    model = ReidBaseline(
        pretrained=bool(bcfg["pretrained"]),
        last_conv_stride=int(bcfg["last_conv_stride"]),
        embedding_dim=int(hcfg["embedding_dim"]),
        bnneck=bool(hcfg["bnneck"]),
        normalize=bool(hcfg["normalize"]),
        metric_feat=str(hcfg.get("metric_feat", "bn")).lower(),
        classifier_enabled=bool(hcfg.get("classifier", False)),
        num_classes=num_classes,
    )
    return model
