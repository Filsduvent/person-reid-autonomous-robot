from reid.models.baseline import ReidBaseline

def build_model(cfg):
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
    )
    return model
