import torch


def ensure_output_dict(output):
    """
    Normalize model outputs into a dict-based interface.

    Supported forms:
    - dict: returned unchanged
    - tuple/list: interpreted as (embedding, logits?)
    - tensor: interpreted as embedding only
    """
    if isinstance(output, dict):
        return output
    if isinstance(output, (tuple, list)):
        emb = output[0]
        logits = output[1] if len(output) > 1 else None
        return {
            "feat_raw": None,
            "feat_bn": None,
            "emb": emb,
            "logits": logits,
        }
    if torch.is_tensor(output):
        return {
            "feat_raw": None,
            "feat_bn": None,
            "emb": output,
            "logits": None,
        }
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def get_embedding(output: dict) -> torch.Tensor:
    emb = output.get("emb")
    if emb is None:
        raise ValueError("Model output dict is missing 'emb'.")
    return emb
