import torch


def validate_logits(logits, *, batch_size=None):
    """Validate without copying, detaching, reordering or aggregating heads.

    Accept None, one floating [B,C] tensor, or a nonempty flat tuple/list of
    tensors with matching shape, dtype and device. Head count is unrestricted.
    batch_size can be the embedding/label batch size supplied by a caller.
    """
    if logits is None:
        return logits
    if torch.is_tensor(logits):
        heads = (logits,)
    elif isinstance(logits, (tuple, list)) and logits:
        heads = logits
    else:
        raise ValueError("logits must be None, a tensor, or a nonempty flat tensor tuple/list.")
    first = None
    for head in heads:
        if not torch.is_tensor(head) or head.ndim != 2 or not head.is_floating_point():
            raise ValueError("Each logits head must be a floating 2-D [B,C] tensor.")
        if head.shape[0] <= 0 or head.shape[1] <= 0:
            raise ValueError("Logits batch and class dimensions must be positive.")
        if batch_size is not None and head.shape[0] != batch_size:
            raise ValueError("Logits batch size does not match the expected batch size.")
        if first is not None:
            if head.shape != first.shape:
                raise ValueError("Logits heads must have the same batch and class dimensions.")
            if head.dtype != first.dtype or head.device != first.device:
                raise ValueError("Logits heads must have matching floating dtype and device.")
        first = head
    return logits


def ensure_output_dict(output):
    """
    Normalize model outputs into a dict-based interface.

    Supported forms:
    - dict: returned unchanged after logits validation
    - tuple/list: interpreted as (embedding, logits?)
    - tensor: interpreted as embedding only
    """
    if isinstance(output, dict):
        emb = output.get("emb")
        batch_size = emb.shape[0] if torch.is_tensor(emb) and emb.ndim > 0 else None
        validate_logits(output.get("logits"), batch_size=batch_size)
        return output
    if isinstance(output, (tuple, list)):
        emb = output[0]
        logits = output[1] if len(output) > 1 else None
        validate_logits(logits, batch_size=emb.shape[0] if torch.is_tensor(emb) and emb.ndim > 0 else None)
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
