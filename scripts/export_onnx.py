#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_model(backbone: str, weights: str):
    try:
        import torchreid
    except Exception as exc:
        raise SystemExit("torchreid is required to export OSNet models.") from exc
    model = torchreid.models.build_model(
        name=backbone,
        num_classes=1,
        loss="softmax",
        pretrained=False,
    )
    torchreid.utils.load_pretrained_weights(model, weights)
    model.eval()
    return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, help="e.g., osnet_x0_25")
    p.add_argument("--weights", required=True, help="path to .pth weights")
    p.add_argument("--output", required=True, help="output .onnx path")
    p.add_argument("--input_h", type=int, default=256)
    p.add_argument("--input_w", type=int, default=128)
    p.add_argument("--opset", type=int, default=11)
    args = p.parse_args()

    model = build_model(args.backbone, args.weights)
    dummy = torch.zeros((1, 3, args.input_h, args.input_w), dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
