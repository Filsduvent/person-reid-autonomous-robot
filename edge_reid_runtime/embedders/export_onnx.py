"""Export runtime ReID embedding paths to ONNX with reproducibility metadata."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from edge_reid_runtime.embedders.embedder_factory import EmbedderConfig, create_embedder

OPSET = 17  # Supported by PyTorch 2.7 and ONNX Runtime 1.29 in the qualification environment.


class EmbeddingExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model(images)
        return output["emb"] if isinstance(output, dict) else output


def export_embedder(model_id: str, weights: str, output: str, device: str = "cpu") -> dict:
    embedder = create_embedder(EmbedderConfig(backbone=model_id, weights=weights, device=device))
    model = EmbeddingExportWrapper(embedder.model).eval().cpu()
    h, w = embedder.cfg.input_size
    sample = torch.zeros((1, 3, h, w), dtype=torch.float32)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, sample, output_path, input_names=["images"], output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}}, opset_version=OPSET,
    )
    metadata = {
        "model": model_id, "source_weights": str(Path(weights).resolve()), "opset": OPSET,
        "input_name": "images", "output_name": "embeddings", "input_shape": [None, 3, h, w],
        "embedding_dim": embedder.embedding_dim, "dynamic_batch": True,
        "normalization": {"mean": list(embedder.cfg.mean), "std": list(embedder.cfg.std), "bgr_to_rgb": True},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(export_embedder(args.model, args.weights, args.output), indent=2))


if __name__ == "__main__":
    main()
