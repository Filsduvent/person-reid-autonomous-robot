# person-reid-autonomous-robot

## Phase 5: Embedding Extraction (Summary)

### Preprocessing (ReID)
- **Input crops:** extracted from tracked person boxes (clipped to image bounds)
- **Color space:** BGR -> RGB
- **Resize:**
  - OSNet: 256 x 128 (H x W)
  - MobileNetV3 / ShuffleNetV2 / EfficientNet-Lite: 224 x 224 (H x W)
- **Normalization:** ImageNet mean/std
  - mean = (0.485, 0.456, 0.406)
  - std = (0.229, 0.224, 0.225)
- **Layout:** CHW tensor, float32, values in [0, 1]

### Outputs
- Per-frame logs: `outputs/.../frames.jsonl`
  - Includes `num_crops`, `embed_dim`, and per-stage timings
- Per-embedding logs: `outputs/.../embeddings.jsonl`
  - Each entry includes `frame_id`, `track_id`, `embedding`, `embedding_dim`
