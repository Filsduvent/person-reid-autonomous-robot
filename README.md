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

## Phase 6: Gallery + Identity Assignment (Summary)

### Behavior
- Cosine similarity against gallery prototypes (L2-normalized).
- **Known vs Unknown** decided via `known_threshold` and `unknown_threshold`.
- Enrollment only for stable tracks (age/hits) and similarity below unknown threshold.
- Prototype update via EMA with drift protection gates.

### Key CLI parameters (gallery)
- `--known_threshold`, `--unknown_threshold`
- `--update_threshold`, `--margin_threshold`
- `--stable_age`, `--stable_hits`
- `--min_det_conf`
- `--area_drop_ratio`, `--aspect_ratio_min`, `--aspect_ratio_max`
- `--ema_alpha`
- `--reacquire_cooldown_frames`
- `--max_identities`, `--id_prefix`, `--id_width`

### Outputs
- Per-frame logs: `outputs/.../frames.jsonl`
  - Includes `gallery_size`, `num_known`, `num_unknown`
- Per-identity logs: `outputs/.../identities.jsonl`
  - Each entry includes `track_id`, `identity_id`, `label`, `score`, `margin`,
    `enrolled`, `updated`, `skip_reason`

### Gallery persistence
- Use `--gallery_path /path/to/gallery.json` to save and reload identity prototypes.
- Use `--reset_gallery` to ignore any existing file and start fresh.

### Demo (OSNet)
```
PYTHONPATH=.. python -m edge_reid_runtime.run \
  --source webcam --device cpu \
  --output_dir outputs/reid_gallery_webcam \
  --reid_backbone osnet_x0_25 \
  --weights /path/to/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth \
  --save_video --display
```
