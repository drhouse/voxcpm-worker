# voxcpm-worker

Runpod Serverless worker for VoxCPM2 cross-lingual voice cloning, model baked in.
Defaults match the WitchSpring R English voice mod v1 pack (cfg 2.0, 30 steps,
per-line retry threshold). The caller renders text and picks references.

Image: `ghcr.io/drhouse/voxcpm-worker`, built by GitHub Actions on push.

```json
{"input": {"items": [{"id": "Home_0", "text": "...", "ref_b64": "<wav>", "ref_text": "<transcript>"}]}}
```
