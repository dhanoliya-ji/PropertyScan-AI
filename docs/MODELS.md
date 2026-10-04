# Models, datasets and external tools (disclosure)

| What | Used for | Where | Licence |
|---|---|---|---|
| Depth Anything V2 Metric-Indoor Small (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`) | Per-frame metric depth for the video and photo tiers | Hugging Face, fetched by `scripts/fetch_weights.py` | Apache-2.0 |
| OpenCV SIFT / LK optical flow / PnP / essential matrix | Tracking and registration (video and photos) | `opencv-python-headless` | Apache-2.0 |
| Open3D ICP and pose-graph optimisation | Drift correction (all tiers that have a trajectory) | `open3d` | MIT |
| Stray Scanner (iOS app) | LiDAR-tier capture: depth, confidence, ARKit poses, intrinsics | App Store | — |

No other learned components. Damage detection is classical image processing (`propscan/damage/detect.py`). Nothing calls our own infrastructure; after the weights are fetched, everything runs offline.

**Cache:** monocular depth predictions are cached in `.cache/depth/<sha1>.npy`, keyed by model id and image content. A cached run replays exactly. Delete `.cache/` to force the live path (`PROPSCAN_CACHE` overrides the location).
