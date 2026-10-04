"""LiDAR tier: Stray Scanner depth + ARKit poses + intrinsics."""
from __future__ import annotations

import numpy as np

from ..geometry.frames import FrameSet
from ..io.stray import load_stray


def load(path, frame_step: int | None = None) -> tuple[FrameSet, object]:
    cap = load_stray(path)
    if frame_step is None:
        # ~1500 frames is plenty for fusion; ARKit runs at ~46 fps
        frame_step = max(1, cap.n // 1500)
    depths, masks, loose, Ks, poses, ids = [], [], [], [], [], []
    for i in range(0, cap.n, frame_step):
        d = cap.depth(i)
        c = cap.confidence(i)
        depths.append(d)
        masks.append(c >= 2)
        loose.append(c >= 1)
        Ks.append(cap.K_depth(i, d.shape))
        poses.append(cap.poses[i])
        ids.append(i)
    fs = FrameSet(depths, masks, Ks, np.array(poses), ids, max_depth=4.5,
                  meta=dict(tier="lidar", source=str(path), n_source_frames=cap.n, frame_step=frame_step))
    fs.masks_loose = loose
    return fs, cap
