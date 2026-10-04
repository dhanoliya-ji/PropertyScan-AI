"""Tier-independent input to the geometry engine.

Every tier reduces its capture to a FrameSet: per-frame metric depth, a validity mask,
pinhole intrinsics at depth resolution, and a camera->world pose (OpenCV camera axes,
world y up). The engine never knows which tier produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameSet:
    depths: list[np.ndarray]          # (H,W) metres, 0 = invalid
    masks: list[np.ndarray]           # (H,W) bool, trusted pixels
    Ks: list[np.ndarray]              # (3,3) at depth resolution
    poses: np.ndarray                 # (N,4,4) camera->world
    frame_ids: list[int]              # index into the source (video frame / photo index)
    max_depth: float = 4.5
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.depths)


def backproject(depth, mask, K, pose, max_depth, with_normals=True):
    """Return world points (M,3) and world normals (M,3) (oriented towards the camera)."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    P = np.stack([x, y, z], -1)
    valid = mask & (z > 0.15) & (z < max_depth)
    N = None
    if with_normals:
        # central differences on the organised cloud; reject across depth discontinuities
        dx = np.zeros_like(P)
        dy = np.zeros_like(P)
        dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
        dy[1:-1] = P[2:] - P[:-2]
        n = np.cross(dx, dy)
        nn = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.maximum(nn, 1e-9)
        # orient towards camera
        flip = (n * P).sum(-1) > 0
        n[flip] *= -1
        jump = (np.abs(dx[..., 2]) > 0.06 * z) | (np.abs(dy[..., 2]) > 0.06 * z)
        valid &= (nn[..., 0] > 0) & ~jump
        valid[0, :] = valid[-1, :] = valid[:, 0] = valid[:, -1] = False
        N = n[valid] @ pose[:3, :3].T
    Pw = P[valid] @ pose[:3, :3].T + pose[:3, 3]
    return Pw, N


def fuse(fs: FrameSet, step: int = 1, voxel: float = 0.02):
    """Fuse all frames into one voxel-averaged cloud with normals and observation counts."""
    pts, nrm = [], []
    for i in range(0, fs.n, step):
        P, N = backproject(fs.depths[i], fs.masks[i], fs.Ks[i], fs.poses[i], fs.max_depth)
        if len(P) > 6000:  # cap per-frame contribution so slow sweeps don't dominate
            sel = np.random.default_rng(i).choice(len(P), 6000, replace=False)
            P, N = P[sel], N[sel]
        pts.append(P)
        nrm.append(N)
    P = np.concatenate(pts)
    N = np.concatenate(nrm)
    return voxel_down(P, N, voxel)


def voxel_down(P, N, voxel):
    keys = np.floor(P / voxel).astype(np.int64)
    keys -= keys.min(0)
    dims = keys.max(0) + 1
    lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    uniq, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
    Ps = np.zeros((len(uniq), 3))
    Ns = np.zeros((len(uniq), 3))
    np.add.at(Ps, inv, P)
    np.add.at(Ns, inv, N)
    Ps /= cnt[:, None]
    Ns /= np.maximum(np.linalg.norm(Ns, axis=1, keepdims=True), 1e-9)
    return Ps, Ns, cnt
