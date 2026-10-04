"""Gravity and Manhattan alignment of a fused cloud."""
from __future__ import annotations

import numpy as np


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def gravity_from_normals(N):
    """Estimate the up vector from normals (for tiers without an IMU-aligned world).
    The largest horizontal-surface cluster is assumed to be the floor; its normal points up."""
    # mean-shift-lite: pick the normal direction with most neighbours within 15 deg
    rng = np.random.default_rng(0)
    cand = N[rng.choice(len(N), min(400, len(N)), replace=False)]
    cos = np.abs(N @ cand.T)
    support = (cos > np.cos(np.radians(15))).sum(0)
    # vertical axis is the one orthogonal to most other surfaces too; test the top-5 clusters
    best, best_score = None, -1
    for idx in np.argsort(support)[-8:]:
        a = cand[idx]
        horiz = (np.abs(N @ a) < np.sin(np.radians(15))).sum()   # walls perpendicular to it
        para = (np.abs(N @ a) > np.cos(np.radians(15))).sum()
        score = para + horiz
        if score > best_score:
            best, best_score = a, score
    sel = np.abs(N @ best) > np.cos(np.radians(15))
    up = np.linalg.svd(N[sel] * np.sign(N[sel] @ best)[:, None], full_matrices=False)[2][0]
    up /= np.linalg.norm(up)
    # normals face the camera, so floor and ceiling are sign-symmetric; break the tie with the
    # fact that up-facing surfaces (floor, tables, beds, counters) outnumber down-facing ones
    if (N @ up > 0.9).sum() < (N @ up < -0.9).sum():
        up = -up
    return up


def manhattan_yaw(N, horizontal_tol=0.25):
    """Dominant wall direction about +y; returns theta such that rot_y(-theta) aligns walls to x/z."""
    h = np.abs(N[:, 1]) < horizontal_tol
    ang = np.arctan2(N[h, 2], N[h, 0])
    w = np.hypot(N[h, 0], N[h, 2])
    z = (w * np.exp(4j * ang)).sum()
    return np.angle(z) / 4.0, float(np.abs(z) / max(w.sum(), 1e-9))


def align_to_manhattan(P, N):
    theta, strength = manhattan_yaw(N)
    R = rot_y(theta)  # rotating by +theta about y maps the dominant wall normal onto +x
    # verify sign: after rotation, wall normals should cluster on axes
    Pa, Na = P @ R.T, N @ R.T
    h = np.abs(Na[:, 1]) < 0.25
    resid = np.abs(np.sin(2 * np.arctan2(Na[h, 2], Na[h, 0]))).mean()
    R2 = rot_y(-theta)
    Pb, Nb = P @ R2.T, N @ R2.T
    resid2 = np.abs(np.sin(2 * np.arctan2(Nb[h, 2], Nb[h, 0]))).mean()
    if resid2 < resid:
        return Pb, Nb, R2, strength
    return Pa, Na, R, strength
