"""One capture in, one PropertyScan out. Shared by all tiers."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .assemble import assemble
from .geometry.align import align_to_manhattan
from .geometry.drift import correct_drift
from .geometry.frames import fuse
from .geometry.plan import build_plan
from .render import render
from .schema import DriftReport


def detect_tier(path: Path) -> str:
    from .io.stray import is_stray
    if path.is_file() and path.suffix.lower() in (".mp4", ".mov", ".m4v"):
        return "video"
    if is_stray(path):
        return "lidar"
    if path.is_dir():
        return "photos"
    raise ValueError(f"cannot infer tier for {path}")


def geometry_from_frameset(fs, tier, drift=True, timings=None, gravity_aligned=True, single_room=False):
    """FrameSet -> (PlanGeom, aligned points, drift report). Shared engine for lidar and video."""
    timings = timings if timings is not None else {}
    t = time.time()
    poses, drep = correct_drift(fs, enabled=drift)
    fs.poses = poses
    timings["drift"] = time.time() - t
    t = time.time()
    P, N, _ = fuse(fs, step=1 if fs.n < 1200 else 2)
    if not gravity_aligned:
        from .geometry.align import gravity_from_normals
        up = gravity_from_normals(N)
        R = _rot_a_to_b(up, np.array([0.0, 1.0, 0.0]))
        P, N = P @ R.T, N @ R.T
        W = np.eye(4)
        W[:3, :3] = R
        fs.poses = np.einsum("ij,njk->nik", W, fs.poses)
    P, N, R, _ = align_to_manhattan(P, N)
    W = np.eye(4)
    W[:3, :3] = R
    fs.poses = np.einsum("ij,njk->nik", W, fs.poses)
    timings["fuse"] = time.time() - t
    t = time.time()
    cams = fs.poses[:, :3, 3]
    pg = build_plan(P, N, cams[:, [0, 2]], cams[:, 1], fs, single_room=single_room)
    timings["plan"] = time.time() - t
    w = (np.abs(N[:, 1]) < 0.3) & (P[:, 1] > pg.floor_y + 0.3) & (P[:, 1] < pg.floor_y + 1.9)
    drep["ghost_wall_area_m2"] = round(len(np.unique(np.floor(P[w][:, [0, 2]] / 0.025).astype(np.int64), axis=0))
                                       * 0.025 ** 2, 3)
    return pg, P, N, drep


def _rot_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def run(path, out_dir, tier="auto", drift=True, damage=True, capture_id=None, debug_points=False):
    path = Path(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tier = detect_tier(path) if tier == "auto" else tier
    capture_id = capture_id or path.stem
    timings = {}
    t0 = time.time()
    if tier == "lidar":
        from .tiers import lidar
        fs, src = lidar.load(path)
        timings["load"] = time.time() - t0
        pg, P, N, drep = geometry_from_frameset(fs, tier, drift, timings)
        scan = assemble(pg, tier, capture_id)
    elif tier == "video":
        from .tiers import video
        fs, src = video.load(path, timings=timings)
        pg, P, N, drep = geometry_from_frameset(fs, tier, drift, timings, gravity_aligned=False)
        scan = assemble(pg, tier, capture_id)
    elif tier == "photos":
        from .tiers import photos
        scan, P, fs, drep = photos.run(path, timings=timings, damage=damage)
        pg = None
    else:
        raise ValueError(tier)
    if drep is not None:
        scan.drift = DriftReport(**drep)
    if damage and fs is not None and tier != "photos":   # photo tier analyses per room
        t = time.time()
        try:
            from .damage.detect import analyse
            analyse(scan, fs, src, tier, floor_y=pg.floor_y)
        except Exception as e:  # damage must never take the plan down
            scan.warnings.append(f"damage stage failed: {type(e).__name__}: {e}")
        timings["damage"] = time.time() - t
    timings["total"] = time.time() - t0
    scan.timings_s = {k: round(v, 2) for k, v in timings.items()}
    (out / f"{capture_id}.json").write_text(scan.model_dump_json(indent=2))
    dp = None
    if debug_points and P is not None:
        dp = P[(np.abs(P[:, 1] - (pg.floor_y + 1.0)) < 0.6)][::4][:, [0, 2]] if pg else None
    render(scan, str(out / f"{capture_id}_plan.png"), str(out / f"{capture_id}_plan.svg"), debug_points=dp)
    return scan
