"""SYNTHETIC damage-injection test (the sample data contains no staged damage).

Two damage regions of known metric size are defined *in 3D on a measured wall plane* and
painted into every analysed RGB frame by projecting them with that frame's pose, so they
are multi-view consistent like real damage (occlusion-tested against the frame's depth):
  - water_stain: brownish diffuse ellipse 0.40 m x 0.25 m
  - crack: dark 4 mm line, 0.50 m long, at 35 deg
The detector then runs unchanged. We report detected class, surface and metric extent vs truth.
This validates localisation/extent/rule logic only; it is NOT evidence on real damage.

    python scripts/synthetic_damage_test.py "sample data/single_room.zip"
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propscan.assemble import assemble  # noqa: E402
from propscan.damage import detect  # noqa: E402
from propscan.pipeline import geometry_from_frameset  # noqa: E402
from propscan.tiers import lidar  # noqa: E402


def main(src):
    fs, cap = lidar.load(src)
    pg, P, N, _ = geometry_from_frameset(fs, "lidar", drift=True)
    scan = assemble(pg, "lidar", "synthetic")
    # pick the observed wall seen by the most frames
    best = None
    for r in scan.rooms:
        for w in r.walls:
            if not w.length.observed or w.length.value < 1.5:
                continue
            (x0, z0), (x1, z1) = w.start, w.end
            mid = np.array([(x0 + x1) / 2, pg.floor_y + 1.2, (z0 + z1) / 2])
            cams = fs.poses[:, :3, 3]
            fwd = fs.poses[:, :3, 2]
            d = mid - cams
            dist = np.linalg.norm(d, axis=1)
            seen = ((d * fwd).sum(1) / dist > 0.85) & (dist < 3.0)
            if seen.sum() < 5:
                continue
            # brightness/flatness of the wall around `mid` in a few viewing frames (plain painted wall)
            vals = []
            for k in np.flatnonzero(seen)[:: max(1, seen.sum() // 5)][:5]:
                Ti = np.linalg.inv(fs.poses[k])
                Xc = Ti[:3, :3] @ mid + Ti[:3, 3]
                K = fs.Ks[k]
                u, v = int(K[0, 0] * Xc[0] / Xc[2] + K[0, 2]), int(K[1, 1] * Xc[1] / Xc[2] + K[1, 2])
                vals.append((u, v, fs.frame_ids[k]))
            score = seen.sum()
            if best is None or score > best[0]:
                best = (score, r, w, mid, vals)
    _, room, wall, mid, _ = best
    (x0, z0), (x1, z1) = wall.start, wall.end
    along = np.array([x1 - x0, 0, z1 - z0])
    along /= np.linalg.norm(along)
    up = np.array([0, 1.0, 0])
    stain_c = mid - 0.45 * along
    crack_c = mid + 0.45 * along
    truth = [dict(cls="water_stain", w=0.40, h=0.25), dict(cls="crack", w=0.50 * np.cos(np.radians(35)),
                                                          h=0.50 * np.sin(np.radians(35)))]

    def paint(bgr, depth, K, T):
        H, W = bgr.shape[:2]
        Ku = K.copy()
        Ku[:2] *= 2
        Tinv = np.linalg.inv(T)
        img = bgr.astype(np.float32)
        dz = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        def proj(X):
            Xc = X @ Tinv[:3, :3].T + Tinv[:3, 3]
            ok = Xc[:, 2] > 0.2
            uv = Xc[:, :2] / Xc[:, 2:3] * [Ku[0, 0], Ku[1, 1]] + [Ku[0, 2], Ku[1, 2]]
            return uv, Xc[:, 2], ok
        # stain: project the 3D ellipse outline, fill, feathered alpha
        th = np.linspace(0, 2 * np.pi, 72)
        X = stain_c + (0.2 * np.cos(th))[:, None] * along + (0.125 * np.sin(th))[:, None] * up
        uv, z, ok = proj(X)
        if ok.all():
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [uv.astype(np.int32)], 255)
            if m.any():
                dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
                a = np.clip(dist / max(dist.max() * 0.5, 1), 0, 1)[..., None] * 0.28
                img = img * (1 - a) + img * np.array([0.55, 0.62, 0.70]) * a / 0.28 * 0.28 + img * 0                     if False else img * (1 - a) + (img * np.array([0.62, 0.70, 0.78])) * a
        # crack: thin dark polyline
        t = np.linspace(-0.25, 0.25, 200)
        dirc = np.cos(np.radians(35)) * along + np.sin(np.radians(35)) * up
        X = crack_c + t[:, None] * dirc + 0.006 * np.sin(t * 50)[:, None] * up
        uv, z, ok = proj(X)
        if ok.all():
            cv2.polylines(img, [uv.astype(np.int32)], False, (45, 45, 50), 2, cv2.LINE_AA)
        return np.clip(img, 0, 255).astype(np.uint8)

    orig = detect._frames

    def frames(fs_, src_, tier_, max_frames=60):
        for bgr, depth, K, T, fid, mask in orig(fs_, src_, tier_, max_frames):
            yield paint(bgr, depth, K, T), depth, K, T, fid, mask
    detect._frames = frames
    detect.analyse(scan, fs, cap, "lidar", floor_y=pg.floor_y)
    res = dict(note="SYNTHETIC injection on a real wall of the sample capture; not real damage",
               wall=wall.surface_id, truth=truth,
               detected=[dict(id=d.id, cls=d.damage_class, surface=d.surface_id, w=round(d.width.value, 3),
                              h=round(d.height.value, 3), w_ci=d.width.ci95, h_ci=d.height.ci95,
                              frames=len(d.source_frames)) for d in scan.damage],
               concealed=[dict(rule=f.rule_id, surface=f.surface_id) for f in scan.concealed_damage],
               scope=[dict(code=s.code, surface=s.surface_id, qty=s.quantity.value, unit=s.unit) for s in scan.scope])
    out = Path(__file__).resolve().parents[1] / "benchmark" / "synthetic_damage.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
