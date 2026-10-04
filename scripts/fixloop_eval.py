"""Fix-loop evaluation: LiDAR repeatability between the two whole-apartment captures under
both wall-fit modes, from the *same* fused clouds (so only the wall fit differs).

    python scripts/fixloop_eval.py            # fuse once (cached in .cache/geom), evaluate both modes
    python scripts/fixloop_eval.py --evidence # also print the root-cause evidence table

Cache: .cache/geom/<capture>.npz holds the drift-corrected, Manhattan-aligned fused cloud and
camera path. It is a speed cache only; deleting it regenerates the same data.
The full end-to-end before/after runs are scripts/benchmark.py with PROPSCAN_WALL_FIT=legacy|high.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SD = ROOT / "sample data"
CAPS = {"L_ceil": SD / "single_scan_with_ceiling.zip", "L_floor": SD / "single_scan_floor_only.zip"}


def fused(rid):
    f = ROOT / ".cache" / "geom" / f"{rid}.npz"
    if f.exists():
        d = np.load(f)
        return d["P"], d["N"], d["cams"]
    from propscan.geometry.align import align_to_manhattan
    from propscan.geometry.drift import correct_drift
    from propscan.geometry.frames import fuse
    from propscan.tiers import lidar
    fs, _ = lidar.load(CAPS[rid])
    fs.poses, _ = correct_drift(fs)
    P, N, _ = fuse(fs, step=2)
    P, N, R, _ = align_to_manhattan(P, N)
    cams = fs.poses[:, :3, 3] @ R.T
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, P=P, N=N, cams=cams)
    return P, N, cams


def plan(rid, mode):
    os.environ["PROPSCAN_WALL_FIT"] = mode
    import propscan.geometry.plan as plan_mod
    importlib.reload(plan_mod)
    from propscan.assemble import assemble
    P, N, cams = fused(rid)
    pg = plan_mod.build_plan(P, N, cams[:, [0, 2]], cams[:, 1], None)
    return assemble(pg, "lidar", rid), pg, P, N


def main(evidence=False):
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark import repeatability
    out = {}
    for mode in ("legacy", "high", "planes"):
        A, pgA, PA, NA = plan("L_ceil", mode)
        B, pgB, PB, NB = plan("L_floor", mode)
        rep = repeatability(A, B)
        rep.pop("wall_rows")
        out[mode] = rep
        print(mode, json.dumps({k: v for k, v in rep.items() if k not in ("area_pairs", "ceiling_pairs")}))
    if evidence:
        # root-cause evidence: per edge, offset between the legacy plane (0.12-2.2 m band, first
        # room-facing surface) and the high-band plane (>= 1.5 m, outermost)
        import propscan.geometry.plan as pm
        offs = []
        for rid in ("L_ceil", "L_floor"):
            _, pg_l, _, _ = plan(rid, "legacy")
            _, pg_h, _, _ = plan(rid, "planes")
            for rl in pg_l.rooms:
                rh = min(pg_h.rooms, key=lambda r: np.linalg.norm(np.mean(r.polygon, 0) - np.mean(rl.polygon, 0)))
                for e in rl.edges:
                    cand = [h for h in rh.edges if h.axis == e.axis and abs(h.c - e.c) < 0.9 and
                            min(max(e.a, e.b), max(h.a, h.b)) - max(min(e.a, e.b), min(h.a, h.b)) > 0.3]
                    if cand and e.observed:
                        h = min(cand, key=lambda h: abs(h.c - e.c))
                        offs.append((h.c - e.c) * e.outward)
        offs = np.array(offs)
        ev = dict(edges=len(offs), moved_outward_gt_5cm=float(np.mean(offs > 0.05)),
                  moved_outward_gt_15cm=float(np.mean(offs > 0.15)), median_abs_move_cm=float(np.median(np.abs(offs)) * 100))
        out["evidence"] = ev
        print("evidence", ev)
    (ROOT / "benchmark" / "fixloop_eval.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main("--evidence" in sys.argv)
