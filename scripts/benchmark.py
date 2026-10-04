"""Regenerate every benchmark number from the raw captures.

    python scripts/benchmark.py            # run missing captures, then score
    python scripts/benchmark.py --force    # re-run everything
    python scripts/benchmark.py --score    # score existing runs only
    python scripts/benchmark.py --tag after --runs P_ceil   # e.g. fix-loop after-run

Outputs: benchmark/runs/<tag>/<run>.json|png, benchmark/results_<tag>.json, benchmark/REPORT_<tag>.md

Ground truth: the sample captures came with no laser/tape measurements, so every accuracy
number here is either (a) repeatability between two captures of the same rooms at the same
tier, or (b) agreement of a thinner tier with the LiDAR tier of the same capture, with the
LiDAR result's own interval folded into the calibration test. Gates that need laser ground
truth are reported as NOT MEASURED, never as passed.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from propscan.compare import match_openings, match_rooms, match_walls, register  # noqa: E402
from propscan.schema import PropertyScan  # noqa: E402

SD = ROOT / "sample data"
RUNS = {
    # id: (capture path, extra args, description)
    "L_room": (SD / "single_room.zip", [], "LiDAR, single_room"),
    "L_floor": (SD / "single_scan_floor_only.zip", [], "LiDAR, whole apartment (walls/floor)"),
    "L_ceil": (SD / "single_scan_with_ceiling.zip", [], "LiDAR, whole apartment (with ceiling)"),
    "L_floor_nodrift": (SD / "single_scan_floor_only.zip", ["--no-drift"], "ablation: poses as-is"),
    "L_ceil_nodrift": (SD / "single_scan_with_ceiling.zip", ["--no-drift"], "ablation: poses as-is"),
    "V_room": (SD / "single_room" / "rgb.mp4", ["--tier", "video"], "video, single_room"),
    "V_floor": (SD / "single_scan_floor_only" / "rgb.mp4", ["--tier", "video"], "video, apartment"),
    "V_ceil": (SD / "single_scan_with_ceiling" / "rgb.mp4", ["--tier", "video"], "video, apartment"),
    "P_ceil": (ROOT / "data" / "photos" / "with_ceiling", ["--tier", "photos"], "photos, per-room folders"),
}


def run(rid, tag, force=False):
    cap, extra, _ = RUNS[rid]
    out = ROOT / "benchmark" / "runs" / tag
    js = out / f"{rid}.json"
    if js.exists() and not force:
        return js
    if rid.startswith("V_") and not cap.exists():
        # the video lives inside the Stray zip; materialise it
        from propscan.io.stray import load_stray
        load_stray(SD / f"{cap.parent.name}.zip").video_path()
    if rid.startswith("P_") and not cap.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_photo_benchmark.py"),
                        str(SD / "single_scan_with_ceiling.zip"), str(cap)], check=True)
    cmd = [sys.executable, "-m", "propscan", "run", str(cap), "-o", str(out), "--id", rid] + extra
    print(">>", " ".join(cmd), flush=True)
    t = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"   {rid}: {time.time() - t:.0f}s", flush=True)
    return js


def load(tag, rid):
    p = ROOT / "benchmark" / "runs" / tag / f"{rid}.json"
    return PropertyScan.model_validate_json(p.read_text()) if p.exists() else None


def wall_table(A, B):
    R, t, iou = register(A, B)
    rows = []
    for ra, rb, riou in match_rooms(A, B, R, t):
        for wa, wb in match_walls(ra, rb, R, t):
            rows.append(dict(room=ra.id, wall_a=wa.id, wall_b=wb.id, a=wa.length.value, b=wb.length.value,
                             sa=wa.length.sigma, sb=wb.length.sigma, obs=wa.length.observed and wb.length.observed))
    return rows, R, t, iou


def repeatability(A, B):
    rows, R, t, iou = wall_table(A, B)
    obs = [r for r in rows if r["obs"]]
    d = np.array([abs(r["a"] - r["b"]) for r in obs])
    L = np.array([r["a"] for r in obs])
    ok = d <= np.maximum(0.01, 0.005 * L)
    rooms = match_rooms(A, B, R, t)
    ceil = [(ra.id, ra.ceiling_height.value, rb.ceiling_height.value) for ra, rb, _ in rooms
            if ra.ceiling_height.observed and rb.ceiling_height.observed]
    area = [(ra.id, ra.floor_area.value, rb.floor_area.value) for ra, rb, _ in rooms]
    ops = match_openings(A, B, R, t)
    door_d = [abs(a.width.value - b.width.value) for a, b in ops]
    return dict(plan_iou=round(iou, 3), matched_rooms=len(rooms), walls_compared=int(len(obs)),
                walls_within_1cm_or_0p5pct=round(float(ok.mean()), 3) if len(ok) else None,
                wall_abs_diff_median_cm=round(float(np.median(d) * 100), 2) if len(d) else None,
                wall_abs_diff_p90_cm=round(float(np.percentile(d, 90) * 100), 2) if len(d) else None,
                ceiling_pairs=[(r, round(a, 3), round(b, 3), round(abs(a - b) * 100, 2)) for r, a, b in ceil],
                area_pairs=[(r, round(a, 2), round(b, 2)) for r, a, b in area],
                openings_a=len(A.openings), openings_b=len(B.openings), openings_matched=len(ops),
                opening_width_diff_le_2cm=round(float(np.mean(np.array(door_d) <= 0.02)), 3) if door_d else None,
                wall_rows=rows)


def cross_tier(ref, X, gate_rel):
    rows, R, t, iou = wall_table(ref, X)
    rows = [r for r in rows if r["obs"]]
    if not rows:
        return dict(plan_iou=round(iou, 3), walls_compared=0)
    a = np.array([r["a"] for r in rows])
    b = np.array([r["b"] for r in rows])
    z = (b - a) / np.sqrt(np.array([r["sa"] for r in rows]) ** 2 + np.array([r["sb"] for r in rows]) ** 2)
    rel = (b - a) / a
    rooms = match_rooms(ref, X, R, t)
    fa = sum(ra.floor_area.value for ra, _, _ in rooms)
    fb = sum(rb.floor_area.value for _, rb, _ in rooms)
    fb_sig = float(np.sqrt(sum(rb.floor_area.sigma ** 2 for _, rb, _ in rooms)))
    return dict(plan_iou=round(iou, 3), rooms_ref=len(ref.rooms), rooms_tier=len(X.rooms), rooms_matched=len(rooms),
                walls_compared=len(rows), wall_rel_err_median_pct=round(float(np.median(np.abs(rel)) * 100), 2),
                wall_within_gate=round(float(np.mean(np.abs(rel) <= gate_rel)), 3), gate_rel_pct=gate_rel * 100,
                ci95_coverage=round(float(np.mean(np.abs(z) <= 1.96)), 3),
                matched_footprint_ref_m2=round(fa, 2), matched_footprint_tier_m2=round(fb, 2),
                matched_footprint_err_pct=round((fb - fa) / max(fa, 1e-9) * 100, 2) if fa else None,
                footprint_in_ci=bool(abs(fb - fa) <= 1.96 * fb_sig) if fa else None)


def photo_stitch(ref_json, P):
    ref = json.loads(Path(ref_json).read_text())
    ref_adj = {tuple(sorted(e)) for e in ref["adjacency"]}
    folders = {r.name for r in P.rooms}
    ref_adj_present = {e for e in ref_adj if e[0] in folders and e[1] in folders}
    name = {r.id: r.name for r in P.rooms}
    got = {tuple(sorted((name[a], name[b]))) for a, b, _ in P.adjacency}
    tp = len(got & ref_adj_present)
    overlaps = [w for w in P.warnings if "overlap" in w]
    fa = sum(v["floor_area"] for k, v in ref["rooms"].items() if k in folders)
    fb = sum(r.floor_area.value for r in P.rooms)
    sig = P.footprint_area.sigma
    folders_given = len([d for d in (ROOT / "data" / "photos" / "with_ceiling").iterdir() if d.is_dir()])
    return dict(room_folders=folders_given, rooms_recovered=len(P.rooms),
                adjacency_ref=sorted(ref_adj_present), adjacency_found=sorted(got),
                adjacency_precision=round(tp / len(got), 3) if got else None,
                adjacency_recall=round(tp / len(ref_adj_present), 3) if ref_adj_present else None,
                overlaps=overlaps, footprint_ref_m2=round(fa, 2), footprint_tier_m2=round(fb, 2),
                footprint_err_pct=round((fb - fa) / fa * 100, 2) if fa else None,
                footprint_in_ci=bool(abs(fb - fa) <= 1.96 * sig),
                gate_pass=bool(fa and abs(fb - fa) / fa <= 0.08 and not overlaps and tp == len(ref_adj_present)
                               and len(P.rooms) == folders_given))


def score(tag):
    S = {}
    L = {k: load(tag, k) for k in RUNS}
    if L["L_floor"] and L["L_ceil"]:
        S["repeatability_lidar_floor_vs_ceil"] = repeatability(L["L_ceil"], L["L_floor"])
    if L["L_room"] and L["L_ceil"]:
        S["repeatability_lidar_room_vs_ceil"] = repeatability(L["L_ceil"], L["L_room"])
    if L["L_floor_nodrift"] and L["L_ceil_nodrift"]:
        S["repeatability_lidar_floor_vs_ceil_NO_DRIFT"] = repeatability(L["L_ceil_nodrift"], L["L_floor_nodrift"])
    S["drift_ablation"] = {}
    for cap in ("floor", "ceil"):
        on, off = L[f"L_{cap}"], L[f"L_{cap}_nodrift"]
        if on and off:
            S["drift_ablation"][cap] = dict(
                footprint_on=on.footprint_area.value, footprint_off=off.footprint_area.value,
                rooms_on=len(on.rooms), rooms_off=len(off.rooms),
                ghost_wall_area_on=on.drift.ghost_wall_area_m2 if on.drift else None,
                ghost_wall_area_off=off.drift.ghost_wall_area_m2 if off.drift else None,
                loop_closures=on.drift.loop_closures if on.drift else None,
                mean_correction_m=on.drift.mean_correction_m if on.drift else None,
                max_correction_m=on.drift.max_correction_m if on.drift else None)
    for v, ref in (("V_room", "L_room"), ("V_floor", "L_floor"), ("V_ceil", "L_ceil")):
        if L[v] and L[ref]:
            S[f"video_vs_lidar_{v}"] = cross_tier(L[ref], L[v], 0.03)
    if L["P_ceil"] and L["L_ceil"]:
        S["photos_vs_lidar_P_ceil"] = cross_tier(L["L_ceil"], L["P_ceil"], 0.08)
        S["photo_stitch_P_ceil"] = photo_stitch(ROOT / "data" / "photos" / "with_ceiling" / "reference.json",
                                                L["P_ceil"])
    S["timing_s"] = {k: (v.timings_s if v else None) for k, v in L.items()}
    S["counts"] = {k: dict(rooms=len(v.rooms), openings=len(v.openings), damage=len(v.damage),
                           footprint=v.footprint_area.value, footprint_ci=v.footprint_area.ci95)
                   for k, v in L.items() if v}
    out = ROOT / "benchmark" / f"results_{tag}.json"
    slim = json.loads(json.dumps(S, default=float))
    for k in slim:
        if isinstance(slim[k], dict):
            slim[k].pop("wall_rows", None)
    out.write_text(json.dumps(slim, indent=1))
    (ROOT / "benchmark" / f"walls_{tag}.json").write_text(json.dumps(
        {k: v.get("wall_rows") for k, v in S.items() if isinstance(v, dict) and "wall_rows" in v}, indent=1,
        default=float))
    print(json.dumps(slim, indent=1)[:6000])
    return slim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    a = ap.parse_args()
    if not a.score:
        for rid in a.runs:
            run(rid, a.tag, a.force)
    score(a.tag)
