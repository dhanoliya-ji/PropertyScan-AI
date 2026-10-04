"""Per-surface damage regions, concealed-damage flags and scope line items.

Detection is deliberately classical and offline (no weights to fetch, deterministic):
only pixels whose 3D point lies within 3 cm of a measured wall/ceiling plane are examined,
so furniture never enters. On those pixels:
  - stain / mould: local darkening vs a large-scale (shading-free) estimate of the surface;
    large smooth blobs -> water_stain, many small dark spots -> mold
  - crack: thin dark ridges (black-hat morphology), elongated components
  - hole: (LiDAR only) depth recessed > 2.5 cm behind the plane in a compact blob
A region is kept only if it is re-detected in >= 2 frames at the same 3D location; metric
extent comes from the back-projected pixels expressed in the surface's own (u, height) axes.
Classes and thresholds are documented in docs/DAMAGE.md, with known false positives.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import uncertainty as U
from ..schema import ConcealedDamageFlag, DamageRegion, ScopeLineItem

PLANE_TOL = 0.03


def _surfaces(scan):
    """Planes of every wall and ceiling in plan coordinates: (surface_id, kind, room, axis, c, span, y0, y1)."""
    out = []
    for r in scan.rooms:
        V = np.array(r.polygon)
        fl = None
        for w in r.walls:
            (x0, z0), (x1, z1) = w.start, w.end
            if abs(x1 - x0) >= abs(z1 - z0):
                out.append((w.surface_id, "wall", r, "H", (z0 + z1) / 2, sorted((x0, x1)), w))
            else:
                out.append((w.surface_id, "wall", r, "V", (x0 + x1) / 2, sorted((z0, z1)), w))
        out.append((r.ceiling_surface_id, "ceiling", r, "Y", None, None, None))
    return out


def _frames(fs, src, tier, max_frames=60):
    """Yield (rgb at depth resolution, depth, K, pose, frame id)."""
    if isinstance(fs, list):          # photo tier: one FrameSet per room
        for f in fs:
            yield from _frames(f, None, "photos", max_frames // max(len(fs), 1) + 2)
        return
    n = fs.n
    pick = set(np.linspace(0, n - 1, min(max_frames, n)).round().astype(int).tolist())
    if getattr(fs, "rgb", None) is not None:
        for k in sorted(pick):
            yield _rs(fs.rgb[k], fs.depths[k].shape), fs.depths[k], fs.Ks[k], fs.poses[k], fs.frame_ids[k], fs.masks[k]
        return
    if src is not None and hasattr(src, "video_path"):
        want = {fs.frame_ids[k]: k for k in pick}
        vc = cv2.VideoCapture(src.video_path())
        i = 0
        last = max(want)
        while i <= last:
            ok = vc.grab()
            if not ok:
                break
            if i in want:
                ok, f = vc.retrieve()
                k = want[i]
                yield _rs(f, fs.depths[k].shape), fs.depths[k], fs.Ks[k], fs.poses[k], i, fs.masks[k]
            i += 1


def _rs(bgr, shape):
    return cv2.resize(bgr, (shape[1] * 2, shape[0] * 2), interpolation=cv2.INTER_AREA)


def _detect_2d(bgr, valid):
    """Return list of (class, mask) candidates inside `valid` (surface pixels, image res)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    bg = cv2.medianBlur(g.astype(np.uint8), 31).astype(np.float32)
    rel = (bg - g) / np.maximum(bg, 20)              # darker than surroundings, shading-free
    out = []
    v8 = valid.astype(np.uint8)
    inner = cv2.erode(v8, np.ones((9, 9), np.uint8)) > 0
    dark = (rel > 0.18) & inner
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    small_spots = np.zeros_like(dark)
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        m = lab == k
        if a >= 150 and min(w, h) > 8 and a / (w * h) > 0.35 and max(w, h) < 3 * min(w, h)                 and np.median(rel[m]) < 0.40:
            # stains are diffuse and low-contrast; handles, switches and frames are dark, thin, crisp
            # broad, filled, low-contrast-edged blob: stain. Reject hard-edged dark objects
            # (pictures, outlets, door hardware) by edge sharpness along its boundary
            edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            grad = cv2.Laplacian(g, cv2.CV_32F)[edge]
            if np.mean(np.abs(grad)) < 12:
                out.append(("water_stain", m))
        elif 4 <= a < 60 and np.median(sat[m]) < 70:
            # mould spots are grey/black/dark green; magnets, labels and prints are saturated
            small_spots |= m.astype(np.uint8)
    # mould: dense clusters of small dark spots
    dens = cv2.boxFilter(small_spots.astype(np.float32), -1, (25, 25))
    cl = (dens > 0.12) & inner
    n, lab, st, _ = cv2.connectedComponentsWithStats(cl.astype(np.uint8))
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= 200:
            out.append(("mold", lab == k))
    # cracks: thin dark ridges, long and thin
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ridge = ((bh > 14) & inner).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ridge)
    for k in range(1, n):
        w, h, a = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT], st[k, cv2.CC_STAT_AREA]
        L = max(w, h)
        if L >= 40 and a / max(L, 1) < 3.0 and min(w, h) > 3:   # long, thin, and not an axis-aligned edge
            out.append(("crack", lab == k))
    return out


def analyse(scan, fs, src, tier, floor_y=0.0):
    global _FLOOR
    _FLOOR = floor_y
    surfaces = _surfaces(scan)
    if not surfaces:
        return
    dets = []   # (class, surface_id, 3D points)
    for bgr, depth, K, T, fid, mask in _frames(fs, src, tier):
        H, W = depth.shape
        Ku = K.copy()
        Ku[:2] *= 2  # rgb is at 2x depth resolution
        d2 = cv2.resize(depth, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
        m2 = cv2.resize(mask.astype(np.uint8), (W * 2, H * 2), interpolation=cv2.INTER_NEAREST) > 0
        u, v = np.meshgrid(np.arange(W * 2), np.arange(H * 2))
        z = d2
        Pc = np.stack([(u + 0.5 - Ku[0, 2]) * z / Ku[0, 0], (v + 0.5 - Ku[1, 2]) * z / Ku[1, 1], z], -1)
        Pw = Pc @ T[:3, :3].T + T[:3, 3]
        # assign every pixel to the nearest surface plane within tolerance
        sid = np.full(z.shape, -1, int)
        best = np.full(z.shape, PLANE_TOL)
        for k, (s_id, kind, r, axis, c, span, w) in enumerate(surfaces):
            if kind == "wall":
                ax = 0 if axis == "V" else 2
                al = 2 if axis == "V" else 0
                dist = np.abs(Pw[..., ax] - c)
                ok = (Pw[..., al] > span[0]) & (Pw[..., al] < span[1])
            else:
                yc = scan_floor_y(scan) + r.ceiling_height.value
                dist = np.abs(Pw[..., 1] - yc)
                ok = np.ones_like(dist, bool)
            sel = ok & (dist < best) & m2 & (z > 0.3)
            sid[sel] = k
            best[sel] = dist[sel]
        valid = sid >= 0
        if valid.mean() < 0.05:
            continue
        for cls, m in _detect_2d(bgr, valid):
            ks = sid[m & valid]
            if len(ks) < 20:
                continue
            k = int(np.bincount(ks).argmax())
            pts = Pw[m & (sid == k)]
            dets.append((cls, k, pts, fid))
        # holes (LiDAR tier only: monocular depth cannot resolve 2.5 cm recesses)
        if tier == "lidar":
            for k in np.unique(sid[valid]):
                s_id, kind, r, axis, c, span, w = surfaces[k]
                if kind != "wall":
                    continue
                ax = 0 if axis == "V" else 2
                al = 2 if axis == "V" else 0
                band = (np.abs(Pw[..., ax] - c) < 0.12) & (Pw[..., al] > span[0] + 0.1) & (Pw[..., al] < span[1] - 0.1) & m2
                rec = (Pw[..., ax] - c) * _outward(r, w)
                behind = band & (rec > 0.03) & (rec < 0.08)
                onplane = band & (np.abs(rec) < 0.012)
                n, lab, st, _ = cv2.connectedComponentsWithStats(behind.astype(np.uint8))
                for j in range(1, n):
                    a, ww, hh = st[j, cv2.CC_STAT_AREA], st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
                    if 15 <= a <= 300 and max(ww, hh) < 2.5 * min(ww, hh):
                        blob = (lab == j).astype(np.uint8)
                        ring = (cv2.dilate(blob, np.ones((7, 7), np.uint8)) > 0) & (blob == 0)
                        # a hole is a recess *in* a flat wall: its surround must be on the plane
                        if ring.sum() and (onplane & ring).sum() / ring.sum() > 0.8:
                            dets.append(("hole", int(k), Pw[lab == j], fid))
    _cluster_and_emit(scan, surfaces, dets, tier)


_FLOOR = 0.0


def scan_floor_y(scan):
    """World y of the floor in the fused frame the surfaces were measured in."""
    return _FLOOR


def _outward(r, w):
    V = np.array(r.polygon)
    c = V.mean(0)
    (x0, z0), (x1, z1) = w.start, w.end
    if abs(x1 - x0) >= abs(z1 - z0):
        return 1.0 if (z0 + z1) / 2 > c[1] else -1.0
    return 1.0 if (x0 + x1) / 2 > c[0] else -1.0


def _cluster_and_emit(scan, surfaces, dets, tier):
    clusters = []
    for cls, k, pts, fid in dets:
        cen = pts.mean(0)
        for cl in clusters:
            if cl["cls"] == cls and cl["k"] == k and np.linalg.norm(cl["cen"] - cen) < 0.15:
                cl["pts"].append(pts)
                cl["frames"].add(fid)
                cl["cen"] = np.concatenate(cl["pts"]).mean(0)
                break
        else:
            clusters.append(dict(cls=cls, k=k, cen=cen, pts=[pts], frames={fid}))
    ridx = {r.id: r for r in scan.rooms}
    for cl in clusters:
        if len(cl["frames"]) < 3:
            continue   # detections must repeat across >= 3 views to be trusted
        s_id, kind, r, axis, c, span, w = surfaces[cl["k"]]
        P = np.concatenate(cl["pts"])
        if kind == "wall":
            al = 2 if axis == "V" else 0
            uu, hh = P[:, al], P[:, 1]
        else:
            uu, hh = P[:, 0], P[:, 2]
        wd = float(np.percentile(uu, 97) - np.percentile(uu, 3))
        ht = float(np.percentile(hh, 97) - np.percentile(hh, 3))
        if cl["cls"] == "crack":
            # tile grout, skirting and panel joints are axis-aligned; cracks rarely are
            q = np.c_[uu - uu.mean(), hh - hh.mean()]
            ev = np.linalg.eigh(q.T @ q)[1][:, -1]
            ang = np.degrees(np.arctan2(abs(ev[1]), abs(ev[0])))
            if min(ang, 90 - ang) < 12:
                continue
        if cl["cls"] == "crack":
            area = wd * 0.002 + ht * 0.002
        else:
            area = wd * ht * 0.7
        did = f"D{len(scan.damage) + 1}"
        scan.damage.append(DamageRegion(
            id=did, surface_id=s_id, damage_class=cl["cls"], score=round(min(0.95, 0.4 + 0.1 * len(cl["frames"])), 2),
            width=U.length(tier, wd, (0.01,)), height=U.length(tier, ht, (0.01,)),
            area=U._m(area, max(0.2 * area, 0.002), "m2", True),
            centroid_3d=tuple(map(float, cl["cen"])), source_frames=sorted(int(f) for f in cl["frames"])[:20]))
    _rules(scan, surfaces)
    _scope(scan, surfaces)


RULES = {
    "R1": ("water_stain on a ceiling", "Active or past leak from the floor/roof above; cavity likely wet", "high"),
    "R2": ("water_stain or mold within 0.3 m of the floor on a wall",
           "Plumbing leak or rising damp behind the wall; insulation/bottom plate may be affected", "medium"),
    "R3": ("mold on any surface", "Visible mould implies sustained moisture; growth behind the board is likely", "high"),
    "R4": ("crack longer than 0.3 m", "Possible movement/settlement; check framing and the opening corners", "medium"),
    "R5": ("two or more damage regions on surfaces of the same room, one of them moisture-related",
           "Room-level moisture source; perform a moisture-meter survey of all its walls", "medium"),
}


def _rules(scan, surfaces):
    sid_kind = {s[0]: s[1] for s in surfaces}
    for d in scan.damage:
        fired = []
        if d.damage_class == "water_stain" and sid_kind.get(d.surface_id) == "ceiling":
            fired.append("R1")
        if d.damage_class in ("water_stain", "mold") and sid_kind.get(d.surface_id) == "wall":
            if d.centroid_3d[1] - d.height.value / 2 - _floor_for(scan, d) < 0.3:
                fired.append("R2")
        if d.damage_class == "mold":
            fired.append("R3")
        if d.damage_class == "crack" and max(d.width.value, d.height.value) > 0.3:
            fired.append("R4")
        for rid in fired:
            cond, rule, sev = RULES[rid]
            scan.concealed_damage.append(ConcealedDamageFlag(
                id=f"C{len(scan.concealed_damage) + 1}", surface_id=d.surface_id, rule_id=rid,
                rule=f"IF {cond} THEN {rule}", evidence=[d.id], severity=sev))
    by_room = {}
    for d in scan.damage:
        by_room.setdefault(d.surface_id.split("-")[0], []).append(d)
    for rid, ds in by_room.items():
        if len(ds) >= 2 and any(x.damage_class in ("water_stain", "mold") for x in ds):
            cond, rule, sev = RULES["R5"]
            scan.concealed_damage.append(ConcealedDamageFlag(
                id=f"C{len(scan.concealed_damage) + 1}", surface_id=f"{rid}-FLOOR", rule_id="R5",
                rule=f"IF {cond} THEN {rule}", evidence=[x.id for x in ds], severity=sev))


def _floor_for(scan, d):
    return _FLOOR


def _scope(scan, surfaces):
    area_of = {}
    for r in scan.rooms:
        for w in r.walls:
            area_of[w.surface_id] = w.area
        area_of[r.ceiling_surface_id] = r.floor_area
    items = []

    def add(sid, code, desc, qty, unit, reason):
        items.append(ScopeLineItem(id=f"S{len(items) + 1}", surface_id=sid, code=code, description=desc,
                                   quantity=qty, unit=unit, reason=reason))
    painted = set()
    for d in scan.damage:
        if d.damage_class == "crack":
            L = max(d.width.value, d.height.value)
            add(d.surface_id, "DRY-CRK", "Rake out, tape and fill drywall/plaster crack",
                U._m(L * 1.1, max(d.width.sigma, d.height.sigma), "m", True), "m", d.id)
        elif d.damage_class == "hole":
            add(d.surface_id, "DRY-PATCH", "Cut and patch hole in drywall (up to 0.1 m2)",
                U._m(max(d.area.value, 0.01), d.area.sigma, "m2", True), "m2", d.id)
        elif d.damage_class == "water_stain":
            a = max(1.0, d.area.value * 1.5)
            add(d.surface_id, "PNT-SEAL", "Stain-blocking primer over water stain",
                U._m(a, max(d.area.sigma * 1.5, 0.1), "m2", True), "m2", d.id)
        elif d.damage_class == "mold":
            a = (d.width.value + 0.6) * (d.height.value + 0.6)
            add(d.surface_id, "MLD-REM", "Mould remediation: HEPA clean, remove affected board + 0.3 m margin",
                U._m(a, max(d.area.sigma, 0.1), "m2", True), "m2", d.id)
            add(d.surface_id, "DRY-REPL", "Replace drywall removed for remediation",
                U._m(a, max(d.area.sigma, 0.1), "m2", True), "m2", d.id)
        if d.surface_id not in painted and d.surface_id in area_of:
            painted.add(d.surface_id)
            add(d.surface_id, "PNT-FULL", "Prime and paint the full surface (match finish edge to edge)",
                area_of[d.surface_id], "m2", d.id)
    for f in scan.concealed_damage:
        add(f.surface_id, "INSP-MOIST", "Moisture-meter / thermal survey of the cavity",
            U._m(1.0, 0.0, "m", True), "ea", f"{f.id} ({f.rule_id})")
    scan.scope.extend(items)
