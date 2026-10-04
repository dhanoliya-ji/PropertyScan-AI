"""Photo tier: a folder of per-room photo folders, 2-8 stills each, no depth, no poses.

    property/
      kitchen/   IMG_0001.jpg ... door_to_hall.jpg
      hall/      IMG_0101.jpg ... door_to_kitchen.jpg door_to_bedroom.jpg
      bedroom/   ...

Per room:  monocular metric depth per photo -> SIFT matches between photo pairs -> PnP on the
           scaled depth, registered along a maximum spanning tree of inlier counts -> fused
           cloud -> the shared geometry engine in single-room mode.
Stitching: each `door_to_<room>.jpg` is taken from inside the room, centred on the doorway.
           Its optical axis hits the room's wall at the door: that gives the door's position
           and wall in both rooms. Rooms are then placed so the two door walls face each
           other (rooms are Manhattan, so the relative rotation is a multiple of 90 degrees)
           and the door points coincide across a wall-thickness gap.
           Without door photos, adjacency falls back to cross-room SIFT matches and the
           placement is flagged as unverified.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from shapely.geometry import Polygon

from .. import uncertainty as U
from ..assemble import assemble
from ..geometry.frames import FrameSet
from ..models import depth as mono
from ..schema import Opening, PropertyScan

DEPTH_HW = (192, 256)
WORK_W = 960
IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".PNG"}
HFOV_PRIOR_DEG = 67.2         # iPhone 1x main camera, 4:3 still (26 mm equivalent)
WALL_THICKNESS = 0.12


def _read(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    f35 = None
    try:
        ex = im.getexif() or Image.open(path).getexif()
        sub = ex.get_ifd(0x8769)
        f35 = sub.get(0xA405) or ex.get(0xA405)
    except Exception:
        pass
    w, h = im.size
    im = im.resize((WORK_W, int(round(h * WORK_W / w))), Image.LANCZOS)
    return np.array(im), (float(f35) if f35 else None)


def _K(w, h, f35):
    if f35:
        f = f35 * np.hypot(w, h) / 43.27   # 35 mm-equivalent focal is defined on the diagonal
        src = "exif"
    else:
        f = (w / 2) / np.tan(np.radians(HFOV_PRIOR_DEG) / 2)
        src = "prior"
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), src


def _essential(p1, p2, depth_i, scale_i, K, sx, depth_j):
    """Relative pose from the essential matrix (depth-free), metric baseline from the median
    ratio of model depth to unit-baseline triangulated depth."""
    if len(p1) < 10:
        return None
    E, m = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.5)
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, m2 = cv2.recoverPose(E, p1, p2, K, mask=m.copy())
    inl = m2.ravel().astype(bool)
    if inl.sum() < 10:
        return None
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    Xh = cv2.triangulatePoints(P1, P2, p1[inl].T.astype(np.float64), p2[inl].T.astype(np.float64))
    X = (Xh[:3] / Xh[3]).T
    ztri = X[:, 2]
    u = np.clip((p1[inl, 0] * sx).astype(int), 0, DEPTH_HW[1] - 1)
    v = np.clip((p1[inl, 1] * sx).astype(int), 0, DEPTH_HW[0] - 1)
    zm = depth_i[v, u] * scale_i
    g = (ztri > 1e-3) & (zm > 0.2)
    # parallax check: with too little baseline the triangulated depth is meaningless
    b1 = np.c_[p1[inl], np.ones(inl.sum())] @ np.linalg.inv(K).T
    b2 = np.c_[p2[inl], np.ones(inl.sum())] @ np.linalg.inv(K).T @ R
    cosang = (b1 * b2).sum(1) / np.linalg.norm(b1, axis=1) / np.linalg.norm(b2, axis=1)
    parallax = np.degrees(np.median(np.arccos(np.clip(cosang, -1, 1))))
    T = np.eye(4)
    T[:3, :3] = R
    if g.sum() >= 8 and parallax > 1.0:
        base = float(np.median(zm[g] / ztri[g]))
        T[:3, 3] = t.ravel() * base
        Xm = X[g] * base
    else:
        # near-pure rotation: translation unobservable, assume none
        Xm = (np.c_[p1[inl], np.ones(inl.sum())] @ np.linalg.inv(K).T)[g] * zm[g, None]
    Xj = Xm @ R.T + T[:3, 3]
    uj = np.clip((p2[inl][g, 0] * sx).astype(int), 0, DEPTH_HW[1] - 1)
    vj = np.clip((p2[inl][g, 1] * sx).astype(int), 0, DEPTH_HW[0] - 1)
    raw = depth_j[vj, uj]
    r = Xj[:, 2] / np.maximum(raw, 1e-3)
    r = r[(raw > 0.2) & (Xj[:, 2] > 0.2)]
    if len(r) < 6:
        return None
    s_ = float(np.median(r))
    if not (0.25 < s_ / max(scale_i, 1e-6) < 4.0):
        return None
    return int(inl.sum()), T, s_


def _pnp(p1, p2, depth_i, scale_i, K, sx, depth_j):
    u = np.clip((p1[:, 0] * sx).astype(int), 0, DEPTH_HW[1] - 1)
    v = np.clip((p1[:, 1] * sx).astype(int), 0, DEPTH_HW[0] - 1)
    z = depth_i[v, u] * scale_i
    g = (z > 0.2) & (z < 8)
    if g.sum() < 12:
        return None
    X = np.c_[(p1[g, 0] - K[0, 2]) * z[g] / K[0, 0], (p1[g, 1] - K[1, 2]) * z[g] / K[1, 1], z[g]]
    ok, rv, tv, inl = cv2.solvePnPRansac(X, p2[g], K, None, iterationsCount=1000, reprojectionError=8.0,
                                         confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 10:
        return None
    inl = inl[:, 0]
    rv, tv = cv2.solvePnPRefineLM(X[inl], p2[g][inl], K, None, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    Xj = X[inl] @ R.T + tv.ravel()
    uj = np.clip((p2[g][inl, 0] * sx).astype(int), 0, DEPTH_HW[1] - 1)
    vj = np.clip((p2[g][inl, 1] * sx).astype(int), 0, DEPTH_HW[0] - 1)
    raw = depth_j[vj, uj]
    r = Xj[:, 2] / np.maximum(raw, 1e-3)
    r = r[(raw > 0.2) & (Xj[:, 2] > 0.2)]
    if len(r) < 8:
        return None
    s_ = float(np.median(r))
    if not (0.25 < s_ / max(scale_i, 1e-6) < 4.0):
        return None   # implausible scale jump or weak geometry: do not let it poison the chain
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tv.ravel()
    return len(inl), T, s_


def register_room(imgs, depths, K):
    """Register a room's photos; returns poses (cam->room), scales, registered mask."""
    n = len(imgs)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    feats = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None) for im in imgs]
    bf = cv2.BFMatcher(cv2.NORM_L2)
    M = {}
    for i in range(n):
        for j in range(i + 1, n):
            if feats[i][1] is None or feats[j][1] is None:
                continue
            m = bf.knnMatch(feats[i][1], feats[j][1], k=2)
            m = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
            if len(m) >= 15:
                p1 = np.float32([feats[i][0][x.queryIdx].pt for x in m])
                p2 = np.float32([feats[j][0][x.trainIdx].pt for x in m])
                F, inl = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 2.0, 0.999)
                if inl is not None and inl.sum() >= 15:
                    k = inl.ravel().astype(bool)
                    M[(i, j)] = (p1[k], p2[k])
                    M[(j, i)] = (p2[k], p1[k])
    deg = np.zeros(n)
    for (i, j), (p, _) in M.items():
        deg[i] += len(p)
    root = int(np.argmax(deg)) if n else 0
    poses = [None] * n
    scale = np.ones(n)
    poses[root] = np.eye(4)
    sx = DEPTH_HW[1] / imgs[0].shape[1]
    changed = True
    while changed:
        changed = False
        best = None
        for (i, j), (p1, p2) in M.items():
            if poses[i] is None or poses[j] is not None:
                continue
            r = _essential(p1, p2, depths[i], scale[i], K, sx, depths[j]) or                 _pnp(p1, p2, depths[i], scale[i], K, sx, depths[j])
            if r is not None and (best is None or r[0] > best[0]):
                best = (r[0], i, j, r[1], r[2])
        if best is not None:
            _, i, j, T_j_i, s = best
            poses[j] = poses[i] @ np.linalg.inv(T_j_i)
            scale[j] = s
            changed = True
    reg = np.array([p is not None for p in poses])
    # metric anchor: mean model scale over registered photos
    g = float(np.exp(np.mean(np.log(scale[reg])))) if reg.any() else 1.0
    out = []
    for k in range(n):
        if poses[k] is None:
            out.append(np.eye(4))
        else:
            T = poses[k].copy()
            T[:3, 3] /= g
            out.append(T)
    return np.array(out), scale / g, reg, M


def _trust(d):
    gx = np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))
    m = (gx + gy) < 0.25 * d
    m[:4] = m[-4:] = False
    m[:, :4] = m[:, -4:] = False
    return m & (d > 0.2) & (d < 6.0)


def load_room(folder: Path):
    files = sorted([p for p in folder.iterdir() if p.suffix in IMG_EXT])
    imgs, f35s = zip(*[_read(p) for p in files])
    h, w = imgs[0].shape[:2]
    K, ksrc = _K(w, h, f35s[0])
    depths = [mono.predict(im, DEPTH_HW) for im in imgs]
    poses, scale, reg, _ = register_room(list(imgs), depths, K)
    Kd = K.copy()
    Kd[0] *= DEPTH_HW[1] / w
    Kd[1] *= DEPTH_HW[0] / h
    keep = np.flatnonzero(reg)
    d = [(depths[k] * scale[k]).astype(np.float32) for k in keep]
    fs = FrameSet(d, [_trust(x) for x in d], [Kd] * len(keep), poses[keep], [int(k) for k in keep],
                  max_depth=6.0, meta=dict(tier="photos", room=folder.name, files=[files[k].name for k in keep],
                                           intrinsics=ksrc, n_photos=len(files), registered=int(reg.sum())))
    fs.masks_loose = fs.masks
    door_files = {k: re.match(r"door_to_(.+)\.\w+$", files[k].name).group(1)
                  for k in keep if re.match(r"door_to_(.+)\.\w+$", files[k].name)}
    return fs, door_files, [files[k] for k in range(len(files)) if not reg[k]]


def _rot(k90):
    c, s = np.cos(k90 * np.pi / 2), np.sin(k90 * np.pi / 2)
    return np.array([[c, -s], [s, c]])


def _door_on_wall(scan_room, cam_xz, dir_xz):
    """Intersect the photo's optical axis with the room polygon; return point and outward normal."""
    V = np.array(scan_room.polygon)
    best = None
    for k in range(len(V)):
        p, q = V[k - 1], V[k]
        e = q - p
        A = np.array([[dir_xz[0], -e[0]], [dir_xz[1], -e[1]]])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        t, u = np.linalg.solve(A, p - cam_xz)
        if t > 0.2 and -0.05 <= u <= 1.05 and (best is None or t < best[0]):
            n = np.array([e[1], -e[0]])
            n /= np.linalg.norm(n)
            c = V.mean(0)
            if np.dot(n, (p + q) / 2 - c) < 0:
                n = -n
            best = (t, cam_xz + t * dir_xz, n, k)
    return best


def run(path, timings=None, damage=True):
    from ..pipeline import geometry_from_frameset
    timings = timings if timings is not None else {}
    root = Path(path)
    room_dirs = sorted([d for d in root.iterdir() if d.is_dir() and any(p.suffix in IMG_EXT for p in d.iterdir())])
    rooms, warnings = [], []
    t = time.time()
    for d in room_dirs:
        fs, doors, unreg = load_room(d)
        if unreg:
            warnings.append(f"{d.name}: {len(unreg)} photo(s) could not be registered: {[u.name for u in unreg]}")
        if fs.n < 1:
            warnings.append(f"{d.name}: no registered photos, room skipped")
            continue
        pg, P, N, _ = geometry_from_frameset(fs, "photos", drift=False, gravity_aligned=False,
                                             single_room=True)
        if not pg.rooms:
            warnings.append(f"{d.name}: no room geometry recovered")
            continue
        sub = assemble(pg, "photos", d.name)
        if damage:
            try:
                from ..damage.detect import analyse
                analyse(sub, fs, None, "photos", floor_y=pg.floor_y)
            except Exception as e:
                warnings.append(f"{d.name}: damage stage failed: {type(e).__name__}: {e}")
        r = sub.rooms[0]
        r.name = d.name
        rooms.append(dict(name=d.name, scan=sub, room=r, fs=fs, doors=doors, P=P))
    timings["rooms"] = time.time() - t

    # ---- stitch
    t = time.time()
    names = [r["name"] for r in rooms]
    door_pts = {}
    for r in rooms:
        for k, other in r["doors"].items():
            idx = r["fs"].frame_ids.index(k)
            T = r["fs"].poses[idx]
            cam = T[:3, 3][[0, 2]]
            fwd = T[:3, 2][[0, 2]]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
            hit = _door_on_wall(r["room"], cam, fwd)
            if hit is not None and other in names:
                door_pts[(r["name"], other)] = hit
    placed = {}
    transforms = {}
    order = []
    if rooms:
        first = max(rooms, key=lambda r: r["room"].floor_area.value)["name"]
        placed[first] = True
        transforms[first] = (np.eye(2), np.zeros(2))
        order.append(first)
    adjacency, openings = [], []
    progress = True
    while progress:
        progress = False
        for (a, b), hit_a in list(door_pts.items()):
            if a in placed and b not in placed and (b, a) in door_pts:
                hit_b = door_pts[(b, a)]
                Ra, ta = transforms[a]
                pa = Ra @ hit_a[1] + ta
                na = Ra @ hit_a[2]
                # choose the 90-degree rotation of b that makes its door normal oppose a's
                best = max(range(4), key=lambda k: -np.dot(_rot(k) @ hit_b[2], na))
                Rb = _rot(best)
                pb = Rb @ hit_b[1]
                tb = pa + na * WALL_THICKNESS - pb
                transforms[b] = (Rb, tb)
                placed[b] = True
                order.append(b)
                progress = True
    unplaced = [r["name"] for r in rooms if r["name"] not in placed]
    if unplaced:
        warnings.append(f"no door photos linking {unplaced}; placed in a row, adjacency unverified")
        x = max((np.array(rr["room"].polygon) @ transforms[n][0].T + transforms[n][1])[:, 0].max()
                for rr in rooms for n in [rr["name"]] if n in transforms) if transforms else 0.0
        for n in unplaced:
            rr = next(r for r in rooms if r["name"] == n)
            V = np.array(rr["room"].polygon)
            transforms[n] = (np.eye(2), np.array([x + 0.5 - V[:, 0].min(), -V[:, 1].min()]))
            x += V[:, 0].max() - V[:, 0].min() + 0.5

    # ---- assemble the whole-property scan
    out_rooms = []
    scan_damage, scan_flags, scan_scope = [], [], []
    for k, r in enumerate(rooms, 1):
        R2, t2 = transforms[r["name"]]
        rm = r["room"].model_copy(deep=True)
        rid = f"R{k}"
        rm.id = rid
        rm.floor_surface_id, rm.ceiling_surface_id = f"{rid}-FLOOR", f"{rid}-CEIL"
        rm.polygon = [tuple((R2 @ np.array(p) + t2).tolist()) for p in rm.polygon]
        rm.opening_ids = []
        for j, w in enumerate(rm.walls, 1):
            w.id = f"{rid}-W{j}"
            w.room_id = rid
            w.surface_id = f"{w.id}-S"
            w.start = tuple((R2 @ np.array(w.start) + t2).tolist())
            w.end = tuple((R2 @ np.array(w.end) + t2).tolist())
        out_rooms.append(rm)
    name2id = {r["name"]: f"R{k}" for k, r in enumerate(rooms, 1)}
    seen = set()
    for (a, b), hit in door_pts.items():
        if (b, a) in seen or a not in name2id or b not in name2id:
            continue
        seen.add((a, b))
        R2, t2 = transforms[a]
        c = R2 @ hit[1] + t2
        ra = next(r for r in out_rooms if r.id == name2id[a])
        wa = ra.walls[hit[3]].id if hit[3] < len(ra.walls) else ra.walls[0].id
        wb = []
        if (b, a) in door_pts:
            rb = next(r for r in out_rooms if r.id == name2id[b])
            kb = door_pts[(b, a)][3]
            wb = [rb.walls[kb].id] if kb < len(rb.walls) else []
        oid = f"O{len(openings) + 1}"
        # door width is not observable from a single centred photo with confidence: use the
        # room's own wall-gap detection near the door point if any, else a wide prior
        width = _nearest_gap_width(rooms, a, hit, transforms)
        w_m = U.length("photos", width[0], (width[1],)) if width else U._m(0.82, 0.08, "m", False)
        openings.append(Opening(id=oid, kind="door", room_ids=[name2id[a], name2id[b]], wall_ids=[wa] + wb,
                                center=tuple(c.tolist()), width=w_m, height=None,
                                detection_confidence=0.6 if width else 0.4))
        adjacency.append((name2id[a], name2id[b], oid))
        for rid in (name2id[a], name2id[b]):
            next(r for r in out_rooms if r.id == rid).opening_ids.append(oid)
    # other openings each room detected on its own walls
    for k, r in enumerate(rooms, 1):
        R2, t2 = transforms[r["name"]]
        for o in r["scan"].openings:
            c = R2 @ np.array(o.center) + t2
            if any(np.hypot(*(np.array(q.center) - c)) < 0.6 for q in openings):
                continue
            oid = f"O{len(openings) + 1}"
            openings.append(o.model_copy(update=dict(id=oid, room_ids=[f"R{k}"], center=tuple(c.tolist()),
                                                     wall_ids=[w.replace(o.room_ids[0], f"R{k}") for w in o.wall_ids])))
            out_rooms[k - 1].opening_ids.append(oid)
    # damage found per room (room-local frame) -> re-key to the stitched ids
    for k, r in enumerate(rooms, 1):
        sub = r["scan"]
        R2, t2 = transforms[r["name"]]
        rk = lambda sid: sid.replace("R1-", f"R{k}-", 1)
        dmap = {}
        for dmg in sub.damage:
            nid = f"D{len(scan_damage) + 1}"
            dmap[dmg.id] = nid
            scan_damage.append(dmg.model_copy(update=dict(id=nid, surface_id=rk(dmg.surface_id))))
        for f in sub.concealed_damage:
            scan_flags.append(f.model_copy(update=dict(id=f"C{len(scan_flags) + 1}", surface_id=rk(f.surface_id),
                                                       evidence=[dmap.get(e, e) for e in f.evidence])))
        for it in sub.scope:
            scan_scope.append(it.model_copy(update=dict(id=f"S{len(scan_scope) + 1}", surface_id=rk(it.surface_id))))
    # overlap check
    polys = [Polygon(r.polygon).buffer(0) for r in out_rooms]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            ov = polys[i].intersection(polys[j]).area
            if ov > 0.05:
                warnings.append(f"rooms {out_rooms[i].name} and {out_rooms[j].name} overlap by {ov:.2f} m2")
    tot = sum(r.floor_area.value for r in out_rooms)
    sig = float(np.sqrt(sum(r.floor_area.sigma ** 2 for r in out_rooms)))
    # room-level scale errors are independent per room in this tier
    fp = U._m(tot, sig, "m2", True)
    timings["stitch"] = time.time() - t
    scan = PropertyScan(capture_id=root.name, tier="photos", rooms=out_rooms, openings=openings,
                        adjacency=adjacency, footprint_area=fp, warnings=warnings,
                        damage=scan_damage, concealed_damage=scan_flags, scope=scan_scope,
                        error_budget=dict(U.BUDGETS["photos"]))
    allP = None
    # FrameSet list kept for damage analysis
    scan_fs = [r["fs"] for r in rooms]
    return scan, allP, scan_fs, None


def _nearest_gap_width(rooms, a, hit, transforms):
    r = next(x for x in rooms if x["name"] == a)
    best = None
    for o in r["scan"].openings:
        d = np.hypot(*(np.array(o.center) - hit[1]))
        if d < 0.8 and (best is None or d < best[0]):
            best = (d, o.width.value, o.width.sigma)
    return (best[1], best[2]) if best else None
