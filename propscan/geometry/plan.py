"""Fused cloud -> dimensioned, segmented, stitched floor plan.

Pipeline (all tiers):
  1. floor / ceiling levels from horizontal-normal height histograms
  2. 2D occupancy: wall evidence, interior evidence (floor + furniture), camera path
  3. interior = flood fill from camera path, bounded by walls
  4. room segmentation: distance-transform watershed; doorways are constrictions
  5. per-room rectilinear polygon, every edge re-fit to the wall points facing the room
  6. openings: room-room boundaries measured between jamb points; wall gaps found by
     camera rays passing through the wall plane (mirrors rejected by reflection test)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import cv2
import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

RES = 0.025  # plan grid, metres
# Wall-surface fitting mode (fix loop, docs/FIX_LOOP.md):
#   "legacy": fit each room edge to the first room-facing vertical surface 0.12-2.2 m high
#   "high":   fit to the outermost well-supported room-facing plane *above furniture height*
#             (1.5 m .. ceiling), searching up to 0.8 m outwards; merge furniture jogs
#             (fix attempt 1: rejected, see docs/FIX_LOOP.md)
#   "planes": snap each edge to the room-facing wall plane with the best *coverage along the
#             edge*, searching up to 1.0 m outwards from the occupancy boundary (fix attempt 2)
WALL_FIT = os.environ.get("PROPSCAN_WALL_FIT", "planes")


@dataclass
class Edge:
    axis: str            # 'H': wall at constant z running along x; 'V': constant x along z
    c: float             # constant coordinate
    a: float = 0.0       # span start (other axis)
    b: float = 0.0       # span end
    outward: int = 0     # +1/-1: sign of the outward normal along the constant axis
    sigma_fit: float = 0.0
    n_pts: int = 0
    observed: bool = True


@dataclass
class RoomGeom:
    label: int
    mask: np.ndarray
    edges: list[Edge]
    polygon: list[tuple[float, float]]
    floor_y: float
    floor_sigma: float
    ceil_y: float
    ceil_sigma: float
    ceil_observed: bool
    area: float = 0.0
    perimeter: float = 0.0


@dataclass
class OpeningGeom:
    kind: str
    rooms: list[int]
    edge_refs: list[tuple[int, int]]      # (room label, edge index)
    center: tuple[float, float]
    width: float
    width_sigma: float
    height: float | None
    sill: float | None
    confidence: float
    axis: str


@dataclass
class PlanGeom:
    rooms: list[RoomGeom]
    openings: list[OpeningGeom]
    floor_y: float
    ceil_y: float | None
    grid_origin: np.ndarray
    labels: np.ndarray
    warnings: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- levels

def _peak(vals, lo, hi, bin_w=0.01):
    vals = vals[(vals > lo) & (vals < hi)]
    if len(vals) < 50:
        return None, 0
    h, e = np.histogram(vals, bins=np.arange(lo, hi + bin_w, bin_w))
    h = ndi.uniform_filter1d(h.astype(float), 3)
    k = int(np.argmax(h))
    c = e[k] + bin_w / 2
    near = vals[np.abs(vals - c) < 0.03]
    return float(np.median(near)), len(near)


def levels(P, N, cam_y):
    cy = float(np.median(cam_y))
    up = P[N[:, 1] > 0.85, 1]
    dn = P[N[:, 1] < -0.85, 1]
    floor, _ = _peak(up, cy - 2.0, cy - 0.6)
    if floor is None:
        floor = float(np.percentile(P[:, 1], 1))
    ceil, n = _peak(dn, floor + 1.9, floor + 4.5)
    return floor, (ceil if n > 300 else None)


# --------------------------------------------------------------------------- grid

def _grid(P, cams, pad=0.5):
    xz = np.r_[P[:, [0, 2]], cams]
    lo = np.percentile(xz, 0.2, axis=0) - pad
    hi = np.percentile(xz, 99.8, axis=0) + pad
    shape = np.ceil((hi - lo) / RES).astype(int)[::-1]  # rows=z, cols=x
    return lo, shape


def _to_ij(xz, lo, shape):
    ij = np.floor((xz - lo) / RES).astype(int)[:, ::-1]
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    return ij, ok


def _count(xz, lo, shape):
    ij, ok = _to_ij(xz, lo, shape)
    m = np.zeros(shape, np.int32)
    np.add.at(m, (ij[ok, 0], ij[ok, 1]), 1)
    return m


def _disk(r_m):
    r = max(1, int(round(r_m / RES)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# --------------------------------------------------------------------------- polygon

def _rectilinear(contour_xz, min_edge=0.25):
    pts = contour_xz
    n = len(pts)
    edges = []
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        d = q - p
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        axis = 'H' if abs(d[0]) >= abs(d[1]) else 'V'
        c = (p[1] + q[1]) / 2 if axis == 'H' else (p[0] + q[0]) / 2
        edges.append([axis, c, L])

    def merge(es):
        changed = True
        while changed and len(es) > 4:
            changed = False
            out = []
            for e in es:
                if out and out[-1][0] == e[0]:
                    a = out[-1]
                    w = a[2] + e[2]
                    out[-1] = [a[0], (a[1] * a[2] + e[1] * e[2]) / w, w]
                    changed = True
                else:
                    out.append(e)
            if len(out) > 1 and out[0][0] == out[-1][0]:
                a, e = out[-1], out[0]
                w = a[2] + e[2]
                out[0] = [a[0], (a[1] * a[2] + e[1] * e[2]) / w, w]
                out.pop()
                changed = True
            es = out
        return es

    edges = merge(edges)
    # drop short edges repeatedly (they create jogs), then re-merge
    for _ in range(20):
        if len(edges) <= 4:
            break
        verts = _vertices(edges)
        lens = [np.hypot(*(verts[(k + 1) % len(verts)] - verts[k])) for k in range(len(verts))]
        # edge k spans vertex k-1 -> k  (vertex k joins edge k and k+1)
        span = [lens[(k - 1) % len(lens)] for k in range(len(edges))]
        k = int(np.argmin(span))
        if span[k] >= min_edge:
            break
        edges.pop(k)
        edges = merge(edges)
    return edges


def _vertices(edges):
    """vertex k is the intersection of edge k and edge k+1."""
    V = []
    for k in range(len(edges)):
        e1, e2 = edges[k], edges[(k + 1) % len(edges)]
        if e1[0] == 'H':
            V.append(np.array([e2[1], e1[1]]))
        else:
            V.append(np.array([e1[1], e2[1]]))
    return V


def _edges_to_geom(edges):
    V = _vertices(edges)
    out = []
    n = len(edges)
    area2 = 0.0
    for k in range(n):
        p, q = V[k], V[(k + 1) % n]
        area2 += p[0] * q[1] - q[0] * p[1]
    orient = np.sign(area2) or 1.0  # +1: counter-clockwise in (x,z)
    for k in range(n):
        axis, c = edges[k][0], edges[k][1]
        p, q = V[k - 1], V[k]  # edge k runs from vertex k-1 to vertex k
        if axis == 'H':
            a, b = p[0], q[0]
            # direction along +x with CCW orientation => interior is +z side => outward -z
            outward = -int(np.sign(b - a) * orient) or 1
        else:
            a, b = p[1], q[1]
            outward = int(np.sign(b - a) * orient) or 1
        out.append(Edge(axis, float(c), float(a), float(b), outward))
    return out


def _poly_from_edges(E):
    raw = [[e.axis, e.c, 1.0] for e in E]
    return [tuple(map(float, v)) for v in _vertices(raw)]


# --------------------------------------------------------------------------- main

def build_plan(P, N, cams_xz, cam_y, fs=None, wall_band=None, min_room_area=1.2,
               open_plan_merge=1.6, single_room=False):
    warnings = []
    floor, ceil = levels(P, N, cam_y)
    top = (ceil - 0.12) if ceil else floor + 2.1
    yb = P[:, 1]
    is_wall = (np.abs(N[:, 1]) < 0.3) & (yb > floor + 0.12) & (yb < min(top, floor + 2.2))
    is_floor = (N[:, 1] > 0.8) & (np.abs(yb - floor) < 0.05)
    is_stuff = (yb > floor - 0.05) & (yb < floor + 1.3) & ~is_wall
    is_hi = (np.abs(N[:, 1]) < 0.3) & (yb > floor + 1.5) & (yb < ((ceil - 0.08) if ceil else floor + 2.4))

    lo, shape = _grid(P, cams_xz)
    wall = _count(P[is_wall][:, [0, 2]], lo, shape)
    flo = _count(P[is_floor][:, [0, 2]], lo, shape)
    stuff = _count(P[is_stuff][:, [0, 2]], lo, shape)
    camm = _count(cams_xz, lo, shape)

    barrier = (wall >= 2).astype(np.uint8)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, _disk(0.04))
    evidence = ((flo > 0) | (stuff > 0) | (camm > 0)).astype(np.uint8)
    evidence = cv2.dilate(evidence, _disk(0.12))
    free = (evidence > 0) & (barrier == 0)
    lab, nl = ndi.label(free)
    cam_labels = np.unique(lab[camm > 0])
    cam_labels = cam_labels[cam_labels > 0]
    interior = np.isin(lab, cam_labels)
    interior = ndi.binary_fill_holes(interior)
    interior = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_OPEN, _disk(0.06)) > 0
    # keep components with real area
    lab, nl = ndi.label(interior)
    sizes = ndi.sum(interior, lab, range(1, nl + 1)) * RES * RES
    interior = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.8))

    # ---- room segmentation
    dt = ndi.distance_transform_edt(interior) * RES
    seeds, ns = ndi.label(dt > 0.42)
    labels = watershed(-dt, seeds, mask=interior)
    labels = _merge_regions(labels, min_room_area, open_plan_merge)
    if single_room:
        # photo tier: one folder = one room by construction; keep the largest connected piece
        lab, nl = ndi.label(labels > 0)
        if nl:
            big = 1 + int(np.argmax(ndi.sum(labels > 0, lab, range(1, nl + 1))))
            labels = (lab == big).astype(labels.dtype)

    rooms = []
    for lbl in [l for l in np.unique(labels) if l > 0]:
        m = labels == lbl
        r = _room_geometry(lbl, m, lo, P, N, is_wall, floor, ceil, warnings, is_hi)
        if r is not None:
            rooms.append(r)

    openings = [] if single_room else _crossing_doors(labels, rooms, lo, shape, P, N, floor, cams_xz)
    for o in _room_openings(labels, rooms, lo, P, N, is_wall, floor):
        if not any(set(o.rooms) == set(q.rooms) and np.hypot(o.center[0] - q.center[0], o.center[1] - q.center[1]) < 0.8
                   for q in openings):
            openings.append(o)
    if fs is not None:
        openings += _wall_gap_openings(rooms, openings, fs, P, lo, warnings)
    return PlanGeom(rooms, openings, floor, ceil, lo, labels, warnings,
                    debug=dict(wall=wall, interior=interior, dt=dt))


def _merge_regions(labels, min_area, merge_len):
    labels = labels.copy()
    for _ in range(50):
        ids = [l for l in np.unique(labels) if l > 0]
        if len(ids) <= 1:
            break
        areas = {l: (labels == l).sum() * RES * RES for l in ids}
        bnd = _boundaries(labels)
        changed = False
        # tiny regions -> neighbour with the longest shared boundary
        for l in sorted(ids, key=lambda l: areas[l]):
            if areas[l] >= min_area:
                break
            nb = {k: v for k, v in bnd.items() if l in k}
            if not nb:
                continue
            (a, b), _ = max(nb.items(), key=lambda kv: kv[1])
            other = b if a == l else a
            labels[labels == l] = other
            changed = True
            break
        if changed:
            continue
        # wide shared boundaries are open-plan, not doorways
        for (a, b), L in sorted(bnd.items(), key=lambda kv: -kv[1]):
            if L * RES > merge_len:
                labels[labels == b] = a
                changed = True
                break
        if not changed:
            break
    # relabel 1..n by descending area
    ids = [l for l in np.unique(labels) if l > 0]
    ids.sort(key=lambda l: -(labels == l).sum())
    out = np.zeros_like(labels)
    for k, l in enumerate(ids, 1):
        out[labels == l] = k
    return out


def _boundaries(labels):
    """length (in cells) of the shared boundary between every pair of labels."""
    out = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = labels[: labels.shape[0] - dy, : labels.shape[1] - dx]
        b = labels[dy:, dx:]
        m = (a != b) & (a > 0) & (b > 0)
        pairs = np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], 1)
        if len(pairs):
            u, c = np.unique(pairs, axis=0, return_counts=True)
            for (p, q), n in zip(u, c):
                out[(int(p), int(q))] = out.get((int(p), int(q)), 0) + int(n)
    return out


def _cells_to_xz(ij, lo):
    return np.c_[ij[:, 1] * RES + lo[0] + RES / 2, ij[:, 0] * RES + lo[1] + RES / 2]


def _room_geometry(lbl, m, lo, P, N, is_wall, floor, ceil, warnings, is_hi=None):
    m8 = m.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.08 / RES, True)[:, 0, :]
    if len(c) < 4:
        return None
    # cell centres -> plan coords; contour is on the inner cell centres so push out half a cell
    xz = np.c_[c[:, 0] * RES + lo[0] + RES / 2, c[:, 1] * RES + lo[1] + RES / 2]
    raw = _rectilinear(xz)
    if len(raw) < 4:
        return None
    E = _edges_to_geom(raw)

    # points inside the room footprint (slightly dilated so wall faces are included)
    poly = _poly_from_edges(E)
    path = MplPath(np.array(poly))
    xzP = P[:, [0, 2]]
    bb_lo = np.min(poly, 0) - 0.4
    bb_hi = np.max(poly, 0) + 0.4
    near = np.all((xzP > bb_lo) & (xzP < bb_hi), 1)
    idx = np.flatnonzero(near)
    inside = idx[path.contains_points(xzP[idx], radius=-0.0)]

    # refine every edge onto the wall surface that faces into the room
    Wi = np.flatnonzero(is_wall & near)
    Wp, Wn = P[Wi], N[Wi]
    if WALL_FIT == "high" and is_hi is not None:
        Hi = np.flatnonzero(is_hi & near)
        Hp, Hn = P[Hi], N[Hi]
    for e in E:
        if WALL_FIT == "high" and is_hi is not None and _fit_edge_high(e, Hp, Hn):
            continue
        if WALL_FIT == "planes":
            if not _fit_edge_planes(e, Wp, Wn):
                # no plane covers this edge: the wall was not seen. Do not fall back to whatever
                # room-facing surface is nearest (that is how furniture became "walls")
                e.observed = False
                e.sigma_fit = 0.05
            continue
        ax = 0 if e.axis == 'V' else 2       # constant coordinate axis
        al = 2 if e.axis == 'V' else 0       # along axis
        lo_s, hi_s = sorted((e.a, e.b))
        span = hi_s - lo_s
        trim = min(0.15, 0.2 * span)
        sel = ((np.abs(Wp[:, ax] - e.c) < 0.25) & (Wp[:, al] > lo_s + trim) & (Wp[:, al] < hi_s - trim)
               & (Wn[:, ax] * e.outward < -0.8))   # normal points back into the room
        v = Wp[sel, ax]
        if len(v) >= 25:
            # the surface closest to the room interior wins (skirting / furniture in front is rare
            # at this height band; a wall behind a gap is further out)
            med = np.median(v)
            core = v[np.abs(v - med) < 0.04]
            e.c = float(np.median(core))
            mad = 1.4826 * np.median(np.abs(core - e.c)) if len(core) else 0.02
            e.sigma_fit = float(mad / np.sqrt(max(len(core) / 20.0, 1)))  # ~20 pts per independent patch
            e.n_pts = int(len(core))
        else:
            e.observed = False
            e.sigma_fit = 0.03
            e.n_pts = int(len(v))
    if WALL_FIT in ("high", "planes"):
        E = _merge_jogs(E)
    poly = _poly_from_edges(E)
    V = np.array(poly)
    # update spans from refined vertices
    for k, e in enumerate(E):
        p, q = V[k - 1], V[k]
        e.a, e.b = (p[0], q[0]) if e.axis == 'H' else (p[1], q[1])

    # per-room floor / ceiling
    Pin, Nin = P[inside], N[inside]
    fl = Pin[(Nin[:, 1] > 0.85) & (np.abs(Pin[:, 1] - floor) < 0.12), 1]
    if len(fl) > 100:
        fy = float(np.median(fl))
        core = fl[np.abs(fl - fy) < 0.02]
        fs_ = float(1.4826 * np.median(np.abs(core - fy)) / np.sqrt(max(len(core) / 50, 1)))
    else:
        fy, fs_ = floor, 0.01
    ce = Pin[(Nin[:, 1] < -0.85) & (Pin[:, 1] > fy + 1.9), 1]
    cy, cs, cobs = None, 0.0, False
    if len(ce) > 150:
        y0, n = _peak(ce, fy + 1.9, fy + 4.5, 0.005)
        if y0 is not None and n > 100:
            core = ce[np.abs(ce - y0) < 0.02]
            cy = float(np.median(core))
            cs = float(1.4826 * np.median(np.abs(core - cy)) / np.sqrt(max(len(core) / 50, 1)))
            cobs = True
    if cy is None:
        # never saw this room's ceiling: lower-bound from the highest wall points, prior 2.6 m
        wtop = Pin[(np.abs(Nin[:, 1]) < 0.3), 1]
        lb = float(np.percentile(wtop, 99.5)) if len(wtop) > 50 else fy + 2.3
        cy = max(lb + 0.05, ceil if ceil else fy + 2.6)
        cs = 0.12
        cobs = False
    area = abs(_shoelace(V))
    per = float(sum(np.hypot(*(V[(k + 1) % len(V)] - V[k])) for k in range(len(V))))
    return RoomGeom(lbl, m, E, poly, fy, fs_, cy, cs, cobs, area, per)


def _fit_edge_high(e, Hp, Hn, out_max=0.8, in_max=0.25, bin_w=0.02):
    """Fit edge `e` to the outermost well-supported room-facing plane above furniture height."""
    ax = 0 if e.axis == 'V' else 2
    al = 2 if e.axis == 'V' else 0
    lo_s, hi_s = sorted((e.a, e.b))
    trim = min(0.15, 0.2 * (hi_s - lo_s))
    off = (Hp[:, ax] - e.c) * e.outward          # + = outwards from the room
    sel = ((off > -in_max) & (off < out_max) & (Hp[:, al] > lo_s + trim) & (Hp[:, al] < hi_s - trim)
           & (Hn[:, ax] * e.outward < -0.8))
    o = off[sel]
    if len(o) < 25:
        return False
    h, edges = np.histogram(o, bins=np.arange(-in_max, out_max + bin_w, bin_w))
    h = np.convolve(h, [1, 1, 1], "same")
    strong = np.flatnonzero((h >= max(15, 0.4 * h.max())))
    k = strong.max()                              # outermost strong peak
    # walk to the local maximum of that peak
    while k + 1 < len(h) and h[k + 1] >= h[k]:
        k += 1
    while k - 1 >= 0 and h[k - 1] > h[k]:
        k -= 1
    c0 = edges[k] + bin_w / 2
    core = o[np.abs(o - c0) < 0.03]
    if len(core) < 15:
        return False
    oc = float(np.median(core))
    e.c = float(e.c + oc * e.outward)
    mad = 1.4826 * np.median(np.abs(core - oc))
    e.sigma_fit = float(mad / np.sqrt(max(len(core) / 20.0, 1)))
    e.n_pts = int(len(core))
    e.observed = True
    return True


def _fit_edge_planes(e, Wp, Wn, out_max=1.0, in_max=0.25, bin_w=0.02, cell=0.05):
    """Snap edge `e` to the room-facing plane that covers the largest fraction of the edge's span.

    The occupancy mask stops wherever the floor next to a wall was not seen (furniture, camera
    never looked down there), so the wall can be up to ~1 m outside the mask. Coverage (fraction
    of 5 cm cells along the span that hold wall points within +-2 cm of the plane) is what
    separates a wall from furniture faces and clutter: a wall runs the full span."""
    ax = 0 if e.axis == 'V' else 2
    al = 2 if e.axis == 'V' else 0
    lo_s, hi_s = sorted((e.a, e.b))
    trim = min(0.15, 0.2 * (hi_s - lo_s))
    a0, a1 = lo_s + trim, hi_s - trim
    if a1 - a0 < 0.1:
        return False
    off = (Wp[:, ax] - e.c) * e.outward
    sel = (off > -in_max) & (off < out_max) & (Wp[:, al] > a0) & (Wp[:, al] < a1) & (Wn[:, ax] * e.outward < -0.8)
    o, along = off[sel], Wp[sel, al]
    if len(o) < 25:
        return False
    nb = int(np.ceil((out_max + in_max) / bin_w))
    nc = max(1, int(np.ceil((a1 - a0) / cell)))
    ib = np.clip(((o + in_max) / bin_w).astype(int), 0, nb - 1)
    ic = np.clip(((along - a0) / cell).astype(int), 0, nc - 1)
    occ = np.zeros((nb, nc), bool)
    occ[ib, ic] = True
    # a plane at bin k covers cell c if any point lies within +-1 bin (+-2 cm)
    cov = np.zeros(nb)
    for k in range(nb):
        cov[k] = occ[max(0, k - 1):k + 2].any(0).mean()
    best = cov.max()
    if best < 0.3:
        return False
    # outermost plane whose coverage is close to the best (walls are behind furniture faces)
    k = int(np.flatnonzero(cov >= 0.8 * best).max())
    c0 = -in_max + (k + 0.5) * bin_w
    core = o[np.abs(o - c0) < 0.03]
    if len(core) < 15:
        return False
    oc = float(np.median(core))
    e.c = float(e.c + oc * e.outward)
    mad = 1.4826 * np.median(np.abs(core - oc))
    e.sigma_fit = float(mad / np.sqrt(max(len(core) / 20.0, 1)))
    e.n_pts = int(len(core))
    e.observed = True
    return True


def _merge_jogs(E, max_jog=0.5, max_step=0.06, weak_jog=0.6, weak_step=0.15, weak_pts=40):
    """Remove short edges between two parallel edges that refined onto (nearly) the same plane:
    they are notches cut by furniture in the occupancy mask, not walls. A short edge with weak
    wall support (unobserved or < `weak_pts` points) is also removed when its neighbours lie
    within `weak_step` of each other: a real alcove/nib has its own wall face."""
    E = list(E)
    changed = True
    while changed and len(E) > 4:
        changed = False
        n = len(E)
        for k in range(n):
            a, e, b = E[k - 1], E[k], E[(k + 1) % n]
            L = abs(e.b - e.a)
            weak = (not e.observed) or e.n_pts < weak_pts
            if a.axis == b.axis and ((L < max_jog and abs(a.c - b.c) < max_step) or
                                     (WALL_FIT == "planes" and weak and L < weak_jog and abs(a.c - b.c) < weak_step)):
                wa, wb = abs(a.b - a.a), abs(b.b - b.a)
                a.c = (a.c * wa + b.c * wb) / max(wa + wb, 1e-9)
                a.sigma_fit = max(a.sigma_fit, b.sigma_fit)
                a.observed = a.observed and b.observed
                drop = {k, (k + 1) % n}
                E = [E[i] for i in range(n) if i not in drop]
                changed = True
                break
    return E


def _shoelace(V):
    x, z = V[:, 0], V[:, 1]
    return 0.5 * (np.dot(x, np.roll(z, -1)) - np.dot(np.roll(x, -1), z))


def _nearest_edge(room, center, axis):
    best, bd = None, 1e9
    for k, e in enumerate(room.edges):
        if e.axis != axis:
            continue
        ax = 0 if axis == 'V' else 1
        al = 1 - ax
        lo_s, hi_s = sorted((e.a, e.b))
        d = abs(center[ax] - e.c) + max(0, lo_s - center[al]) + max(0, center[al] - hi_s)
        if d < bd:
            best, bd = k, d
    return best


def _crossing_doors(labels, rooms, lo, shape, P, N, floor, cams_xz):
    """Doorways between rooms located where the camera path crosses from one room to another,
    then measured between the jamb surfaces in the wall band."""
    byl = {r.label: r for r in rooms}
    # label every camera position (nearest room within 40 cm)
    idx = ndi.distance_transform_edt(labels == 0, return_distances=True, return_indices=True)
    dist, (ii, jj) = idx
    ij, ok = _to_ij(cams_xz, lo, shape)
    cl = np.zeros(len(cams_xz), int)
    cl[ok] = labels[ii[ij[ok, 0], ij[ok, 1]], jj[ij[ok, 0], ij[ok, 1]]]
    cl[ok & (dist[np.clip(ij[:, 0], 0, shape[0] - 1), np.clip(ij[:, 1], 0, shape[1] - 1)] * RES > 0.4)] = 0
    crossings = {}
    last, last_i = 0, -1
    for i, l in enumerate(cl):
        if l == 0:
            continue
        if last and l != last and i - last_i < 200:
            key = (min(last, l), max(last, l))
            crossings.setdefault(key, []).append((cams_xz[last_i] + cams_xz[i]) / 2)
        last, last_i = l, i
    Wp = P[(np.abs(N[:, 1]) < 0.5) & (P[:, 1] > floor + 0.4) & (P[:, 1] < floor + 1.8)]
    Hp = P[(P[:, 1] > floor + 1.6)]
    out = []
    for (a, b), pts in crossings.items():
        if a not in byl or b not in byl:
            continue
        pts = np.array(pts)
        # cluster crossings (a pair of rooms can share more than one door)
        used = np.zeros(len(pts), bool)
        for k in range(len(pts)):
            if used[k]:
                continue
            grp = np.linalg.norm(pts - pts[k], axis=1) < 0.8
            used |= grp
            c = np.median(pts[grp], 0)
            o = _measure_door(byl[a], byl[b], c, Wp, Hp, floor)
            if o is not None and not any(np.hypot(o.center[0] - q.center[0], o.center[1] - q.center[1]) < 0.5
                                         for q in out):
                o.confidence = float(min(0.95, 0.6 + 0.1 * grp.sum()))
                out.append(o)
    return out


def _measure_door(ra, rb, c, Wp, Hp, floor):
    best = None
    for axis in ("H", "V"):
        ka, kb = _nearest_edge(ra, c, axis), _nearest_edge(rb, c, axis)
        if ka is None or kb is None:
            continue
        ea, eb = ra.edges[ka], rb.edges[kb]
        thick = abs(ea.c - eb.c)
        ax = 1 if axis == "H" else 0
        d = abs(c[ax] - (ea.c + eb.c) / 2)
        if thick < 0.6 and (best is None or d < best[0]):
            best = (d, axis, ka, kb, (ea.c + eb.c) / 2, thick)
    if best is None:
        return None
    _, axis, ka, kb, cpe, thick = best
    al = 0 if axis == "H" else 2
    pe = 2 if axis == "H" else 0
    cal = c[0] if axis == "H" else c[1]
    half = thick / 2 + 0.06
    band = Wp[(np.abs(Wp[:, pe] - cpe) < half) & (np.abs(Wp[:, al] - cal) < 1.6), al]
    v = np.sort(band)
    if len(v) < 20:
        return None
    gaps = np.diff(v)
    cand = [(gaps[i], v[i], v[i + 1]) for i in range(len(gaps))
            if gaps[i] > 0.45 and v[i] < cal + 0.35 and v[i + 1] > cal - 0.35]
    if not cand:
        return None
    w, l, r = max(cand)
    if w > 2.4:
        return None
    mid = (l + r) / 2
    hdr = Hp[(np.abs(Hp[:, pe] - cpe) < half) & (Hp[:, al] > l + 0.05) & (Hp[:, al] < r - 0.05)]
    ceil_here = min(ra.ceil_y, rb.ceil_y)
    h = float(np.percentile(hdr[:, 1], 2)) if len(hdr) > 15 else None
    if h is not None and h < ceil_here - 0.08:
        kind, height = "door", h - floor
    else:
        kind, height = "passage", ceil_here - floor
    center = (float(mid), float(cpe)) if axis == "H" else (float(cpe), float(mid))
    return OpeningGeom(kind, [ra.label, rb.label], [(ra.label, ka), (rb.label, kb)], center,
                       float(w), 0.008, float(height), 0.0, 0.8, axis)


def _room_openings(labels, rooms, lo, P, N, is_wall, floor):
    """Doorways between rooms = watershed boundaries; width measured between jamb points."""
    out = []
    bnd = _boundaries(labels)
    byl = {r.label: r for r in rooms}
    Wp = P[is_wall | ((np.abs(N[:, 1]) < 0.5) & (P[:, 1] > floor + 0.3))]
    for (a, b), L in bnd.items():
        if a not in byl or b not in byl or L * RES < 0.4:
            continue
        # boundary cells
        ma, mb = labels == a, labels == b
        nb = (ma & ndi.binary_dilation(mb)) | (mb & ndi.binary_dilation(ma))
        ij = np.argwhere(nb)
        xz = _cells_to_xz(ij, lo)
        ext = xz.max(0) - xz.min(0)
        center = xz.mean(0)
        # boundary running along x => doorway in a wall at constant z ('H')
        axis = 'H' if ext[0] >= ext[1] else 'V'
        al = 0 if axis == 'H' else 2
        pe = 2 if axis == 'H' else 0
        cal = center[0] if axis == 'H' else center[1]
        cpe = center[1] if axis == 'H' else center[0]
        band = Wp[(np.abs(Wp[:, pe] - cpe) < 0.15) & (np.abs(Wp[:, al] - cal) < 1.5)
                  & (Wp[:, 1] > floor + 0.4) & (Wp[:, 1] < floor + 1.8)]
        left = band[band[:, al] < cal - 0.05, al]
        right = band[band[:, al] > cal + 0.05, al]
        if len(left) < 10 or len(right) < 10:
            width = L * RES / 2
            ws, conf = 0.05, 0.4
        else:
            # jamb faces: the innermost dense layer of points on each side
            lj = float(np.percentile(left, 98))
            rj = float(np.percentile(right, 2))
            width = rj - lj
            ws, conf = 0.008, 0.85
        if width < 0.45 or width > 2.4:
            continue
        # header: lowest wall point above the gap
        gap = Wp[(Wp[:, al] > cal - width / 2 + 0.05) & (Wp[:, al] < cal + width / 2 - 0.05)
                 & (np.abs(Wp[:, pe] - cpe) < 0.15) & (Wp[:, 1] > floor + 1.5)]
        ra, rb = byl[a], byl[b]
        ceil_here = min(ra.ceil_y, rb.ceil_y)
        hdr = float(np.percentile(gap[:, 1], 2)) if len(gap) > 15 else None
        if hdr is not None and hdr < ceil_here - 0.08:
            kind, height = "door", hdr - floor
        else:
            kind, height = "passage", ceil_here - floor
        ea = _nearest_edge(ra, center, axis)
        eb = _nearest_edge(rb, center, axis)
        out.append(OpeningGeom(kind, [a, b], [(a, ea), (b, eb)], (float(center[0]), float(center[1])),
                               float(width), ws, float(height), 0.0, conf, axis))
    return out


def _wall_gap_openings(rooms, existing, fs, P, lo, warnings, cell=0.05):
    """Openings in a room's own walls (windows, doors to unscanned space): cells of the wall
    rectangle that camera rays pass *through* and that hold no wall return. A gap whose
    through-returns are the mirror image of the room is a mirror and is rejected."""
    rays_o, rays_p = _sample_rays(fs)
    tree = cKDTree(P)
    out = []
    for r in rooms:
        for k, e in enumerate(r.edges):
            if not e.observed:
                continue
            ax = 0 if e.axis == 'V' else 2
            al = 2 if e.axis == 'V' else 0
            lo_s, hi_s = sorted((e.a, e.b))
            if hi_s - lo_s < 0.6:
                continue
            o, p = rays_o, rays_p
            # camera inside (behind the wall face, room side), return beyond the wall by > 12 cm
            side_o = (o[:, ax] - e.c) * e.outward
            side_p = (p[:, ax] - e.c) * e.outward
            s = (side_o < -0.2) & (side_p > 0.35)
            if s.sum() < 30:
                continue
            o_s, p_s = o[s], p[s]
            t = (e.c - o_s[:, ax]) / (p_s[:, ax] - o_s[:, ax])
            X = o_s + t[:, None] * (p_s - o_s)
            inspan = (X[:, al] > lo_s + 0.05) & (X[:, al] < hi_s - 0.05) & \
                     (X[:, 1] > r.floor_y + 0.02) & (X[:, 1] < r.ceil_y - 0.05)
            X, p_s = X[inspan], p_s[inspan]
            if len(X) < 30:
                continue
            nx = int(np.ceil((hi_s - lo_s) / cell))
            ny = int(np.ceil((r.ceil_y - r.floor_y) / cell))
            thr = np.zeros((ny, nx), np.int32)
            ia = np.clip(((X[:, al] - lo_s) / cell).astype(int), 0, nx - 1)
            ib = np.clip(((X[:, 1] - r.floor_y) / cell).astype(int), 0, ny - 1)
            np.add.at(thr, (ib, ia), 1)
            # wall returns on the plane
            Wq = P[(np.abs(P[:, ax] - e.c) < 0.05) & (P[:, al] > lo_s) & (P[:, al] < hi_s)
                   & (P[:, 1] > r.floor_y) & (P[:, 1] < r.ceil_y)]
            on = np.zeros((ny, nx), np.int32)
            ja = np.clip(((Wq[:, al] - lo_s) / cell).astype(int), 0, nx - 1)
            jb = np.clip(((Wq[:, 1] - r.floor_y) / cell).astype(int), 0, ny - 1)
            np.add.at(on, (jb, ja), 1)
            gap = (thr >= 2) & (on == 0)
            gap = ndi.binary_closing(gap, np.ones((3, 3)))
            gl, ng = ndi.label(gap)
            for g in range(1, ng + 1):
                cells = np.argwhere(gl == g)
                if len(cells) < 40:
                    continue
                b0, b1 = cells[:, 0].min() * cell, (cells[:, 0].max() + 1) * cell
                a0, a1 = cells[:, 1].min() * cell + lo_s, (cells[:, 1].max() + 1) * cell + lo_s
                w, h = a1 - a0, b1 - b0
                if w < 0.45 or h < 0.45 or w > 3.0:
                    continue
                cen_al = (a0 + a1) / 2
                center = (cen_al, e.c) if e.axis == 'H' else (e.c, cen_al)
                if any(np.hypot(center[0] - o_.center[0], center[1] - o_.center[1]) < max(0.6, w / 2)
                       for o_ in existing + out):
                    continue
                # mirror test: reflect the through-returns of this gap back across the plane
                selg = (X[:, al] > a0) & (X[:, al] < a1) & (X[:, 1] - r.floor_y > b0) & (X[:, 1] - r.floor_y < b1)
                pr = p_s[selg].copy()
                pr[:, ax] = 2 * e.c - pr[:, ax]
                if len(pr) >= 20:
                    d, _ = tree.query(pr, distance_upper_bound=0.06)
                    frac = float(np.isfinite(d).mean())
                    if frac > 0.5:
                        warnings.append(f"room {r.label} wall {k}: gap {w:.2f}x{h:.2f} m rejected as mirror "
                                        f"({frac:.0%} of reflected returns land on real geometry)")
                        continue
                # refine width with the wall returns bounding the gap at mid-height
                mid = (Wq[:, 1] - r.floor_y > b0 + 0.1) & (Wq[:, 1] - r.floor_y < b1 - 0.1)
                L_ = Wq[mid & (Wq[:, al] < cen_al), al]
                R_ = Wq[mid & (Wq[:, al] > cen_al), al]
                ws = cell / 2
                if len(L_) > 10 and len(R_) > 10:
                    a0r, a1r = float(np.percentile(L_, 98)), float(np.percentile(R_, 2))
                    if 0.4 < a1r - a0r < w + 2 * cell:
                        a0, a1, ws = a0r, a1r, 0.01
                w = a1 - a0
                is_door = b0 < 0.12 and h > 1.8
                if not is_door and (b0 < 0.25 or np.mean(thr[gl == g]) < 3):
                    continue  # low gaps that are not doors are furniture shadows / clutter
                kind = "door" if is_door else "window"
                out.append(OpeningGeom(kind, [r.label], [(r.label, k)], center, float(w), ws,
                                       float(h), None if is_door else float(b0),
                                       float(min(0.9, 0.4 + len(cells) / 400)), e.axis))
    return out


def _sample_rays(fs, frame_step=8, px_step=8, max_depth=8.0):
    O, Pp = [], []
    for i in range(0, fs.n, frame_step):
        d = fs.depths[i]
        K, T = fs.Ks[i], fs.poses[i]
        v, u = np.mgrid[px_step // 2:d.shape[0]:px_step, px_step // 2:d.shape[1]:px_step]
        z = d[v, u]
        ok = (z > 0.2) & (z < max_depth) & fs.masks_loose[i][v, u] if hasattr(fs, "masks_loose") else (z > 0.2) & (z < max_depth)
        u, v, z = u[ok], v[ok], z[ok]
        Pc = np.c_[(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z]
        Pw = Pc @ T[:3, :3].T + T[:3, 3]
        O.append(np.repeat(T[None, :3, 3], len(Pw), 0))
        Pp.append(Pw)
    return np.concatenate(O), np.concatenate(Pp)
