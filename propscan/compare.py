"""Compare two PropertyScans of the same space (different captures or tiers).

Captures have unrelated world frames, so plan B is registered onto plan A first: rooms are
rasterised, the 4 Manhattan rotations are tried and the translation is found by FFT
cross-correlation of the occupancy masks (maximum overlap). No scale is estimated: a
tier's scale error is exactly what is being measured.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

RES = 0.05


def _mask(polys, lo, shape):
    from matplotlib.path import Path as MP
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.c_[xx.ravel() * RES + lo[0], yy.ravel() * RES + lo[1]]
    m = np.zeros(shape[0] * shape[1], bool)
    for p in polys:
        m |= MP(np.array(p)).contains_points(pts)
    return m.reshape(shape)


def _rot(k):
    c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
    return np.array([[c, -s], [s, c]])


def register(A, B):
    """Return (R, t) mapping B plan coords onto A, and the IoU achieved."""
    PA = [r.polygon for r in A.rooms]
    best = None
    for k in range(4):
        R = _rot(k)
        PB = [[tuple(R @ np.array(p)) for p in r.polygon] for r in B.rooms]
        allp = np.array([p for poly in PA + PB for p in poly])
        span = allp.max(0) - allp.min(0)
        loA = np.array([p for poly in PA for p in poly]).min(0) - span
        loB = np.array([p for poly in PB for p in poly]).min(0)
        shape = (np.ceil(3 * span[::-1] / RES)).astype(int) + 2
        shapeB = (np.ceil(span[::-1] / RES)).astype(int) + 2
        mA = _mask(PA, loA, shape).astype(np.float32)
        mB = _mask(PB, loB, shapeB).astype(np.float32)
        cc = fftconvolve(mA, mB[::-1, ::-1], mode="valid")
        iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        inter = cc[iy, ix]
        iou = inter / (mA.sum() + mB.sum() - inter)
        t = loA + np.array([ix, iy]) * RES - loB
        if best is None or iou > best[2]:
            best = (R, t, float(iou))
    # sub-cell refinement: align matched wall coordinates (median offset per axis)
    R, t, iou = best
    return R, t, iou


def _tf(poly, R, t):
    return [tuple(R @ np.array(p) + t) for p in poly]


def match_rooms(A, B, R, t, min_iou=0.3):
    out = []
    PB = {r.id: Polygon(_tf(r.polygon, R, t)).buffer(0) for r in B.rooms}
    for ra in A.rooms:
        pa = Polygon(ra.polygon).buffer(0)
        best = None
        for rb in B.rooms:
            pb = PB[rb.id]
            inter = pa.intersection(pb).area
            iou = inter / max(pa.union(pb).area, 1e-9)
            if iou >= min_iou and (best is None or iou > best[1]):
                best = (rb, iou)
        if best:
            out.append((ra, best[0], best[1]))
    return out


def match_walls(ra, rb, R, t, max_off=0.25, min_len=0.5):
    pairs = []
    for wa in ra.walls:
        if wa.length.value < min_len:
            continue
        a0, a1 = np.array(wa.start), np.array(wa.end)
        ha = abs(a1[0] - a0[0]) >= abs(a1[1] - a0[1])
        best = None
        for wb in rb.walls:
            if wb.length.value < min_len:
                continue
            b0, b1 = R @ np.array(wb.start) + t, R @ np.array(wb.end) + t
            hb = abs(b1[0] - b0[0]) >= abs(b1[1] - b0[1])
            if ha != hb:
                continue
            ax = 1 if ha else 0
            al = 0 if ha else 1
            off = abs((a0[ax] + a1[ax]) / 2 - (b0[ax] + b1[ax]) / 2)
            sa = sorted((a0[al], a1[al]))
            sb = sorted((b0[al], b1[al]))
            ov = min(sa[1], sb[1]) - max(sa[0], sb[0])
            frac = ov / max(sa[1] - sa[0], sb[1] - sb[0])
            if off < max_off and frac > 0.6 and (best is None or off < best[0]):
                best = (off, wb)
        if best:
            pairs.append((wa, best[1]))
    return pairs


def match_openings(A, B, R, t, max_d=0.4):
    out = []
    used = set()
    for oa in A.openings:
        ca = np.array(oa.center)
        best = None
        for ob in B.openings:
            if ob.id in used:
                continue
            cb = R @ np.array(ob.center) + t
            d = np.linalg.norm(ca - cb)
            if d < max_d and (best is None or d < best[0]):
                best = (d, ob)
        if best:
            used.add(best[1].id)
            out.append((oa, best[1]))
    return out
