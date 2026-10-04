"""PlanGeom (engine output) -> PropertyScan (published contract), with per-tier intervals."""
from __future__ import annotations

import math

import numpy as np

from . import uncertainty as U
from .schema import Opening, PropertyScan, Room, Wall


def room_name(r, k):
    V = np.array(r.polygon)
    ext = V.max(0) - V.min(0)
    if min(ext) < 1.5 and max(ext) / max(min(ext), 1e-3) > 2.2:
        return f"Connector {k}"
    return f"Room {k}"


def assemble(pg, tier, capture_id, extra_rel=0.0, scale_unobserved=1.0) -> PropertyScan:
    rooms, openings, adjacency = [], [], []
    wall_id = {}
    for k, r in enumerate(pg.rooms, 1):
        rid = f"R{k}"
        h_sig = (r.floor_sigma, r.ceil_sigma)
        ceil = U.height(tier, r.ceil_y - r.floor_y, h_sig, observed=r.ceil_observed)
        walls = []
        V = np.array(r.polygon)
        n = len(r.edges)
        for j, e in enumerate(r.edges):
            p, q = V[j - 1], V[j]
            L = float(np.hypot(*(q - p)))
            # length = distance between the two neighbouring wall surfaces
            s_prev = r.edges[j - 1].sigma_fit
            s_next = r.edges[(j + 1) % n].sigma_fit
            m = U.length(tier, L, (s_prev, s_next), observed=e.observed, extra_rel=extra_rel)
            if not e.observed:
                m = U.widen(m, 2.0 * scale_unobserved, observed=False)
            wid = f"{rid}-W{j + 1}"
            wall_id[(r.label, j)] = wid
            walls.append(Wall(id=wid, room_id=rid, surface_id=f"{wid}-S", start=tuple(map(float, p)),
                              end=tuple(map(float, q)), length=m, height=ceil,
                              area=U._m(L * ceil.value, math.hypot(m.sigma * ceil.value, ceil.sigma * L), "m2",
                                        e.observed)))
        area = U.area_from_dims(tier, r.area, r.perimeter,
                                float(np.sqrt(np.mean([e.sigma_fit ** 2 for e in r.edges]))))
        per = U.length(tier, r.perimeter, [e.sigma_fit for e in r.edges])
        rooms.append(Room(id=rid, name=room_name(r, k), polygon=[tuple(map(float, v)) for v in r.polygon],
                          floor_area=area, perimeter=per, ceiling_height=ceil, walls=walls, opening_ids=[],
                          floor_surface_id=f"{rid}-FLOOR", ceiling_surface_id=f"{rid}-CEIL"))
    lbl2id = {r.label: f"R{k}" for k, r in enumerate(pg.rooms, 1)}
    for k, o in enumerate(pg.openings, 1):
        oid = f"O{k}"
        rids = [lbl2id[int(l)] for l in o.rooms if int(l) in lbl2id]
        wids = [wall_id[(int(l), j)] for l, j in o.edge_refs if (int(l), j) in wall_id]
        w = U.length(tier, o.width, (o.width_sigma,), extra_rel=extra_rel)
        h = U.height(tier, o.height, (0.01,)) if o.height is not None else None
        sill = U.height(tier, o.sill, (0.02,)) if o.sill is not None else None
        openings.append(Opening(id=oid, kind=o.kind, room_ids=rids, wall_ids=wids, center=o.center, width=w,
                                height=h, sill_height=sill, detection_confidence=round(o.confidence, 2)))
        for rid in rids:
            next(r for r in rooms if r.id == rid).opening_ids.append(oid)
        if len(rids) == 2:
            adjacency.append((rids[0], rids[1], oid))
        # mark the shared walls
        if len(wids) == 2:
            for a, b in ((0, 1), (1, 0)):
                for r in rooms:
                    for wl in r.walls:
                        if wl.id == wids[a]:
                            wl.shared_with = rids[b]
    tot = sum(r.area for r in pg.rooms)
    per = sum(r.perimeter for r in pg.rooms)
    fp = U.area_from_dims(tier, tot, per)
    return PropertyScan(capture_id=capture_id, tier=tier, rooms=rooms, openings=openings, adjacency=adjacency,
                        footprint_area=fp, warnings=list(pg.warnings), error_budget=dict(U.BUDGETS[tier]))
