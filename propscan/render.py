"""Render a PropertyScan to a dimensioned floor plan (PNG + SVG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .schema import PropertyScan

FILL = ["#e8eef7", "#eef6e9", "#fbf1e3", "#f3e8f5", "#e6f4f4", "#f7ecec", "#eeeeee", "#f5f5dc"]


def _fmt(m):
    half = (m.ci95[1] - m.ci95[0]) / 2
    return f"{m.value:.2f}±{half * 100:.0f}cm" if half < 0.1 else f"{m.value:.2f}±{half:.2f}m"


def render(scan: PropertyScan, path_png: str, path_svg: str | None = None, debug_points=None):
    allp = np.array([p for r in scan.rooms for p in r.polygon]) if scan.rooms else np.zeros((1, 2))
    span = allp.max(0) - allp.min(0) + 1.5
    fig, ax = plt.subplots(figsize=(min(18, 2.0 + span[0] * 1.3), min(18, 2.0 + span[1] * 1.3)))
    if debug_points is not None:
        ax.scatter(debug_points[:, 0], debug_points[:, 1], s=0.05, c="#bbbbbb", zorder=0)
    for k, r in enumerate(scan.rooms):
        poly = np.array(r.polygon)
        ax.add_patch(Polygon(poly, closed=True, fc=FILL[k % len(FILL)], ec="none", zorder=1))
        for w in r.walls:
            (x0, z0), (x1, z1) = w.start, w.end
            ax.plot([x0, x1], [z0, z1], color="#222" if w.length.observed else "#999",
                    lw=3.2, solid_capstyle="projecting", zorder=3)
            if w.length.value >= 0.35:
                mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
                c = poly.mean(0)
                d = np.array([mx - c[0], mz - c[1]])
                n = np.array([-(z1 - z0), x1 - x0])
                n = n / (np.linalg.norm(n) + 1e-9)
                if np.dot(n, d) > 0:
                    n = -n
                ax.text(mx + n[0] * 0.18, mz + n[1] * 0.18, _fmt(w.length), fontsize=6.5, ha="center",
                        va="center", rotation=0 if abs(x1 - x0) > abs(z1 - z0) else 90, zorder=6, color="#333")
        c = poly.mean(0)
        label = f"{r.name}\n{r.floor_area.value:.1f} m² (±{(r.floor_area.ci95[1] - r.floor_area.ci95[0]) / 2:.1f})\n" \
                f"ceil {_fmt(r.ceiling_height)}{'' if r.ceiling_height.observed else ' (est.)'}"
        ax.text(c[0], c[1], label, fontsize=8, ha="center", va="center", zorder=7,
                bbox=dict(fc="white", ec="#888", alpha=0.85, boxstyle="round,pad=0.25"))
    for o in scan.openings:
        cx, cz = o.center
        w = o.width.value
        horiz = _opening_axis(scan, o) == "H"
        dx, dz = (w / 2, 0) if horiz else (0, w / 2)
        col = {"door": "#c0392b", "passage": "#2980b9", "window": "#16a085"}[o.kind]
        ax.plot([cx - dx, cx + dx], [cz - dz, cz + dz], color="white", lw=4.5, zorder=4)
        ax.plot([cx - dx, cx + dx], [cz - dz, cz + dz], color=col, lw=2.0, zorder=5,
                ls="-" if o.kind != "window" else (0, (2, 1)))
        ax.text(cx, cz, f"{o.kind[0].upper()} {w:.2f}", fontsize=6, color=col, ha="center", va="bottom", zorder=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fa = scan.footprint_area
    ax.set_title(f"{scan.capture_id} — tier: {scan.tier} — {len(scan.rooms)} rooms — footprint "
                 f"{fa.value:.1f} m² [{fa.ci95[0]:.1f}, {fa.ci95[1]:.1f}]", fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    if path_svg:
        fig.savefig(path_svg)
    plt.close(fig)


def _opening_axis(scan, o):
    for r in scan.rooms:
        for w in r.walls:
            if w.id in o.wall_ids:
                return "H" if abs(w.end[0] - w.start[0]) >= abs(w.end[1] - w.start[1]) else "V"
    return "H"
