"""Draw the system block diagrams -> docs/img/diagram_*.png (used by README and PROJECT_REPORT.ipynb).

    python scripts/make_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

C = dict(cap="#e8eef7", tier="#fbf1e3", core="#eef6e9", out="#f3e8f5", eval="#eeeeee", edge="#444444")


def box(ax, x, y, w, h, title, body="", fc="#ffffff", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=C["edge"], lw=1.2))
    if body:
        ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=fs, weight="bold")
        ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=fs - 2.2, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold")
    return (x, y, w, h)


def arrow(ax, a, b, label="", side="r", rad=0.0):
    """a, b are (x, y) points."""
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.2, color=C["edge"],
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx + (0.08 if side == "r" else -0.08), my, label, fontsize=7.5, color="#555",
                ha="left" if side == "r" else "right", va="center", style="italic")


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- 1. system overview
fig, ax = canvas(17, 8.6)
ax.text(8.5, 8.35, "PropertyScan: system overview", ha="center", fontsize=15, weight="bold")

cl = box(ax, 0.3, 5.6, 3.0, 1.9, "LiDAR capture", "Stray Scanner (iPhone Pro)\n.zip: depth, confidence,\nARKit poses, intrinsics", C["cap"])
cv = box(ax, 0.3, 3.4, 3.0, 1.9, "Video capture", "iPhone Camera app\n.mov / .mp4\nRGB only", C["cap"])
cp = box(ax, 0.3, 1.2, 3.0, 1.9, "Photo capture", "iPhone Camera app\n1 folder per room, 2-8 JPEGs\n+ door_to_<room>.jpg", C["cap"])
ax.text(1.8, 0.75, "docs/CAPTURE_PROTOCOL.md\n(one page, stock apps)", ha="center", fontsize=8, color="#555")

cli = box(ax, 3.9, 3.55, 1.7, 1.6, "CLI", "python -m\npropscan run\n<capture>\n(auto tier)", "#ffffff")
for b in (cl, cv, cp):
    arrow(ax, right(b), left(cli))

tl = box(ax, 6.2, 5.6, 3.2, 1.9, "LiDAR front-end", "high-confidence depth\nARKit poses + intrinsics\n~700 frames", C["tier"])
tv = box(ax, 6.2, 3.4, 3.2, 1.9, "Video front-end", "adaptive keyframes (optical flow)\nmonocular metric depth\nPnP tracking + scale chaining", C["tier"])
tp = box(ax, 6.2, 1.2, 3.2, 1.9, "Photo front-end", "monocular metric depth\nper-room registration\n(essential matrix)", C["tier"])
for b in (tl, tv, tp):
    arrow(ax, right(cli), left(b))

fs = box(ax, 10.0, 3.55, 2.0, 1.6, "FrameSet", "per frame:\ndepth, mask,\nK, camera pose", "#fff7cc")
for b in (tl, tv):
    arrow(ax, right(b), left(fs))
eng = box(ax, 12.6, 2.6, 2.1, 3.5, "Shared geometry engine",
          "drift pose graph\nfusion + normals\ngravity + Manhattan\nfloor / ceiling\nrooms (watershed)\nwalls (plane snap)\ndoors / windows\nmirror rejection\n95% intervals", C["core"])
arrow(ax, right(fs), left(eng))
st = box(ax, 10.0, 1.2, 2.0, 1.9, "Door-photo\nstitching", "", C["tier"], fs=9)
arrow(ax, right(tp), left(st))
arrow(ax, top(st), bottom(fs), "per-room\nFrameSets", side="r")

dmg = box(ax, 12.6, 0.4, 2.1, 1.7, "Damage stage", "regions + extent\nconcealed rules R1-R5\nscope line items", C["core"], fs=9)
arrow(ax, bottom(eng), top(dmg))

oj = box(ax, 15.2, 5.0, 1.6, 1.5, "JSON", "schema/\nproperty_scan\n.schema.json", C["out"], fs=9)
op = box(ax, 15.2, 3.0, 1.6, 1.5, "Floor plan", "PNG + SVG\ndimensioned", C["out"], fs=9)
ob = box(ax, 15.2, 0.6, 1.6, 1.8, "Benchmark", "repeatability\ncross-tier, drift\nablation, fix loop", C["eval"], fs=9)
arrow(ax, right(eng), left(oj))
arrow(ax, right(eng), left(op))
arrow(ax, right(dmg), left(ob))
fig.savefig(OUT / "diagram_system.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 2. geometry engine
fig, ax = canvas(17, 7.4)
ax.text(8.5, 7.15, "Shared geometry engine (propscan/geometry, all tiers)", ha="center", fontsize=15, weight="bold")
steps = [
    ("1. Drift correction", "~60 fragments\nICP loop closures\npose graph (Open3D)\nyaw + translation only"),
    ("2. Fusion", "back-project depth\nper-pixel normals\n2 cm voxel average"),
    ("3. Alignment", "gravity (ARKit or\nfrom normals)\nManhattan yaw from\nwall normals"),
    ("4. Levels", "floor / ceiling from\nup- / down-facing\nheight histograms"),
    ("5. Plan occupancy", "2.5 cm grid: walls,\nfloor + furniture,\ncamera path\nflood-fill interior"),
]
y1 = 4.75
prev = None
for i, (t, b) in enumerate(steps):
    bx = box(ax, 0.3 + i * 3.35, y1, 2.95, 1.9, t, b, C["core"])
    if prev:
        arrow(ax, right(prev), left(bx))
    prev = bx
steps2 = [
    ("6. Rooms", "distance-transform\nwatershed (doorway\ncuts), merge tiny\n+ open-plan regions"),
    ("7. Walls", "rectilinear outline\nsnap edge to covered\nwall plane (<= 1 m out,\nback-face barrier)\nrectangle snap"),
    ("8. Openings", "doors: camera path\ncrosses rooms ->\njamb-to-jamb width\nwindows: rays through\nwall, mirror test"),
    ("9. Intervals", "sigma = fit + surface\n+ scale x length\nper-tier budget\nobserved flag"),
    ("10. Assemble", "PropertyScan JSON\n+ damage stage\n+ render plan"),
]
y2 = 0.9
first2 = None
prev = None
for i, (t, b) in enumerate(steps2):
    bx = box(ax, 0.3 + i * 3.35, y2, 2.95, 2.3, t, b, C["core"] if i < 4 else C["out"])
    if prev:
        arrow(ax, right(prev), left(bx))
    if first2 is None:
        first2 = bx
    prev = bx
# elbow connector in the gap between the rows (never crosses a box)
x5 = 0.3 + 4 * 3.35 + 2.95 / 2
x6 = first2[0] + first2[2] / 2
ymid = (y1 + y2 + 2.3) / 2
ax.plot([x5, x5, x6], [y1, ymid, ymid], color=C["edge"], lw=1.2)
arrow(ax, (x6, ymid), (x6, y2 + 2.3))
fig.savefig(OUT / "diagram_engine.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 3. tier front-ends
fig, ax = canvas(17, 9.0)
ax.text(8.5, 8.75, "Tier front-ends: how each input becomes a FrameSet", ha="center", fontsize=15, weight="bold")
rows = [
    ("LiDAR", C["cap"], [("Stray .zip", "odometry.csv\ndepth/*.png\nconfidence/*.png"),
                         ("Pose convention", "raw Stray pose =\ncam->world (verified:\n1 cm floor bin)"),
                         ("Depth mask", "confidence 2 for\nfusion, >= 1 for\nwindow rays"),
                         ("Sub-sample", "~700 frames\n(ARKit ~46 fps)")]),
    ("Video", C["cap"], [("Keyframes", "LK optical flow on every\nframe; cut when view\nhas moved enough"),
                         ("Depth model", "Depth Anything V2\nMetric-Indoor (small)\ncached per image"),
                         ("Tracking", "PnP on LK tracks\nSIFT fallback\nrotation-only RANSAC"),
                         ("Scale", "chain scale frame to\nframe; anchor to the\nmodel's average")]),
    ("Photos", C["cap"], [("Room folder", "2-8 JPEGs\nEXIF 35 mm focal\n-> intrinsics"),
                          ("Depth model", "same model\nper photo"),
                          ("Registration", "SIFT pairs, essential\nmatrix, metric baseline\nfrom median depth ratio"),
                          ("Stitching", "door_to_<room>.jpg:\naxis hits door wall,\n90-deg rotation,\njoin across 12 cm wall")]),
]
for r, (name, col, stages) in enumerate(rows):
    y = 6.0 - r * 2.75
    lab = box(ax, 0.2, y, 1.6, 2.1, name, "", col, fs=12)
    prev = lab
    for i, (t, b) in enumerate(stages):
        bx = box(ax, 2.2 + i * 3.2, y, 2.85, 2.1, t, b, C["tier"])
        arrow(ax, right(prev), left(bx))
        prev = bx
    fsb = box(ax, 15.15, y + 0.45, 1.65, 1.2, "FrameSet", "", "#fff7cc", fs=10)
    arrow(ax, right(prev), left(fsb))
fig.savefig(OUT / "diagram_tiers.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 4. evaluation / fix loop
fig, ax = canvas(17, 5.2)
ax.text(8.5, 4.95, "Evaluation: no ground truth, so what we measure instead", ha="center", fontsize=15, weight="bold")
a = box(ax, 0.3, 1.6, 2.8, 2.4, "Raw captures", "3 Stray zips\n+ derived video\n+ derived photo\nfolders", C["cap"])
b = box(ax, 3.7, 1.6, 2.8, 2.4, "benchmark.py", "9 runs:\nLiDAR x3\nLiDAR no-drift x2\nvideo x3, photos x1", C["eval"])
c1 = box(ax, 7.1, 3.0, 3.2, 1.5, "Repeatability", "LiDAR capture vs capture\n(register, match walls)", C["core"], fs=9)
c2 = box(ax, 7.1, 1.25, 3.2, 1.5, "Cross-tier agreement", "video / photos vs LiDAR\n+ 95% CI coverage", C["core"], fs=9)
c3 = box(ax, 7.1, -0.5 + 0.55, 3.2, 1.0, "Drift ablation", "on vs off, ghost-wall area", C["core"], fs=9)
d = box(ax, 11.0, 1.6, 2.6, 2.4, "Reports", "results_main.json\nREPORT.md\nPROJECT_REPORT\n.ipynb", C["out"])
e = box(ax, 14.2, 1.6, 2.6, 2.4, "Fix loop", "worst gate\n-> hypothesis\n-> shipped fix\n-> before / after", "#fde2e2")
arrow(ax, right(a), left(b))
for cc in (c1, c2, c3):
    arrow(ax, right(b), left(cc))
    arrow(ax, right(cc), left(d))
arrow(ax, right(d), left(e))
ax.text(8.7, 4.65, "synthetic damage injection validates detection extent separately", ha="center", fontsize=8.5, color="#555")
fig.savefig(OUT / "diagram_evaluation.png", dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("diagram_*.png")))
