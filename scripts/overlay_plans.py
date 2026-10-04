"""Diagnostic: overlay the plans of two captures (after plan registration) on wall points."""
import sys
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import fixloop_eval as F
from propscan.compare import register
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def main(mode, out):
    A, pgA, PA, NA = F.plan("L_ceil", mode); B, pgB, PB, NB = F.plan("L_floor", mode)
    R, t, iou = register(A, B)
    fig, ax = plt.subplots(figsize=(11, 11))
    w = (np.abs(NA[:, 1]) < 0.3) & (np.abs(PA[:, 1] - (pgA.floor_y + 1.0)) < 0.5)
    q = PA[w][::5][:, [0, 2]]
    ax.scatter(q[:, 0], q[:, 1], s=0.1, c="gray")
    for S2, c2, R2, t2 in ((A, "r", np.eye(2), np.zeros(2)), (B, "b", R, t)):
        for r in S2.rooms:
            V = np.array(r.polygon) @ R2.T + t2; V = np.r_[V, V[:1]]; ax.plot(V[:, 0], V[:, 1], c2, lw=1)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_title(f"{mode}: red=with_ceiling blue=floor_only")
    fig.savefig(out, dpi=90)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
