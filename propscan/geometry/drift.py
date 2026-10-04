"""Drift correction: fragment pose graph with loop closures (gravity-preserving).

ARKit/VIO poses are locally excellent but accumulate drift over a multi-room walk;
revisited walls show up doubled ("ghost walls"). We:
  1. cut the walk into fragments of consecutive frames and fuse each into a local cloud
  2. register every spatially-overlapping fragment pair with point-to-plane ICP
     (consecutive pairs = odometry edges, others = loop closures)
  3. optimise a pose graph over per-fragment corrections (Open3D, LM + line-process pruning)
  4. project each correction to yaw + translation (ARKit gravity is trusted) and apply it,
     interpolating between fragment centres so the trajectory stays continuous.
`enabled=False` returns poses unchanged; this is the ablation switch.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from .frames import FrameSet, backproject, voxel_down


def _frag_cloud(fs, idx, voxel):
    pts, nrm = [], []
    for i in idx:
        P, N = backproject(fs.depths[i], fs.masks[i], fs.Ks[i], fs.poses[i], min(fs.max_depth, 3.5))
        if len(P) > 3000:
            s = np.random.default_rng(i).choice(len(P), 3000, replace=False)
            P, N = P[s], N[s]
        pts.append(P)
        nrm.append(N)
    P, N, _ = voxel_down(np.concatenate(pts), np.concatenate(nrm), voxel)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    return pc


def _yaw_only(T):
    R = T[:3, :3]
    yaw = np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    out[:3, 3] = T[:3, 3]
    out[1, 3] = 0.0 if abs(T[1, 3]) < 0.05 else T[1, 3]  # vertical drift is tiny with ARKit gravity
    return out


def correct_drift(fs: FrameSet, enabled=True, frag_len=None, voxel=0.03, max_pair_dist=3.0, max_loop_per_frag=4):
    report = dict(method="fragment pose graph + ICP loop closures, yaw/translation only",
                  enabled=enabled, loop_closures=0, fragments=0, mean_correction_m=0.0, max_correction_m=0.0)
    if not enabled or fs.n < 40:
        return fs.poses.copy(), report
    if frag_len is None:
        frag_len = max(15, fs.n // 60)
    starts = list(range(0, fs.n, frag_len))
    frags = [list(range(s, min(s + frag_len, fs.n))) for s in starts]
    clouds = [_frag_cloud(fs, f, voxel) for f in frags]
    cents = np.array([np.asarray(c.points).mean(0) if len(c.points) else np.zeros(3) for c in clouds])
    nf = len(frags)
    report["fragments"] = nf

    reg = o3d.pipelines.registration
    pg = reg.PoseGraph()
    for _ in range(nf):
        pg.nodes.append(reg.PoseGraphNode(np.eye(4)))
    crit = reg.ICPConvergenceCriteria(max_iteration=40)
    n_loop = 0
    for i in range(nf):
        if len(clouds[i].points) < 200:
            continue
        # consecutive edge + the nearest non-adjacent fragments (bounded ICP budget)
        dist = np.linalg.norm(cents - cents[i], axis=1)
        later = [j for j in np.argsort(dist) if j > i + 1 and dist[j] <= max_pair_dist][:max_loop_per_frag]
        for j in ([i + 1] if i + 1 < nf else []) + later:
            if len(clouds[j].points) < 200:
                continue
            consecutive = j == i + 1
            # coarse-to-fine ICP; fragments are already in a common (drifted) world frame
            T = np.eye(4)
            ok = True
            for d in (0.15, 0.06, 0.03):
                r = reg.registration_icp(clouds[j], clouds[i], d, T,
                                         reg.TransformationEstimationPointToPlane(), crit)
                T = r.transformation
            if r.fitness < (0.25 if consecutive else 0.35) or r.inlier_rmse > 0.02:
                ok = False
            if np.linalg.norm(T[:3, 3]) > 0.5:
                ok = False
            if not ok and not consecutive:
                continue
            if not ok:
                T = np.eye(4)
            T = _yaw_only(T)
            info = reg.get_information_matrix_from_point_clouds(clouds[j], clouds[i], 0.03, T)
            # edge: node_j correction maps j's cloud onto i's: X_i = C_i^-1 C_j X_j  ~ T
            pg.edges.append(reg.PoseGraphEdge(j, i, T, info, uncertain=not consecutive))
            if not consecutive:
                n_loop += 1
    report["loop_closures"] = n_loop
    opt = reg.GlobalOptimizationOption(max_correspondence_distance=0.03, edge_prune_threshold=0.25,
                                       preference_loop_closure=1.0, reference_node=0)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(), opt)
    C = np.array([_yaw_only(np.asarray(n.pose)) for n in pg.nodes])

    # interpolate corrections between fragment centres (translation linear, yaw linear)
    centres = np.array([np.mean(f) for f in frags])
    yaws = np.unwrap([np.arctan2(c[0, 2], c[0, 0]) for c in C])
    tr = C[:, :3, 3]
    out = fs.poses.copy()
    mags = []
    for k in range(fs.n):
        yaw = np.interp(k, centres, yaws)
        t = np.array([np.interp(k, centres, tr[:, a]) for a in range(3)])
        c, s = np.cos(yaw), np.sin(yaw)
        Ck = np.eye(4)
        Ck[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        Ck[:3, 3] = t
        out[k] = Ck @ fs.poses[k]
        mags.append(np.linalg.norm(out[k][:3, 3] - fs.poses[k][:3, 3]))
    report["mean_correction_m"] = float(np.mean(mags))
    report["max_correction_m"] = float(np.max(mags))
    return out, report
