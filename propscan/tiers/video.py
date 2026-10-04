"""Video tier: handheld clip only (no depth, no poses, no IMU).

  keyframes (~2 fps)  -> monocular metric depth (Depth Anything V2 Metric-Indoor)
  consecutive SIFT matches + PnP on the previous frame's scaled depth -> relative pose
  per-frame depth scale chained through matched points (monocular depth is shape-accurate
  (~3% after scale) but its per-frame metric scale wanders 0.75-1.9x, so scale is propagated
  geometrically and only the *average* model scale anchors the metric)
  -> FrameSet -> the same geometry engine as LiDAR (drift pose graph included).

Intrinsics: from --fov / EXIF-less prior (iPhone 1x main camera, ARKit/Camera 4:3 video
hFOV ~ 62 deg); the prior's uncertainty is carried in the error budget.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..geometry.frames import FrameSet
from ..models import depth as mono

DEPTH_HW = (192, 256)
HFOV_PRIOR_DEG = 61.9


def _intrinsics(w, h, hfov_deg=HFOV_PRIOR_DEG):
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def read_keyframes(path, fps_target=4.0, max_frames=450, work_w=640, flow_px=0.06, min_gap_s=0.0):
    """Adaptive keyframes: track corners with LK optical flow through *every* frame and cut a
    new keyframe when the median flow since the last keyframe exceeds `flow_px` x width or too
    few corners survive. Fast rotations therefore get dense keyframes (tracking never loses
    overlap) and slow pans get sparse ones (the depth model only runs on keyframes)."""
    vc = cv2.VideoCapture(str(path))
    fps = vc.get(cv2.CAP_PROP_FPS) or 30.0
    min_gap = max(1, int(min_gap_s * fps))
    max_gap = max(min_gap + 1, int(round(fps / max(fps_target / 4, 0.5))))
    frames, ids, tracks = [], [], []
    prev_small, pts, last_k = None, None, -10 ** 9
    up = work_w / 320.0
    i = 0
    while True:
        ok, f = vc.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(f, (320, int(round(f.shape[0] * 320 / f.shape[1])))), cv2.COLOR_BGR2GRAY)
        cut = not frames
        if not cut and pts is not None and len(pts):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_small, small, pts, None, winSize=(21, 21), maxLevel=3)
            st = st.ravel().astype(bool)
            keep = st.mean() if len(st) else 0
            pts = nxt[st]
            base = base[st]
            flow = np.median(np.linalg.norm(pts - base, axis=-1)) if len(pts) else 1e9
            gap = i - last_k
            cut = gap >= min_gap and (flow > flow_px * 320 or keep < 0.7 or len(pts) < 40 or gap >= max_gap)
        if cut:
            if frames and pts is not None and len(pts):
                tracks.append((base.reshape(-1, 2) * up, pts.reshape(-1, 2) * up))
            elif frames:
                tracks.append((np.zeros((0, 2)), np.zeros((0, 2))))
            h, w = f.shape[:2]
            frames.append(cv2.resize(f, (work_w, int(round(h * work_w / w))), interpolation=cv2.INTER_AREA))
            ids.append(i)
            last_k = i
            pts = cv2.goodFeaturesToTrack(small, 400, 0.01, 7)
            pts = pts if pts is not None else np.zeros((0, 1, 2), np.float32)
            base = pts.copy()
        prev_small = small
        i += 1
    tracks = [None] + tracks   # tracks[j]: (points in keyframe j-1, same points in keyframe j)
    if len(frames) > max_frames:
        sel = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[k] for k in sel]
        ids = [ids[k] for k in sel]
        tracks = [None] * len(sel)  # thinned: consecutive tracks no longer valid, SIFT only
    return frames, ids, fps, tracks


def _sift():
    return cv2.SIFT_create(nfeatures=2500)


def _match(d1, d2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(d1, d2, k=2)
    return [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]


def estimate_poses(frames, depths, K, Kd, tracks=None):
    """Chain PnP between consecutive keyframes using scaled monocular depth."""
    sift = _sift()
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    feats = [sift.detectAndCompute(g, None) for g in gray]
    n = len(frames)
    poses = np.zeros((n, 4, 4))
    poses[0] = np.eye(4)
    scale = np.ones(n)
    ok_flags = np.ones(n, bool)
    sx = Kd[0, 0] / K[0, 0]
    for j in range(1, n):
        best = None
        cands = []
        if tracks is not None and tracks[j] is not None and len(tracks[j][0]) >= 25:
            cands.append((j - 1, np.float32(tracks[j][0]), np.float32(tracks[j][1])))
        for i in (j - 1, j - 2, j - 3):
            if i < 0:
                continue
            k1, d1 = feats[i]
            k2, d2 = feats[j]
            if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
                continue
            m = _match(d1, d2)
            if len(m) < 25:
                continue
            cands.append((i, np.float32([k1[x.queryIdx].pt for x in m]), np.float32([k2[x.trainIdx].pt for x in m])))
        for i, p1, p2 in cands:
            # depth of frame i at its keypoints (depth map is at Kd resolution)
            u = np.clip((p1[:, 0] * sx).astype(int), 0, DEPTH_HW[1] - 1)
            v = np.clip((p1[:, 1] * sx).astype(int), 0, DEPTH_HW[0] - 1)
            z = depths[i][v, u] * scale[i]
            good = (z > 0.2) & (z < 6)
            if good.sum() < 20:
                continue
            X = np.c_[(p1[good, 0] - K[0, 2]) * z[good] / K[0, 0], (p1[good, 1] - K[1, 2]) * z[good] / K[1, 1], z[good]]
            okp, rvec, tvec, inl = cv2.solvePnPRansac(X, p2[good], K, None, iterationsCount=200,
                                                       reprojectionError=3.0, confidence=0.999,
                                                       flags=cv2.SOLVEPNP_EPNP)
            if not okp or inl is None or len(inl) < 15:
                continue
            inl = inl[:, 0]
            rvec, tvec = cv2.solvePnPRefineLM(X[inl], p2[good][inl], K, None, rvec, tvec)
            R, _ = cv2.Rodrigues(rvec)
            # chain scale: depth of the inlier points seen from j vs frame j's raw prediction
            Xj = X[inl] @ R.T + tvec.ravel()
            uj = np.clip((p2[good][inl, 0] * sx).astype(int), 0, DEPTH_HW[1] - 1)
            vj = np.clip((p2[good][inl, 1] * sx).astype(int), 0, DEPTH_HW[0] - 1)
            raw = depths[j][vj, uj]
            r = Xj[:, 2] / np.maximum(raw, 1e-3)
            r = r[(raw > 0.2) & (Xj[:, 2] > 0.2)]
            if len(r) < 10:
                continue
            cand = (len(inl), i, R, tvec.ravel(), float(np.median(r)))
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None and tracks is not None and tracks[j] is not None and len(tracks[j][0]) >= 8:
            best = _rotation_only(np.float32(tracks[j][0]), np.float32(tracks[j][1]), K, depths, scale, j, sx)
        if best is None:
            # lost tracking: constant-velocity prediction (flagged; drift graph may re-attach it)
            if j >= 2 and ok_flags[j - 1]:
                poses[j] = poses[j - 1] @ np.linalg.inv(poses[j - 2]) @ poses[j - 1]
            else:
                poses[j] = poses[j - 1]
            scale[j] = scale[j - 1]
            ok_flags[j] = False
            continue
        _, i, R, t, s = best
        T_j_i = np.eye(4)
        T_j_i[:3, :3] = R
        T_j_i[:3, 3] = t
        poses[j] = poses[i] @ np.linalg.inv(T_j_i)   # camera j -> world
        # damp scale chaining towards the model's own scale to stop a random walk
        scale[j] = np.exp(0.85 * np.log(s) + 0.15 * np.log(scale[i]))
    return poses, scale, ok_flags


def _rotation_only(p1, p2, K, depths, scale, j, sx, iters=200, tol_deg=1.0):
    """Fast pans: fit a pure rotation to optical-flow bearings (RANSAC Kabsch); translation over a
    1-2 frame gap is negligible. Keeps the scale chain alive via the rotated points' depths."""
    Ki = np.linalg.inv(K)
    b1 = np.c_[p1, np.ones(len(p1))] @ Ki.T
    b2 = np.c_[p2, np.ones(len(p2))] @ Ki.T
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
    rng = np.random.default_rng(j)
    best_inl, best_R = None, None
    cos_tol = np.cos(np.radians(tol_deg))
    for _ in range(iters):
        s = rng.choice(len(b1), 3, replace=False)
        U, _, Vt = np.linalg.svd(b2[s].T @ b1[s])
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        inl = (b1 @ R.T * b2).sum(1) > cos_tol
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_R = inl, R
    if best_inl is None or best_inl.sum() < max(8, 0.4 * len(b1)):
        return None
    U, _, Vt = np.linalg.svd(b2[best_inl].T @ b1[best_inl])
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    i = j - 1
    u = np.clip((p1[best_inl, 0] * sx).astype(int), 0, DEPTH_HW[1] - 1)
    v = np.clip((p1[best_inl, 1] * sx).astype(int), 0, DEPTH_HW[0] - 1)
    z = depths[i][v, u] * scale[i]
    X = b1[best_inl] / b1[best_inl, 2:3] * z[:, None]
    Xj = X @ R.T
    uj = np.clip((p2[best_inl, 0] * sx).astype(int), 0, DEPTH_HW[1] - 1)
    vj = np.clip((p2[best_inl, 1] * sx).astype(int), 0, DEPTH_HW[0] - 1)
    raw = depths[j][vj, uj]
    r = Xj[:, 2] / np.maximum(raw, 1e-3)
    r = r[(raw > 0.2) & (Xj[:, 2] > 0.2)]
    sc = float(np.median(r)) if len(r) >= 5 else float(scale[i])
    return (int(best_inl.sum()), i, R, np.zeros(3), sc)


def load(path, timings=None, fps_target=4.0, hfov_deg=HFOV_PRIOR_DEG):
    timings = timings if timings is not None else {}
    t = time.time()
    frames, ids, fps, tracks = read_keyframes(path, fps_target)
    timings["decode"] = time.time() - t
    h, w = frames[0].shape[:2]
    K = _intrinsics(w, h, hfov_deg)
    Kd = K.copy()
    Kd[0] *= DEPTH_HW[1] / w
    Kd[1] *= DEPTH_HW[0] / h
    t = time.time()
    depths = [mono.predict(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), DEPTH_HW) for f in frames]
    timings["depth_model"] = time.time() - t
    t = time.time()
    poses, scale, ok = estimate_poses(frames, depths, K, Kd, tracks)
    # metric anchor: the model's scale is unbiased on average, so renormalise the chained
    # scales to geometric mean 1 and scale the trajectory by the same factor
    g = float(np.exp(np.mean(np.log(scale[ok])))) if ok.any() else 1.0
    scale = scale / g
    poses[:, :3, 3] /= g
    timings["tracking"] = time.time() - t
    sd = [(d * s).astype(np.float32) for d, s in zip(depths, scale)]
    masks = [_trust_mask(d) for d in sd]
    fs = FrameSet(sd, masks, [Kd] * len(sd), poses, ids, max_depth=4.0,
                  meta=dict(tier="video", source=str(path), keyframes=len(ids), tracked=int(ok.sum()),
                            hfov_prior_deg=hfov_deg, scale_spread=float(np.std(np.log(scale)))))
    fs.masks_loose = masks
    fs.rgb = frames
    return fs, None


def _trust_mask(d):
    """Monocular depth is least reliable at depth edges and the image border."""
    gx = np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))
    m = (gx + gy) < 0.25 * d
    m[:6] = m[-6:] = False
    m[:, :6] = m[:, -6:] = False
    return m & (d > 0.2)
