"""Diagnostic: video-tier trajectory vs ARKit trajectory (similarity-aligned ATE).
ARKit poses are used ONLY here, as a reference; the video tier never reads them."""
import sys, numpy as np
from propscan.tiers import video
from propscan.io.stray import load_stray

def umeyama(A, B):
    mA, mB = A.mean(0), B.mean(0)
    a, b = A - mA, B - mB
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, mB - s * R @ mA

def main(zip_path, fps=2.0):
    cap = load_stray(zip_path)
    fs, _ = video.load(cap.video_path(), fps_target=fps)
    ref = cap.poses[np.array(fs.frame_ids), :3, 3]
    est = fs.poses[:, :3, 3]
    s, R, t = umeyama(est, ref)
    err = np.linalg.norm((s * est @ R.T + t) - ref, axis=1)
    pl = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
    print(f"keyframes {fs.n} tracked {fs.meta['tracked']} scale(ref/est) {s:.3f} "
          f"ATE rmse {np.sqrt((err**2).mean()):.3f} m over path {pl:.1f} m, max {err.max():.3f}")
    return s

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
