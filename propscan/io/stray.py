"""Loader for Stray Scanner exports (LiDAR tier).

Layout of one capture directory:
    rgb.mp4               1920x1440 colour video, one frame per odometry row
    depth/NNNNNN.png      256x192 uint16 depth in millimetres
    confidence/NNNNNN.png 256x192 uint8 ARKit confidence (0 low, 1 medium, 2 high)
    odometry.csv          per-frame camera->world pose (ARKit convention) + intrinsics
    camera_matrix.csv     3x3 intrinsics at RGB resolution
    imu.csv               accelerometer + gyro (unused by geometry; logged for the report)

The loader accepts either an extracted directory or the .zip exported by the app.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Stray's odometry pose composes directly with an OpenCV pinhole back-projection
# (x right, y down, z forward). Verified empirically: with this convention the fused
# floor collapses to one 1 cm bin (6% of all points); with the ARKit->CV axis flip it
# smears over 2 m. World frame is ARKit's: y up (gravity-aligned), metres.

RGB_W, RGB_H = 1920, 1440


def quat_to_rot(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


@dataclass
class StrayCapture:
    root: str
    timestamps: np.ndarray      # (N,)
    frame_ids: np.ndarray       # (N,) int
    poses: np.ndarray           # (N,4,4) camera(OpenCV)->world, world is ARKit (y up, metres)
    K_rgb: np.ndarray           # (N,3,3) per-frame intrinsics at RGB resolution
    _zip: zipfile.ZipFile | None = None
    _prefix: str = ""

    @property
    def n(self) -> int:
        return len(self.frame_ids)

    def _read(self, rel: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(self._prefix + rel)
        return (Path(self.root) / rel).read_bytes()

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres, (192,256) float32; 0 where invalid."""
        raw = np.array(Image.open(io.BytesIO(self._read(f"depth/{self.frame_ids[i]:06d}.png"))))
        return raw.astype(np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        return np.array(Image.open(io.BytesIO(self._read(f"confidence/{self.frame_ids[i]:06d}.png"))))

    def K_depth(self, i: int, shape=(192, 256)) -> np.ndarray:
        K = self.K_rgb[i].copy()
        K[0] *= shape[1] / RGB_W
        K[1] *= shape[0] / RGB_H
        return K

    def video_path(self) -> str:
        if self._zip is None:
            return str(Path(self.root) / "rgb.mp4")
        # opencv cannot read from a zip member; extract once next to the zip
        out = Path(self.root).with_suffix("") / "rgb.mp4"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self._read("rgb.mp4"))
        return str(out)


def load_stray(path: str | Path) -> StrayCapture:
    path = Path(path)
    zf, prefix = None, ""
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        odo_name = next(n for n in zf.namelist() if n.endswith("odometry.csv"))
        prefix = odo_name[: -len("odometry.csv")]
        text = zf.read(odo_name).decode()
    else:
        if not (path / "odometry.csv").exists():
            # allow pointing at the parent of the scan-id folder
            sub = [p for p in path.iterdir() if (p / "odometry.csv").exists()]
            if not sub:
                raise FileNotFoundError(f"no odometry.csv under {path}")
            path = sub[0]
        text = (path / "odometry.csv").read_text()

    rows = [r.split(",") for r in text.strip().splitlines()[1:]]
    ts = np.array([float(r[0]) for r in rows])
    fid = np.array([int(r[1]) for r in rows])
    vals = np.array([[float(x) for x in r[2:13]] for r in rows])
    n = len(rows)
    poses = np.zeros((n, 4, 4))
    K = np.zeros((n, 3, 3))
    for i in range(n):
        x, y, z, qx, qy, qz, qw, fx, fy, cx, cy = vals[i]
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
        T[:3, 3] = (x, y, z)
        poses[i] = T
        K[i] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return StrayCapture(str(path), ts, fid, poses, K, zf, prefix)


def is_stray(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            return any(n.endswith("odometry.csv") for n in zf.namelist())
    return (path / "odometry.csv").exists() or any(
        (p / "odometry.csv").exists() for p in path.iterdir() if p.is_dir())
