"""Build the photo-tier benchmark set from a Stray capture, following docs/CAPTURE_PROTOCOL.md.

We have no separate still-photo capture of the sample property, so stills are pulled from the
capture's own RGB video (same iPhone camera; 1920x1440). The LiDAR-tier plan is used ONLY to
choose which frames a person following the protocol would have taken:
  - up to 6 room photos per room, spread across viewing directions, from inside the room
  - for every doorway, one `door_to_<other>.jpg` from inside each room, centred on the door
The photo pipeline receives only the JPEGs (with the camera's 35 mm-equivalent focal in EXIF,
as an iPhone writes it). No depth, no poses.

Also writes reference.json: the LiDAR-tier plan used as the reference for tier-vs-tier scoring.

    python scripts/make_photo_benchmark.py "sample data/single_scan_with_ceiling.zip" benchmark/photos/with_ceiling
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propscan.assemble import assemble  # noqa: E402
from propscan.pipeline import geometry_from_frameset  # noqa: E402
from propscan.tiers import lidar  # noqa: E402


def main(src, dst, per_room=8):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    fs, cap = lidar.load(src)
    pg, P, N, _ = geometry_from_frameset(fs, "lidar", drift=True)
    scan = assemble(pg, "lidar", Path(src).stem)
    poses = fs.poses
    cam = poses[:, :3, 3][:, [0, 2]]
    fwd = poses[:, :3, 2]
    yaw = np.degrees(np.arctan2(fwd[:, 2], fwd[:, 0]))
    level = np.abs(fwd[:, 1]) < 0.45            # not pointing at floor/ceiling
    med_depth = np.array([np.median(d[d > 0]) if (d > 0).any() else 0 for d in fs.depths])
    want = {}                                    # source frame id -> output path
    names = {r.id: f"room{k}" for k, r in enumerate(scan.rooms, 1)}
    polys = {r.id: Polygon(r.polygon) for r in scan.rooms}
    for r in scan.rooms:
        inside = np.array([polys[r.id].buffer(-0.15).contains(Point(*c)) for c in cam]) & level & (med_depth > 1.2)
        idx = np.flatnonzero(inside)
        if len(idx) == 0:
            continue
        # protocol: turn on the spot / walk the room, one photo every ~40 deg of turn or 1 m of
        # walking, each overlapping the previous by about half -> take the capture's own sweep
        chosen = [idx[0]]
        for k in idx[1:]:
            c = chosen[-1]
            turned = abs((yaw[k] - yaw[c] + 180) % 360 - 180)
            if turned >= 35 or np.linalg.norm(cam[k] - cam[c]) >= 1.0:
                chosen.append(k)
        if len(chosen) > per_room:
            chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, per_room).round().astype(int)]
        (dst / names[r.id]).mkdir(exist_ok=True)
        for n, k in enumerate(chosen, 1):
            want[fs.frame_ids[k]] = dst / names[r.id] / f"IMG_{n:04d}.jpg"
    for o in scan.openings:
        if len(o.room_ids) != 2:
            continue
        c = np.array(o.center)
        for a, b in (o.room_ids, o.room_ids[::-1]):
            inside = np.array([polys[a].contains(Point(*p)) for p in cam]) & level
            d = c - cam
            dist = np.linalg.norm(d, axis=1)
            ang = np.degrees(np.arccos(np.clip((d / np.maximum(dist[:, None], 1e-6) * fwd[:, [0, 2]]).sum(1)
                                               / np.maximum(np.linalg.norm(fwd[:, [0, 2]], axis=1), 1e-6), -1, 1)))
            ok = inside & (dist > 0.8) & (dist < 3.5) & (ang < 15)
            if ok.any():
                k = np.flatnonzero(ok)[np.argmin(ang[ok])]
                want[fs.frame_ids[k]] = dst / names[a] / f"door_to_{names[b]}.jpg"
    # extract frames
    vc = cv2.VideoCapture(cap.video_path())
    f35 = int(round(float(np.median(cap.K_rgb[:, 0, 0])) * 43.27 / np.hypot(1920, 1440)))
    i = 0
    left = dict(want)
    while left:
        ok, f = vc.read()
        if not ok:
            break
        if i in left:
            im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            ex = Image.Exif()
            ex[0xA405] = f35                     # FocalLengthIn35mmFilm
            ex[0x010F] = "Apple"
            im.save(left.pop(i), quality=92, exif=ex.tobytes())
        i += 1
    ref = dict(source=str(src), note="LiDAR-tier plan of the same capture; reference for tier-vs-tier scoring",
               rooms={names[r.id]: dict(id=r.id, polygon=r.polygon, floor_area=r.floor_area.value,
                                        ceiling_height=r.ceiling_height.value,
                                        walls=[w.length.value for w in r.walls]) for r in scan.rooms},
               adjacency=[(names[a], names[b]) for a, b, _ in scan.adjacency],
               footprint_area=scan.footprint_area.value)
    (dst / "reference.json").write_text(json.dumps(ref, indent=1))
    print(f"wrote {len(want)} photos for {len(names)} rooms to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
