"""CLI.

    python -m propscan run <capture> [-o out/] [--tier auto|lidar|video|photos] [--no-drift] [--no-damage]
    python -m propscan schema

<capture> is a Stray Scanner .zip/folder (LiDAR), a .mp4/.mov (video), or a folder of
per-room photo folders (photos).
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="propscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("capture")
    r.add_argument("-o", "--out", default="out")
    r.add_argument("--tier", default="auto", choices=["auto", "lidar", "video", "photos"])
    r.add_argument("--no-drift", action="store_true", help="ablation: use poses as-is")
    r.add_argument("--no-damage", action="store_true")
    r.add_argument("--id", default=None, help="capture id (default: file stem)")
    r.add_argument("--debug-points", action="store_true")
    sub.add_parser("schema")
    a = ap.parse_args()
    if a.cmd == "schema":
        from .schema import PropertyScan
        p = Path(__file__).resolve().parent.parent / "schema" / "property_scan.schema.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(PropertyScan.model_json_schema(), indent=2))
        print(p)
        return
    from .pipeline import run
    scan = run(a.capture, a.out, a.tier, drift=not a.no_drift, damage=not a.no_damage, capture_id=a.id,
               debug_points=a.debug_points)
    print(f"{scan.capture_id}: tier={scan.tier} rooms={len(scan.rooms)} openings={len(scan.openings)} "
          f"footprint={scan.footprint_area.value:.2f} m2 {list(scan.footprint_area.ci95)} "
          f"time={scan.timings_s.get('total')}s -> {a.out}")


if __name__ == "__main__":
    main()
