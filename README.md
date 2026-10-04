# PropertyScan: phone capture to a dimensioned whole-property floor plan

One command per capture, three input tiers, one output contract:

```
python -m propscan run <capture> -o out/
```

| `<capture>` | Tier | Example |
|---|---|---|
| Stray Scanner `.zip` (or extracted folder) | **LiDAR**: depth, ARKit poses, intrinsics | `sample data/single_room.zip` |
| `.mov` / `.mp4` walkthrough | **Video**: RGB only | `walk.mov` |
| Folder of per-room photo folders | **Photos**: 2–8 stills per room, no depth or poses | `my_flat/` |

Outputs `out/<id>.json` (schema: [schema/property_scan.schema.json](schema/property_scan.schema.json)) and `out/<id>_plan.png` / `.svg`:
- per-room walls, ceiling height, floor area and openings, all dimensioned;
- the stitched multi-room plan with adjacency;
- damage regions with class and metric extent, concealed-damage flags with the rule that fired, and scope line items keyed to surface IDs;
- a 95% interval on every measurement.

How to capture: **[docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md)** (one page, stock apps). Which tier runs on which phone: [docs/DEVICE_MATRIX.md](docs/DEVICE_MATRIX.md).

## From a clean machine to a first result (≈10 min)
Requires Python 3.10–3.12 (tested on 3.11, Windows 11; pure-pip, no compiler needed).
```bash
git clone <repo> propscan && cd propscan
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                   # torch CPU wheel is the big one
python scripts/fetch_weights.py                   # ~100 MB, Depth Anything V2 Metric-Indoor Small
python -m propscan run "sample data/single_room.zip" -o out/
```
Put the provided sample captures in `sample data/` (they are not in git; see [data/README.md](data/README.md)).

Timing on a 12-thread laptop CPU with no GPU:

| Tier | Time per capture |
|---|---|
| LiDAR | 1.5–4 min per capture (most of it drift correction) |
| Video | about 1.3 s per keyframe for the depth model; cached after the first run |
| Photos | about 2 s per photo |

A CUDA build of torch is used automatically if present.

## Reproduce every reported number
```bash
python scripts/benchmark.py                 # runs all captures and tiers, then scores -> benchmark/results_main.json
python scripts/make_photo_benchmark.py "sample data/single_scan_with_ceiling.zip" data/photos/with_ceiling
python scripts/fixloop_eval.py --evidence   # fix-loop before/after (see docs/FIX_LOOP.md)
python scripts/synthetic_damage_test.py "sample data/single_room.zip"
```
Monocular-depth outputs are cached in `.cache/depth` (keyed by image content and model) and replay deterministically. Delete `.cache/` to exercise the live path.

## Layout
```
propscan/io/stray.py          Stray Scanner loader (pose convention verified on data)
propscan/tiers/{lidar,video,photos}.py   tier front-ends -> FrameSet (depth + pose per frame)
propscan/geometry/frames.py   back-projection, normals, voxel fusion
propscan/geometry/drift.py    fragment pose graph + ICP loop closures (drift correction)
propscan/geometry/align.py    gravity + Manhattan alignment
propscan/geometry/plan.py     rooms, walls, ceiling, openings (shared by all tiers)
propscan/damage/detect.py     damage regions, concealed-damage rules, scope items
propscan/uncertainty.py       per-tier error budget -> 95% intervals
propscan/schema.py            output contract (pydantic -> JSON schema)
scripts/                      benchmark, fix loop, photo-set builder, diagnostics
docs/                         protocol, device matrix, technical report, fix loop, compliance matrix
```

Docs: [Technical report](docs/TECHNICAL_REPORT.md) · [Compliance matrix](docs/COMPLIANCE.md) · [Benchmark report](benchmark/REPORT.md) · [Fix loop](docs/FIX_LOOP.md) · [Models disclosure](docs/MODELS.md)
