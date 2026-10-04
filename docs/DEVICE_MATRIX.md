# Device matrix

| Tier | Runs on | Capture app | What the pipeline gets |
|---|---|---|---|
| **LiDAR** | iPhone 12 Pro – 17 Pro / Pro Max, and iPad Pro (2020+) with LiDAR | Stray Scanner (free) | 256×192 depth + confidence at about 46 fps, ARKit VIO poses (gravity-aligned), per-frame intrinsics, 1920×1440 RGB |
| **Video** | Any iPhone 15 or newer (15, 15 Plus, 15 Pro, 16 series, 17 series); runs on any phone that records 1080p | Built-in Camera, Video, 1x | RGB video only. Intrinsics come from a prior for the 1x main camera. No depth, no IMU. |
| **Photos** | Any iPhone 15 or newer | Built-in Camera, Photo, 1x | JPEG stills. Intrinsics from EXIF 35 mm-equivalent focal length (prior if missing). No depth, no poses. |

The non-Pro iPhone 15/16/17 models have no LiDAR, so they run the video and photo tiers only. Every Pro model runs all three.

## What each tier honestly delivers

These numbers are measured on the provided sample captures. **No laser or tape ground truth came with the sample data**, so "accuracy" here means:
- **LiDAR:** repeatability between two independent captures of the same apartment. That gives an upper bound on random error; it can't reveal shared bias.
- **Video and photos:** agreement with the LiDAR tier of the same capture.

See [benchmark/REPORT.md](../benchmark/REPORT.md) for the tables and how to regenerate them.

| Tier | Measured on the sample data (see benchmark/REPORT.md) | Interval on a 3 m wall (error budget) |
|---|---|---|
| LiDAR | Repeatability between the two apartment captures: wall planes agree to about 3 cm median, but wall *lengths* differ by a median of 23 cm (corner topology and coverage). 0% within 1 cm / 0.5%. Ceiling ±1–2 cm where scanned; otherwise a prior flagged `observed: false`. | ±2 cm |
| Video | Not usable as a measurement tier yet: single_room walls have 59% median error against LiDAR, and the footprint is −35%. Monocular scale chaining fails on long clips. | ±18 cm (not calibrated: coverage 0%) |
| Photos | Stitch runs (4/7 rooms recovered, adjacency correct, no overlaps), but the footprint is −47% against LiDAR because of missing rooms. | ±43 cm |

The interval widths above are the pre-calibration error budget in [propscan/uncertainty.py](../propscan/uncertainty.py). The calibration table in the benchmark report shows how often the LiDAR value falls inside each tier's interval.
