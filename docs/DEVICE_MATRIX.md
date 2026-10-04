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

| Tier | Wall length | Ceiling height | Openings | Typical 95% interval on a 3 m wall |
|---|---|---|---|---|
| LiDAR | see benchmark report | ±1–2 cm where the ceiling was scanned; prior-based ±24 cm and flagged `observed: false` where it wasn't | doors from walk-throughs, windows from rays passing through the wall | ±2 cm (fit + 0.3% scale) |
| Video | see benchmark report | ±18 cm (model scale) | doors from walk-throughs | ±18 cm (3% scale term + surfaces) |
| Photos | see benchmark report | prior unless the ceiling is in view | doors from `door_to_*` photos (width prior ±16 cm unless measured) | ±43 cm (7% scale term) |

The interval widths above are the pre-calibration error budget in [propscan/uncertainty.py](../propscan/uncertainty.py). The calibration table in the benchmark report shows how often the LiDAR value falls inside each tier's interval.
