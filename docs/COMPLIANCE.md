# Compliance matrix

Status key:
- **Done:** implemented and exercised on the sample data.
- **Partial:** implemented, with a stated gap.
- **Not measured:** the code path exists, but the input needed to score it (laser ground truth, staged damage, an incumbent-app export) was not available.
- **Fail:** measured and below the gate.

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Capture route (Route 2: stock apps + one-page protocol) | docs/CAPTURE_PROTOCOL.md | protocol page | Done |
| 1.2 | Photo tier: 2–8 stills per room, folders → stitched whole-property plan | propscan/tiers/photos.py | `benchmark/runs/main/P_ceil.json/.png` | Partial: runs and stitches through door photos; not every room registers on video-derived stills |
| 1.3 | Video tier: handheld walkthrough | propscan/tiers/video.py | `benchmark/runs/main/V_*.json/.png` | Partial: runs end to end; accuracy fails the gate (59% median error) |
| 1.4 | LiDAR tier: depth, poses, intrinsics | propscan/tiers/lidar.py, propscan/io/stray.py | `benchmark/runs/main/L_*.json/.png` | Done |
| 1.5 | Device matrix with honest per-tier accuracy | docs/DEVICE_MATRIX.md | table | Done |
| 2.1 | Per-room plan: walls, ceiling height, floor area, openings | propscan/geometry/plan.py, propscan/assemble.py | `Room` in the JSON | Done |
| 2.2 | Stitched multi-room plan with adjacency | plan.py (`_crossing_doors`), photos.py (stitch) | `adjacency` in the JSON, rendered plan | Done (LiDAR/video), Partial (photos) |
| 2.3 | Per-surface damage regions with class and metric extent | propscan/damage/detect.py | `damage[]` | Partial: synthetic-injection test only, no staged damage in the data |
| 2.4 | Concealed-damage flags with the rule that fired | detect.py (`RULES`, `_rules`) | `concealed_damage[]` (rule_id + rule text) | Done (rules R1–R5) |
| 2.5 | Scope line items keyed to surfaces | detect.py (`_scope`) | `scope[]` (surface_id) | Done |
| 2.6 | Confidence interval on every measurement | propscan/uncertainty.py | `Measurement.ci95`, `sigma`, `observed` | Done |
| 2.7 | One command per capture | propscan/__main__.py | `python -m propscan run <capture>` | Done |
| 2.8 | JSON to a published schema | propscan/schema.py | schema/property_scan.schema.json | Done |
| 2.9 | Rendered plan | propscan/render.py | `*_plan.png`, `*_plan.svg` | Done |
| 2.10 | Benchmark set: multi-room ≥ 3 rooms + connector | sample data (with_ceiling: 10 rooms incl. connector) | benchmark/REPORT.md | Done (provided data) |
| 2.11 | Benchmark: furnished room with staged damage, 2 classes | scripts/synthetic_damage_test.py | benchmark/synthetic_damage.json | Not measured on real damage (none in the data); synthetic injection instead |
| 2.12 | Same rooms at all three tiers | benchmark.py (`L_*`, `V_*`, `P_ceil`) | results_main.json | Done (video/photos derived from the same capture) |
| 2.13 | A room captured twice at the same tier | floor_only vs with_ceiling (LiDAR) | repeatability table | Done |
| 2.14 | Laser/tape ground truth | — | — | Not measured: none provided; LiDAR is used as the reference for the thinner tiers |
| G1 | Opening widths ≤ 2 cm on ≥ 85% | plan.py (`_measure_door`, wall gaps) | repeatability of opening widths | Not measured vs laser; cross-capture consistency reported |
| G2 | Ceiling height ≤ 1.5 cm; spread ≤ 1 cm | plan.py (per-room levels) | REPORT.md | Partial: only with_ceiling observes ceilings, so no cross-capture spread exists |
| G3 | Repeatability 1 cm / 0.5% per wall | scripts/benchmark.py, scripts/fixloop_eval.py | REPORT.md, FIX_LOOP.md | Fail: 0% within gate on the final benchmark; fix loop 0% → 14.3% on fixed clouds |
| G4 | Drift accountability + on/off ablation | propscan/geometry/drift.py, `--no-drift` | drift table, ghost-wall metric | Done |
| G5 | Photo-tier whole-property stitch, ±8% | photos.py, benchmark.py (`photo_stitch`) | REPORT.md | Fail: adjacency correct, no overlaps, but 4/7 rooms and footprint −47% |
| G6 | Video ±3%, photo ±8%, calibration at every tier | benchmark.py (`cross_tier`) | REPORT.md (CI coverage) | Fail: video 0% within ±3%, intervals not calibrated |
| 3 | Head-to-head vs a consumer app | — | — | Not done: needs a Polycam/magicplan scan of the same rooms on an iPhone; no device or access to the apartment |
| 4 | Fix loop: declaration, shipped fix, regenerable before/after, diff | docs/FIX_LOOP.md, scripts/fixloop_eval.py, `PROPSCAN_WALL_FIT` | benchmark/fixloop_eval.json | Done |
| 5 | Process evidence | git history | incremental commits with measured numbers | Done |
| D3 | README to running in < 15 min | README.md | — | Done |
| D4 | Reproduction bundle, deterministic cache + live path | scripts/benchmark.py, propscan/models/depth.py (cache) | `.cache/depth` | Done |
| D7 | Technical report ≤ 6 pages | docs/TECHNICAL_REPORT.md | — | Done |
| D8 | Raw benchmark data | sample data (provided zips), data/photos (regenerable) | — | Done (no app exports exist) |
| C1 | Runs without our infrastructure; weights fetched by script | scripts/fetch_weights.py, docs/MODELS.md | — | Done |
| C2 | Mirrors, glass, wet-look surfaces, low light covered | plan.py (mirror reflection test), TECHNICAL_REPORT §7 | warnings in JSON | Partial: mirrors handled; glass, wet-look and low light documented as failure modes |
