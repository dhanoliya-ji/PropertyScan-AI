# Fix loop

## Declaration (one page)

**1. Worst gate and its failing number.**
Repeatability: two captures of the same rooms at the same tier must agree within 1 cm or 0.5% per wall.

The pair is LiDAR `single_scan_floor_only` vs `single_scan_with_ceiling`, the only same-tier pair in the data. With drift correction on and the original ("legacy") wall fit:
- **0.0%** of the 25 matched observed walls are within the gate;
- the median |Δlength| is **24.1 cm** (p90 66.2 cm);
- plan IoU is 0.705.

Every other gate is either looser or not measurable without laser ground truth. This one fails hardest and is fully measurable from the data we have.

**2. Root-cause hypothesis and evidence.** There were two rounds; the first was wrong.

- **H1 (rejected): walls are fitted to furniture faces.** The legacy fit takes the first room-facing vertical surface 0.12–2.2 m high. A sofa back or fridge front could win, and which furniture is visible differs between captures.
  - *Evidence for:* the plane spacing inside rooms differed by a median of 11 cm between captures, so surfaces, not just corners, were moving.
  - *Shipped:* a fit restricted to points above 1.5 m (`PROPSCAN_WALL_FIT=high`).
  - *Result:* **worse**, median 24.1 → 41.2 cm.
  - *Post-mortem:* only 19% of edges move more than 5 cm under the high-band fit, so furniture explains a minority. And `floor_only` hardly ever pointed above 1.5 m, so the high band was empty there and edges fell back to the old fit.
- **H2 (accepted): wall position is set by where the occupancy mask happens to stop, not by the wall.**
  - *Evidence:* an overlay of both plans on the wall points ([scripts/overlay_plans.py](../scripts/overlay_plans.py)) shows room outlines ending up to about 40 cm inside clearly visible walls, wherever the floor next to the wall wasn't seen. The edge refinement only searched ±25 cm.
  - *Decomposition:* after re-registration, **wall-plane positions repeat to a median of 2.8 cm**, while lengths differ by 23 cm. The length error lives in the *corners* (which walls bound a segment: notches, splits), not in the plane fit.

**3. Fix and predicted number.**
- *Fix:*
  1. Snap each room edge to the room-facing plane with the best *coverage along the edge* (fraction of 5 cm cells holding wall points within ±2 cm), searching up to 1 m outward. Take the outermost plane with at least 80% of the best coverage, and require coverage of at least 50%.
  2. If no plane qualifies, flag the edge `observed: false` with a wide σ, instead of fitting the nearest surface.
  3. Merge weak short jogs.
  4. **Rectangle snap:** a room whose outline fills ≥ 85% of its wall-snapped bounding rectangle *is* that rectangle.
- *Prediction:* **I did not record a numeric prediction before shipping either attempt.** That is a gap in this fix loop, and I'm stating it rather than back-dating one. The commit messages (`903e9a8`, `1cefcd6`) record each number as it was measured. Expectation stated at the time of H2 (commit `903e9a8`): plane snapping alone would not pass the gate, because the residual was topological.

## Result

| Variant (same fused clouds, only the wall fit differs) | Plan IoU | Walls compared | Within 1 cm / 0.5% | Median abs diff | p90 |
|---|---|---|---|---|---|
| **Before:** legacy | 0.705 | 25 | **0.0%** | **24.1 cm** | 66.2 cm |
| H1: high band (rejected) | 0.707 | 21 | 4.8% | 41.2 cm | 81.1 cm |
| H2 step 1: coverage snap, no rectangle snap | 0.759 | 24 | 4.2% | 33.3 cm | 57.3 cm |
| **After (shipped):** coverage snap + rectangle snap, cover ≥ 0.5 | **0.759** | 23 | **13.0%** | **14.4 cm** | **50.8 cm** |

Coverage-threshold sensitivity (from `benchmark/fixloop_cover_sensitivity.txt`):

| Coverage threshold | Within gate | Median abs diff |
|---|---|---|
| 0.3 | 8.3% | 17.1 cm |
| 0.5 (shipped) | 13.0% | 14.4 cm |
| 0.6 | 4.8% | 34.1 cm |

**The threshold was chosen on this same pair**, the only same-tier pair available, so the 13.0% is optimistic. The direction (any threshold from 0.3 to 0.5 beats legacy) is the robust finding.

**Verdict.**
- The root cause (H2, corner topology) is supported: the median dropped 40% and matched rectangular rooms now agree closely. For example:
  - room 3: 2.297 vs 2.276 m, and 3.218 vs 3.208 m;
  - room 5: 2.330 vs 2.301 m.
- **The gate still fails:** 13% of walls are within 1 cm / 0.5%, against the 100% the gate implies.
- *Why it fell short:* the remaining large differences are coverage, not fitting. `floor_only` never observed several walls that `with_ceiling` did (e.g. the top wall of room 2, where the two plans sit 47 cm apart), and the two walks split the open-plan living area differently (27.1 vs 22.5 m²). A same-protocol repeat capture would remove most of that, and the capture protocol now asks for every wall to be swept floor-to-ceiling.

## Regenerate
```bash
python scripts/fixloop_eval.py --evidence          # before (legacy), H1 (high), after (planes) from the same fused clouds
PROPSCAN_WALL_FIT=legacy python scripts/benchmark.py --force --tag before --runs L_floor L_ceil   # end-to-end before
python scripts/benchmark.py --force --tag main --runs L_floor L_ceil                             # end-to-end after
```
Readable diff: `git diff 903e9a8~1 1cefcd6 -- propscan/geometry/plan.py` (`_fit_edge_planes`, `_merge_jogs`, `_rectangle_snap`). Both modes remain selectable through `PROPSCAN_WALL_FIT`.
