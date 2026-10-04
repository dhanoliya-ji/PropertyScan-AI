# Capture protocol (Route 2: stock apps) — one page

Follow it literally. Each tier is captured in one go, without switching apps or changing settings partway through.

## 0. One-time setup (≤ 5 min)
1. iPhone 15 or newer. **Settings → Camera → Formats → "Most Compatible"** (JPEG/H.264; HEIC/HEVC is not accepted).
2. **LiDAR tier only (iPhone Pro / Pro Max):** install **Stray Scanner** (App Store, free, by Kenneth Blomqvist).
3. Turn on all room lights and open the internal doors between the rooms you will capture. Close any doors to rooms you will *not* capture.

## 1. LiDAR tier — Stray Scanner (Pro devices)
1. Open Stray Scanner and tap record. Hold the phone **in front of your chest, in portrait**, with the screen facing you.
2. Start in the doorway of the first room. **Walk slowly, about half your normal speed,** around each room keeping about 1 m from the walls. Sweep the camera **up to the ceiling line and down to the floor line** once on every wall.
3. Walk through each doorway **once in each direction**, filming the door frame as you pass. Visit every room, then **finish where you started**, pointing at the same view as your first second of recording. This closes the loop for drift correction.
4. Typical time: 40 s per room. Do not pause. Do not cover the top of the phone (the LiDAR sensor).
5. Stop. In Stray Scanner: **Library → (your scan) → Share → Save to Files / AirDrop** the `.zip`.

## 2. Video tier — built-in Camera app (any iPhone 15+)
1. Camera → **Video**, **1x** lens, 1080p or 4K at 30 fps. Hold the phone **sideways (landscape)**.
2. Walk the same route as in §1 at the same slow pace: walls at about 1 m, sweeping ceiling to floor, every doorway in both directions, ending where you started.
3. **Turn slowly.** A full 360° turn should take about 8 seconds. Fast whip-pans are what breaks tracking.
4. AirDrop or USB the `.mov`/`.mp4` file **unchanged** (don't trim it in Photos).

## 3. Photo tier — built-in Camera app (any iPhone 15+)
For **each room**, make a folder named after the room (e.g. `kitchen`). In it:
1. **Room photos (4–8):** Camera → **Photo**, **1x**, no Portrait mode. Stand in a corner and photograph the opposite corner, with the floor line and ceiling line both in view. Then turn about **40°** and take the next photo, so each photo **overlaps the previous one by about half**. Repeat from a second corner until every wall has appeared in at least two photos.
2. **Door photos (1 per doorway):** stand inside the room about **2 m back from each doorway** and take one photo with the door frame **centred**. Name it `door_to_<other room folder name>.jpg` (e.g. `door_to_hall.jpg`).
3. Put all room folders inside one property folder.

```
my_flat/
  kitchen/  IMG_0001.jpg … IMG_0006.jpg  door_to_hall.jpg
  hall/     IMG_0101.jpg … door_to_kitchen.jpg  door_to_bedroom.jpg
  bedroom/  … door_to_hall.jpg
```

## 4. What to avoid (all tiers)
- **Mirrors and glass:** don't stand square-on to a mirror or glass door. Capture it at an angle. The pipeline rejects mirror "rooms", but a straight-on view hides the wall behind it.
- **Low light:** turn on every light. With LiDAR, dark rooms still measure; with video and photos they don't.
- **People and pets** walking through the frame. **Wet or shiny floors:** wipe them, or capture them at an angle.
- Don't zoom. Don't switch lenses. Don't edit or crop files.

## 5. Hand-off and run
```
python -m propscan run <scan.zip | walk.mov | my_flat/> -o out/
```
The tier is detected automatically. Output: `out/<id>.json` (schema in `schema/`) and `out/<id>_plan.png|svg`.
