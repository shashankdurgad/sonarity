# sim/: OceanSim on Modal (owner: P2, scene: P3)

## Running

```bash
python -P -m modal run infra/step3_sonar.py   # sonar + UW camera smoke test on Modal L40S
```
Use `python -P` on any machine: the `-P` keeps the repo root off `sys.path`, so a stray local folder can't shadow the `modal` library.

- Image: `infra/image.py`, built on **Isaac Sim 6.0.1** (5.x segfaults on Modal's driver 610) with OceanSim cloned into `/isaac-sim/extsUser/OceanSim`. Every sim job should import `isaac_image`, `cache_volume`, `link_caches` and `run_isaac` from it.
- Isaac-side scripts run under `/isaac-sim/python.sh`, not your local Python. Run them through `run_isaac("sim/...py", ...)`.
- First start on a fresh cache takes about 2.5 min (shader compile). After that the cache on the `isaac-cache` Volume makes it faster.

## Things that bit us (read before writing Isaac code)

1. **Initialise sensors before `world.reset()`.** Otherwise `camera.get_rgba()` returns `None` on Isaac 6.
2. **OceanSim isn't found by the extension manager.** We import it by appending `/isaac-sim/extsUser/OceanSim/isaacsim` to `isaacsim.__path__` (see `sim/smoke/sonar_render.py`).
3. **The sonar only sees objects that have a `reflectivity` label:**
   ```python
   from isaacsim.core.utils.semantics import add_labels
   add_labels(prim, labels=["1.0"], instance_name="reflectivity")   # seabed ~0.3, objects ~1.0
   ```
4. **Sonar headless:** `sonar.sonar_initialize(viewport=False)`, then each frame call `world.step(render=True)`, `sonar.make_sonar_data()` and `sonar.get_sonar_image().numpy()`. The result is a polar grid of shape `(n_range, n_azimuth, 4)` with row 0 at `min_range`.
5. **UW camera:** `cam.initialize(viewport=False)`, then each frame call `cam.render()` and read `cam._uw_frame` (640×360×4 uint8). There is no public getter.

## Sonar frame convention (`runs/<id>/sonar/*.png`)

The geometry lives in `sim/sonar_geometry.py`. It is plain NumPy, so it runs on a laptop.

- 512×512 greyscale **fan** image. The sensor is at the bottom centre, "up" is the robot's heading, and image right is starboard.
- Distance from the sensor pixel is **slant range** at a fixed `SonarParams().m_per_px`.
- Detector box to world position: `pixel_to_world(col, row, pose_row_from_poses_csv)`.
- `world.json` object to label pixel: `world_to_pixel(x, y, pose)`. Use this to make YOLO labels.
