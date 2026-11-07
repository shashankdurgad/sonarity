# sim/: OceanSim on Modal (owner: P2, scene: P3)

## Running

```bash
python -P -m modal run infra/sim_jobs.py::main --seed 3 [--scenario half_buried|murky] [--quick]   # one survey -> runs/<id>/
python -P -m modal run infra/sim_jobs.py::batch --seeds 10-19 --scenarios clear_sand,half_buried   # training data, stays on Volume uxo-runs
python -P -m modal run infra/step3_sonar.py   # sonar + UW camera smoke test
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

## Gotchas found integrating sim/scene.py (P3)

6. **The seabed must cover everything the sonar can see.** If the sonar looks past its edge, OceanSim's point cloud is empty and it throws `KeyError: .../Replicator_pointcloud`. **You can't rebuild the seabed after the sonar has started** (every later frame goes blank), so size it up front: `SimSession(seabed_bounds=seabed_bounds_for(world))`. `observe()` returns a blank sonar frame rather than crashing if it ever sees nothing.
7. SimSession uses `sim/scene.py` when it exists. It passes `build_scene` a labeller that adds the `reflectivity` label, asks for `include_seabed=False` (we build our own textured, labelled seabed), and uses the returned prim paths for `apply_change`. `sim/objects.get_world()` uses `scene.make_world`, and label sizes come from `scene.DIMENSIONS_M` if present.
8. Single-colour meshes must set `displayColor` interpolation to `constant`, or RTX reads it as per-vertex and complains about corrupted data.
