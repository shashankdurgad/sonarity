"""Generate labelled OceanSim surveys for YOLO run 2 on the uxo-runs Volume, one GPU per survey.

Run: python -P -m modal run --detach infra/gen_frames.py
Writes: uxo-runs Volume, train_<scenario>_s<seed>/. Seeds 106 and 112 are held out for testing.
"""
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(__file__))
from image import CACHE_MOUNT, cache_volume, isaac_image, link_caches, run_isaac  # noqa: E402

app = modal.App("uxo-gen-frames")
runs_volume = modal.Volume.from_name("uxo-runs", create_if_missing=True)
JOBS = [("clear_sand", s) for s in range(101, 107)] + [("half_buried", s) for s in range(107, 113)]


@app.function(image=isaac_image, gpu="L40S", timeout=3600, max_containers=8,
              volumes={CACHE_MOUNT: cache_volume, "/runs": runs_volume})
def survey(scenario: str, seed: int) -> int:
    link_caches()
    out = f"/runs/train_{scenario}_s{seed}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rc = run_isaac("sim/survey.py", "--out", out, "--seed", str(seed), "--scenario", scenario, timeout=3300)
    runs_volume.commit()
    print(f"[gen_frames] {scenario} s{seed} rc={rc} in {time.time() - t0:.0f}s", flush=True)
    return rc


@app.local_entrypoint()
def main() -> None:
    calls = [survey.spawn(scenario, seed) for scenario, seed in JOBS]
    print(f"spawned {len(calls)} surveys")
