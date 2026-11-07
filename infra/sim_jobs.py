"""Modal sim jobs (owner: P2).

Survey (PLAN §3 step 4 / Gate A):
    python -P -m modal run infra/sim_jobs.py --seed 3 --scenario clear_sand [--quick]
Downloads to runs/oceansim_<scenario>_s<seed>[_quick]/ and also keeps a copy on the `uxo-runs` Volume.
"""
import io
import os
import sys
import tarfile
import time

import modal

sys.path.insert(0, os.path.dirname(__file__))
from image import CACHE_MOUNT, cache_volume, isaac_image, link_caches, run_isaac  # noqa: E402

app = modal.App("uxo-sim")
runs_volume = modal.Volume.from_name("uxo-runs", create_if_missing=True)


@app.function(image=isaac_image, gpu="L40S", timeout=3600,
              volumes={CACHE_MOUNT: cache_volume, "/runs": runs_volume})
def survey(run_id: str, seed: int, scenario: str, quick: bool = False, flip_azimuth: bool = False) -> bytes:
    link_caches()
    out = f"/runs/{run_id}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    cmd = ["sim/survey.py", "--out", out, "--seed", str(seed), "--scenario", scenario]
    if quick:
        cmd.append("--quick")
    if flip_azimuth:
        cmd.append("--flip-azimuth")
    rc = run_isaac(*cmd, timeout=3300)
    cache_volume.commit()
    runs_volume.commit()
    print(f"[sim_jobs] survey rc={rc} in {time.time() - t0:.0f}s", flush=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=run_id)
    return buf.getvalue()


@app.function(image=isaac_image, gpu="L40S", timeout=3600,
              volumes={CACHE_MOUNT: cache_volume, "/runs": runs_volume})
def survey_to_volume(run_id: str, seed: int, scenario: str) -> dict:
    """Training-data survey: no chase camera, output stays on the uxo-runs Volume (nothing downloaded)."""
    link_caches()
    out = f"/runs/{run_id}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rc = run_isaac("sim/survey.py", "--out", out, "--seed", str(seed), "--scenario", scenario, "--no-chase",
                   timeout=3300)
    runs_volume.commit()
    labels = os.path.join(out, "labels")
    n = len(os.listdir(labels)) if os.path.isdir(labels) else 0
    n_lab = sum(1 for f in os.listdir(labels) if os.path.getsize(os.path.join(labels, f))) if n else 0
    return {"run_id": run_id, "rc": rc, "frames": n, "labelled": n_lab, "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def batch(seeds: str = "10-21", scenarios: str = "clear_sand", tag: str = "train"):
    """Many surveys in parallel, kept on the uxo-runs Volume:
    python -P -m modal run infra/sim_jobs.py::batch --seeds 10-21 --scenarios clear_sand,half_buried
    """
    lo, hi = (int(v) for v in seeds.split("-"))
    jobs = [(f"{tag}_{sc}_s{s}", s, sc) for sc in scenarios.split(",") for s in range(lo, hi + 1)]
    print(f"launching {len(jobs)} surveys: {[j[0] for j in jobs]}")
    total = 0
    for res in survey_to_volume.starmap(jobs):
        total += res["labelled"]
        print(f"  {res}")
    print(f"done: {len(jobs)} runs, {total} labelled frames, on Volume uxo-runs")


@app.local_entrypoint()
def main(seed: int = 3, scenario: str = "clear_sand", quick: bool = False, flip_azimuth: bool = False):
    run_id = f"oceansim_{scenario}_s{seed}" + ("_quick" if quick else "")
    data = survey.remote(run_id, seed, scenario, quick, flip_azimuth)
    os.makedirs("runs", exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall("runs", filter="data")
    print(f"Saved runs/{run_id}/ ({len(data) / 1e6:.1f} MB)")
