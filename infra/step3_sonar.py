"""PLAN.md §3 step 3: OceanSim sonar + underwater camera on Modal.

Run:  python -P -m modal run infra/step3_sonar.py   (-P: repo-root modal/ folder shadows the library)
Pass: out/step3/sonar_raw.png shows the sphere, out/step3/uwcam.png shows an underwater view.
"""
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(__file__))
from image import CACHE_MOUNT, cache_volume, isaac_image, link_caches, run_isaac  # noqa: E402

app = modal.App("uxo-step3-sonar")


@app.function(image=isaac_image, gpu="L40S", timeout=2400, volumes={CACHE_MOUNT: cache_volume})
def render_sonar() -> dict:
    link_caches()
    t0 = time.time()
    rc = run_isaac("sim/smoke/sonar_render.py", "--out", "/tmp/out")
    cache_volume.commit()
    files = {}
    if os.path.isdir("/tmp/out"):
        for name in os.listdir("/tmp/out"):
            files[name] = open(os.path.join("/tmp/out", name), "rb").read()
    commit = open("/isaac-sim/extsUser/OceanSim.commit").read().strip()
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "files": files, "oceansim_commit": commit}


@app.local_entrypoint()
def main():
    res = render_sonar.remote()
    print(f"\nexit code {res['rc']}, Isaac run took {res['seconds']} s, OceanSim commit {res['oceansim_commit']}")
    os.makedirs("out/step3", exist_ok=True)
    for name, data in res["files"].items():
        with open(os.path.join("out/step3", name), "wb") as f:
            f.write(data)
        print(f"Saved out/step3/{name}")
