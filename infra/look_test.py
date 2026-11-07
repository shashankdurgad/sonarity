"""Render-look tuning job: python -P -m modal run infra/look_test.py  -> out/look/"""
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(__file__))
from image import CACHE_MOUNT, cache_volume, isaac_image, link_caches, run_isaac  # noqa: E402

app = modal.App("uxo-look-test")


@app.function(image=isaac_image, env={"LOOK_SCRIPT": os.environ.get("LOOK_SCRIPT", "sim/smoke/look_test.py")}, gpu="L40S", timeout=1800, volumes={CACHE_MOUNT: cache_volume})
def look() -> dict:
    link_caches()
    run_isaac(os.environ.get("LOOK_SCRIPT", "sim/smoke/look_test.py"), "--out", "/tmp/out")
    cache_volume.commit()
    return {n: open(f"/tmp/out/{n}", "rb").read() for n in os.listdir("/tmp/out")}


@app.local_entrypoint()
def main():
    files = look.remote()
    os.makedirs("out/look", exist_ok=True)
    for n, b in files.items():
        open(f"out/look/{n}", "wb").write(b)
    print(f"saved {len(files)} files to out/look/")
