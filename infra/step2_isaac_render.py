"""Step 2 of the OceanSim proof: does Isaac Sim render headless on Modal's L40S?

Run: OUT=/path/to/cube.png modal run infra/step2_isaac_render.py
Pass: a PNG with the cube visible. Saves the shader cache to the uxo-cache Volume.
"""
import os
import pathlib
import subprocess
import time

import modal

ISAAC_TAG = os.environ.get("ISAAC_TAG", "6.0.1")

image = (
    modal.Image.from_registry(f"nvcr.io/nvidia/isaac-sim:{ISAAC_TAG}", add_python="3.11")
    .dockerfile_commands("USER root")
    .entrypoint([])
    .apt_install("vulkan-tools")
    .env({"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMNI_KIT_ACCEPT_EULA": "YES", "OPENBLAS_NUM_THREADS": "1"})
)
app = modal.App("uxo-step2-isaac", image=image)
cache = modal.Volume.from_name("uxo-cache", create_if_missing=True)

CACHE_DIRS = "isaac-sim/kit/cache root/.cache/ov root/.local/share/ov root/.nv"
TARBALL = f"/cache/isaac_shader_cache_{ISAAC_TAG}.tgz"

RENDER = r'''
import time
t0 = time.time()
from isaacsim import SimulationApp
sim = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480})
print(f"PROOF kit_started {time.time()-t0:.0f}s", flush=True)
import omni.kit.app
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")
floor = UsdGeom.Cube.Define(stage, "/World/Floor")
floor.AddScaleOp().Set(Gf.Vec3f(10, 10, 0.05))
floor.GetDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.4, 0.4)])
cube = UsdGeom.Cube.Define(stage, "/World/Cube")
cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.0))
cube.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.15, 0.1)])
UsdLux.DomeLight.Define(stage, "/World/Dome").GetIntensityAttr().Set(1000)

cam = rep.create.camera(position=(5, 5, 4), look_at=(0, 0, 1))
rp = rep.create.render_product(cam, (640, 480))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)
for _ in range(10):
    rep.orchestrator.step(rt_subframes=4)
rgb = np.asarray(ann.get_data())[..., :3]
print(f"PROOF frame shape={rgb.shape} mean={rgb.mean():.1f} std={rgb.std():.1f} total={time.time()-t0:.0f}s", flush=True)
Image.fromarray(rgb.astype("uint8")).save("/tmp/out.png")
print("PROOF saved", flush=True)
sim.close()
'''


@app.function(gpu=os.environ.get("TEST_GPU", "L40S"), timeout=1500, scaledown_window=60, volumes={"/cache": cache})
def render() -> bytes | None:
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    if os.path.exists(TARBALL):
        print("restoring shader cache", flush=True)
        subprocess.run(f"tar -xzf {TARBALL} -C /", shell=True)
    pathlib.Path("/tmp/render.py").write_text(RENDER)
    t0 = time.time()
    p = subprocess.Popen(["/isaac-sim/python.sh", "/tmp/render.py"], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line.rstrip()[:300], flush=True)
    rc = p.wait()
    print(f"PROOF isaac_exit={rc} after {time.time()-t0:.0f}s", flush=True)
    if not os.path.exists("/tmp/out.png"):
        return None
    subprocess.run(f"cd / && tar -czf {TARBALL} --ignore-failed-read {CACHE_DIRS} 2>/dev/null", shell=True)
    cache.commit()
    print("PROOF shader cache saved", flush=True)
    return pathlib.Path("/tmp/out.png").read_bytes()


@app.local_entrypoint()
def main() -> None:
    png = render.remote()
    out = os.environ.get("OUT", "cube.png")
    if png is None:
        print("RESULT: FAIL, no image produced")
        raise SystemExit(1)
    pathlib.Path(out).write_bytes(png)
    print(f"RESULT: image written to {out} ({len(png)} bytes)")
