"""PLAN.md §3 step 2: headless Isaac Sim renders a cube to a PNG.

Runs inside the Isaac container:  /isaac-sim/python.sh sim/smoke/cube_render.py --out /tmp/out/cube.png
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/tmp/out/cube.png")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 640, "height": 480})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import UsdLux  # noqa: E402

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()
UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(3000.0)
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(500.0)

world.scene.add(
    VisualCuboid(
        prim_path="/World/cube",
        position=np.array([0.0, 0.0, 0.5]),
        scale=np.array([1.0, 1.0, 1.0]),
        color=np.array([1.0, 0.2, 0.2]),
    )
)

# Camera at +X looking back at the origin, tilted ~20 deg down (Isaac "world" camera axes: +X forward, +Z up).
camera = Camera(
    prim_path="/World/camera",
    position=np.array([4.0, 0.0, 2.0]),
    frequency=20,
    resolution=(640, 480),
    orientation=euler_angles_to_quat(np.array([0.0, 20.0, 180.0]), degrees=True),
)

camera.initialize()
world.reset()

import omni.replicator.core as rep  # noqa: E402

# Independent annotator on the same render product, as a cross-check.
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([camera.get_render_product_path()])


def grab(tag):
    a = camera.get_rgba()
    b = annot.get_data()
    sa = None if a is None else getattr(a, "shape", None)
    sb = None if b is None else getattr(b, "shape", None)
    print(f"[cube_render] {tag}: camera.get_rgba={sa} annot={sb}", flush=True)
    for x in (a, b):
        if x is not None and getattr(x, "size", 0) > 0 and len(x.shape) == 3:
            return x
    return None


# Isaac 6 may need different pumping than 5.x; try methods in order, stop at first image.
rgba = None
methods = [
    ("world.step x60", lambda: [world.step(render=True) for _ in range(60)]),
    ("app.update x30", lambda: [app.update() for _ in range(30)]),
    ("rep.orchestrator.step x5", lambda: [rep.orchestrator.step(rt_subframes=4, delta_time=0.0) for _ in range(5)]),
]
for tag, fn in methods:
    fn()
    rgba = grab(tag)
    if rgba is not None:
        print(f"[cube_render] WORKS: {tag}", flush=True)
        break

if rgba is None:
    raise SystemExit("[cube_render] FAIL: camera returned no data")

rgb = rgba[..., :3].astype(np.uint8)
print(f"[cube_render] pixel mean={rgb.mean():.1f} max={rgb.max()}", flush=True)

from PIL import Image  # noqa: E402

os.makedirs(os.path.dirname(args.out), exist_ok=True)
Image.fromarray(rgb).save(args.out)
print(f"[cube_render] wrote {args.out}", flush=True)

app.close()
