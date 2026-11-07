"""PLAN.md §3 step 3: OceanSim imaging sonar + underwater camera, headless, aimed at a sphere.

Runs inside the Isaac container:  /isaac-sim/python.sh sim/smoke/sonar_render.py --out /tmp/out
Each sensor is tried independently so one run reports as much as possible.
"""
import argparse
import os
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/tmp/out")
parser.add_argument("--oceansim", default="/isaac-sim/extsUser/OceanSim")
args, _ = parser.parse_known_args()
os.makedirs(args.out, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 640, "height": 360})


def log(msg):
    print(f"[sonar_render] {msg}", flush=True)


# --- Load OceanSim: try the extension manager first, then plain sys.path import. ---
try:
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.add_path(os.path.dirname(args.oceansim))
    mgr.add_path(args.oceansim)
    if hasattr(mgr, "refresh_registry"):
        mgr.refresh_registry()
    for _ in range(5):
        app.update()
    ok = mgr.set_extension_enabled_immediate("isaacsim.oceansim", True)
    log(f"enable isaacsim.oceansim via ext manager: {ok}")
except Exception:
    log("ext manager enable failed:\n" + traceback.format_exc())

# Fallback: `isaacsim` is a regular package, so graft OceanSim's subpackage onto its __path__.
try:
    import isaacsim

    sub = os.path.join(args.oceansim, "isaacsim")
    if sub not in list(isaacsim.__path__):
        isaacsim.__path__.append(sub)
    import isaacsim.oceansim  # noqa: F401

    log(f"isaacsim.oceansim importable from {isaacsim.oceansim.__path__}")
except Exception:
    log("isaacsim.oceansim import fallback failed:\n" + traceback.format_exc())
    os.system(f"ls -la {args.oceansim} {args.oceansim}/isaacsim {args.oceansim}/config")

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid, VisualSphere  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdLux  # noqa: E402


def set_reflectivity(prim_path, value):
    """OceanSim only sees prims with a 'reflectivity' semantic."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    try:
        from isaacsim.core.utils.semantics import add_update_semantics

        add_update_semantics(prim=prim, type_label="reflectivity", semantic_label=str(value))
        return "add_update_semantics"
    except Exception as e1:
        try:
            from isaacsim.core.utils.semantics import add_labels

            add_labels(prim, labels=[str(value)], instance_name="reflectivity")
            return "add_labels"
        except Exception as e2:
            return f"FAILED ({e1!r}; {e2!r})"


world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(3000.0)
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(500.0)

# Seabed slab + target sphere 3 m ahead of the sensors.
world.scene.add(VisualCuboid(prim_path="/World/seabed", position=np.array([5.0, 0.0, -0.05]),
                             scale=np.array([20.0, 20.0, 0.1]), color=np.array([0.76, 0.70, 0.50])))
world.scene.add(VisualSphere(prim_path="/World/target", position=np.array([3.0, 0.0, 0.3]),
                             radius=0.3, color=np.array([0.3, 0.3, 0.3])))
log(f"reflectivity seabed: {set_reflectivity('/World/seabed', 0.3)}")
log(f"reflectivity target: {set_reflectivity('/World/target', 1.0)}")

# Sensors 1.5 m up at the origin, looking +X, pitched ~25 deg down.
pose_q = euler_angles_to_quat(np.array([0.0, 25.0, 0.0]), degrees=True)
pose_p = np.array([0.0, 0.0, 1.5])

sonar = cam = None
try:
    from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor

    sonar = ImagingSonarSensor(prim_path="/World/sonar", position=pose_p, orientation=pose_q,
                               min_range=0.2, max_range=6.0, hori_fov=130.0, vert_fov=20.0,
                               angular_res=0.5, hori_res=1000)
    sonar.initialize()
    log("sonar created")
except Exception:
    log("SONAR create FAILED:\n" + traceback.format_exc())
    sonar = None

try:
    from isaacsim.oceansim.sensors.UW_Camera import UW_Camera

    cam = UW_Camera(prim_path="/World/uwcam", resolution=(640, 360),
                    position=pose_p + np.array([0.0, 0.2, 0.0]), orientation=pose_q)
    log("uw camera created")
except Exception:
    log("UW_CAMERA create FAILED:\n" + traceback.format_exc())
    cam = None

world.reset()

if sonar is not None:
    try:
        sonar.sonar_initialize(viewport=False, include_unlabelled=False)
        log("sonar_initialize ok")
    except Exception:
        log("sonar_initialize FAILED:\n" + traceback.format_exc())
        sonar = None
if cam is not None:
    try:
        cam.initialize(viewport=False)
        log("uw camera initialize ok")
    except Exception:
        log("uw camera initialize FAILED:\n" + traceback.format_exc())
        cam = None

for _ in range(30):
    world.step(render=True)

sonar_ok = cam_ok = False
for i in range(30):
    world.step(render=True)
    if sonar is not None and not sonar_ok:
        try:
            sonar.make_sonar_data()
            img = sonar.get_sonar_image()
            arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
            if arr is not None and arr.size and arr.max() > 0:
                log(f"sonar image shape={arr.shape} dtype={arr.dtype} mean={arr.mean():.1f} max={arr.max()}")
                rgb = arr[..., :3] if arr.ndim == 3 else arr
                Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(args.out, "sonar_raw.png"))
                data = sonar.get_sonar_data()
                d = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
                np.save(os.path.join(args.out, "sonar_data.npy"), d)
                log(f"sonar data shape={d.shape} intensity max={d[..., 2].max():.3f}")
                sonar_ok = True
        except Exception:
            log(f"sonar step {i} FAILED:\n" + traceback.format_exc())
            sonar = None
    if cam is not None and not cam_ok:
        try:
            cam.render()
            fr = getattr(cam, "_uw_frame", None)
            if fr is not None and np.asarray(fr).size:
                fr = np.asarray(fr)
                log(f"uw frame shape={fr.shape} dtype={fr.dtype} max={fr.max()}")
                if fr.dtype != np.uint8:
                    fr = np.clip(fr * (255.0 if fr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                Image.fromarray(fr[..., :3]).save(os.path.join(args.out, "uwcam.png"))
                cam_ok = True
        except Exception:
            log(f"uw camera step {i} FAILED:\n" + traceback.format_exc())
            cam = None
    if (sonar_ok or sonar is None) and (cam_ok or cam is None):
        break

log(f"RESULT sonar={'PASS' if sonar_ok else 'FAIL'} uwcam={'PASS' if cam_ok else 'FAIL'}")
app.close()
