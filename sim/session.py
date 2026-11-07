"""SimSession: one live Isaac Sim + OceanSim session with a sonar + underwater camera rig (owner: P2).

Runs under Isaac's Python (/isaac-sim/python.sh). Survey, visit and a live mission loop all use this:

    sim = SimSession()                       # boots Isaac (~1-2.5 min), creates sensors
    sim.load_world(world)                    # world.json dict (PLAN §11)
    sim.move_to(x, y, yaw, altitude=2.5)     # teleport the rig
    sonar, cam = sim.observe()               # (512x512 uint8 fan, 360x640x3 uint8)
    sim.apply_change({"type": "add_object", "object": {...}})   # mid-run scene change
    sim.close()

The robot's decision logic must only use observe() output, never sim.world (that's ground truth).
"""
import math
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sim.objects import CLASS_SPECS  # noqa: E402
from sim.sonar_geometry import SonarParams, polar_to_fan  # noqa: E402

OCEANSIM = "/isaac-sim/extsUser/OceanSim"
CAM_RES = (640, 360)
CAM_PITCH = 40.0  # deg down
WATER_RGB = np.array([18, 62, 74], dtype=np.float32)  # murk colour (dark teal)
REFLECTIVITY = {"seabed": 0.1, "aerial_bomb": 1.0, "sphere_mine": 1.0, "rock": 0.6, "tyre": 0.5}
COLOURS = {
    "aerial_bomb": (0.28, 0.22, 0.16), "sphere_mine": (0.20, 0.22, 0.20),
    "rock": (0.42, 0.41, 0.38), "tyre": (0.05, 0.05, 0.05),
}


class SimSession:
    def __init__(self, sonar: SonarParams = SonarParams(), flip_azimuth: bool = False, warmup: int = 20,
                 sun: float = 300.0, dome: float = 80.0, sonar_norm: str = "all",
                 sonar_gamma: float = 1.0, sonar_gain: float = 8.0):
        self._gamma = sonar_gamma
        self.sonar_gain = sonar_gain
        self._sun, self._dome, self._sonar_norm = sun, dome, sonar_norm
        self._materials = {}
        from isaacsim import SimulationApp

        self.app = SimulationApp({"headless": True, "width": CAM_RES[0], "height": CAM_RES[1]})
        self.p = sonar
        self.flip_azimuth = flip_azimuth
        self.murk = 0.0
        self.world = {"objects": []}
        self._prims = {}
        self._warmup = warmup

        import isaacsim

        sub = os.path.join(OCEANSIM, "isaacsim")
        if sub not in list(isaacsim.__path__):
            isaacsim.__path__.append(sub)

        import omni.usd
        from isaacsim.core.api import World
        from pxr import UsdLux

        self.World = World
        self.world_api = World(stage_units_in_meters=1.0)
        self.stage = omni.usd.get_context().get_stage()
        self._sun_light = UsdLux.DistantLight.Define(self.stage, "/World/Sun")
        self._dome_light = UsdLux.DomeLight.Define(self.stage, "/World/Dome")
        self.set_lights(self._sun, self._dome)
        self._build_seabed()
        self._make_sensors()

    # ---------- scene ----------
    def _label(self, prim, reflectivity: float, cls: str):
        from isaacsim.core.utils.semantics import add_labels

        add_labels(prim, labels=[f"{reflectivity}"], instance_name="reflectivity")
        add_labels(prim, labels=[cls], instance_name="class")

    def _build_seabed(self, half: float = 45.0, res: float = 0.5, seed: int = 0):
        from pxr import Gf, UsdGeom, Vt

        rng = np.random.default_rng(seed)
        xs = np.arange(-half + 16, half + 16 + 1e-6, res)
        n = len(xs)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        # Smooth long-wave undulation only: per-vertex height noise caused swirl artefacts in the sonar.
        Z = 0.05 * np.sin(X * 0.21) * np.cos(Y * 0.17)
        # Sand texture for the camera via per-vertex colour: ripples + patches + grain.
        ripples = 0.5 + 0.5 * np.sin(X * 3.1 + 0.8 * np.sin(Y * 0.4))
        patches = 0.5 + 0.5 * np.sin(X * 0.33 + 1.3) * np.sin(Y * 0.29 + 0.4)
        shade = 0.78 + 0.12 * ripples + 0.12 * patches + 0.05 * rng.standard_normal(X.shape)
        base = np.array([0.58, 0.52, 0.40])
        cols = [Gf.Vec3f(*(base * s)) for s in np.clip(shade, 0.5, 1.2).ravel()]
        pts = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel())]
        idx = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                idx += [a, a + n, a + n + 1, a + 1]
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/Seabed")
        mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * ((n - 1) ** 2)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray(cols))
        self._bind(mesh.GetPrim(), self._material("seabed", primvar=True))
        self._label(mesh.GetPrim(), REFLECTIVITY["seabed"], "seabed")

    def set_lights(self, sun: float, dome: float):
        self._sun_light.CreateIntensityAttr().Set(float(sun))
        self._dome_light.CreateIntensityAttr().Set(float(dome))

    def _material(self, name: str, rgb=None, primvar: bool = False, roughness: float = 0.9):
        """UsdPreviewSurface material; with primvar=True the diffuse colour comes from per-vertex displayColor."""
        if name in self._materials:
            return self._materials[name]
        from pxr import Gf, Sdf, UsdShade

        root = f"/World/Looks/{name}"
        mat = UsdShade.Material.Define(self.stage, root)
        sh = UsdShade.Shader.Define(self.stage, f"{root}/PBR")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        if primvar:
            rd = UsdShade.Shader.Define(self.stage, f"{root}/colour")
            rd.CreateIdAttr("UsdPrimvarReader_float3")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
            rd.CreateOutput("result", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self._materials[name] = mat
        return mat

    def _bind(self, prim, mat):
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def _gprim(self, kind: str, path: str, cls: str, t=(0, 0, 0), s=(1, 1, 1), r=None, **attrs):
        from pxr import Gf, UsdGeom

        g = getattr(UsdGeom, kind).Define(self.stage, path)
        for k, v in attrs.items():
            getattr(g, f"Create{k}Attr")(v)
        g.AddTranslateOp().Set(Gf.Vec3d(*t))
        if r is not None:
            g.AddRotateXYZOp().Set(Gf.Vec3f(*r))
        g.AddScaleOp().Set(Gf.Vec3f(*s))
        g.CreateDisplayColorAttr([Gf.Vec3f(*COLOURS[cls])])
        self._bind(g.GetPrim(), self._material(cls, rgb=COLOURS[cls], roughness=0.6))
        self._label(g.GetPrim(), REFLECTIVITY[cls], cls)
        return g

    def _add_object(self, obj: dict):
        from pxr import Gf, UsdGeom

        cls, oid = obj["cls"], obj["id"]
        root = f"/World/Objects/{oid}"
        xf = UsdGeom.Xform.Define(self.stage, root)
        x, y, z = obj["xyz"]
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xf.AddRotateZOp().Set(math.degrees(obj["yaw"]))
        if cls == "aerial_bomb":
            r = 0.22
            self._gprim("Cylinder", f"{root}/body", cls, t=(0.1, 0, r), Axis="X", Radius=r, Height=1.2)
            self._gprim("Sphere", f"{root}/nose", cls, t=(0.7, 0, r), s=(1.6, 1, 1), Radius=r)
            self._gprim("Cylinder", f"{root}/tail", cls, t=(-0.65, 0, r), Axis="X", Radius=0.12, Height=0.35)
            self._gprim("Cube", f"{root}/fin_v", cls, t=(-0.8, 0, r), s=(0.25, 0.02, 0.5), Size=1.0)
            self._gprim("Cube", f"{root}/fin_h", cls, t=(-0.8, 0, r), s=(0.25, 0.5, 0.02), Size=1.0)
        elif cls == "sphere_mine":
            self._gprim("Sphere", f"{root}/body", cls, t=(0, 0, 0.45), Radius=0.45)
            for k, (a, e) in enumerate([(0, 30), (90, 30), (180, 30), (270, 30), (0, 90)]):
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                ce, se = math.cos(math.radians(e)), math.sin(math.radians(e))
                pos = (0.5 * ce * ca, 0.5 * ce * sa, 0.45 + 0.5 * se)
                self._gprim("Cylinder", f"{root}/horn{k}", cls, t=pos, r=(0, 90 - e, a), Axis="Z",
                            Radius=0.035, Height=0.16)
        elif cls == "rock":
            rng = random.Random(oid)
            self._gprim("Sphere", f"{root}/a", cls, t=(0, 0, 0.18), s=(0.8, 0.55, 0.42), Radius=1.0)
            self._gprim("Sphere", f"{root}/b", cls, t=(rng.uniform(0.2, 0.45), rng.uniform(-0.2, 0.2), 0.15),
                        s=(0.45, 0.4, 0.35), Radius=1.0)
        elif cls == "tyre":
            n, R, rr = 16, 0.36, 0.12
            for k in range(n):
                a = 2 * math.pi * k / n
                self._gprim("Sphere", f"{root}/s{k}", cls, t=(R * math.cos(a), R * math.sin(a), rr),
                            s=(1.3, 1.3, 1.0), Radius=rr)
        self._prims[oid] = xf

    def load_world(self, world: dict):
        """Build all objects from a world.json dict. Uses P3's sim/scene.py if present."""
        self.world = {**world, "objects": [dict(o) for o in world["objects"]]}
        try:
            from sim import scene  # P3

            scene.build_scene(self.stage, self.world)
            print("[session] scene built by sim/scene.py", flush=True)
        except ImportError:
            for obj in self.world["objects"]:
                self._add_object(obj)
            print(f"[session] placeholder scene: {len(self.world['objects'])} objects", flush=True)
        if world.get("scenario") == "murky":
            self.murk = 0.7

    # ---------- sensors ----------
    def _make_sensors(self):
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
        from isaacsim.oceansim.sensors.UW_Camera import UW_Camera

        self._q = euler_angles_to_quat
        p = self.p
        self.sonar = ImagingSonarSensor(
            prim_path="/World/rig_sonar", position=np.array([0, 0, p.altitude]),
            orientation=self._q(np.array([0.0, p.pitch, 0.0]), degrees=True),
            min_range=p.min_range, max_range=p.max_range, hori_fov=p.hori_fov, vert_fov=p.vert_fov,
            angular_res=0.5, hori_res=1000)
        self.sonar.initialize()
        self.cam = UW_Camera(
            prim_path="/World/rig_cam", resolution=CAM_RES, position=np.array([0, 0, p.altitude - 0.2]),
            orientation=self._q(np.array([0.0, CAM_PITCH, 0.0]), degrees=True))
        self.world_api.reset()
        # "all" = normalise over the whole image; the default "range" flattened object/seabed contrast.
        self.sonar.sonar_initialize(normalizing_method=self._sonar_norm, viewport=False, include_unlabelled=False)
        # OceanSim's default water params (custom ones washed the image out to flat teal).
        self.cam.initialize(viewport=False)
        for _ in range(self._warmup):
            self.world_api.step(render=True)

    def move_to(self, x: float, y: float, yaw: float, altitude: float | None = None):
        """Teleport the sensor rig. yaw in radians, 0 = +x, CCW."""
        alt = self.p.altitude if altitude is None else altitude
        deg = math.degrees(yaw)
        self.sonar.set_world_pose(position=np.array([x, y, alt]),
                                  orientation=self._q(np.array([0.0, self.p.pitch, deg]), degrees=True))
        cx, cy = x - 0.1 * math.cos(yaw), y - 0.1 * math.sin(yaw)
        self.cam.set_world_pose(position=np.array([cx, cy, alt - 0.2]),
                                orientation=self._q(np.array([0.0, CAM_PITCH, deg]), degrees=True))

    def observe(self, settle_steps: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Render and return (sonar 512x512 uint8 fan, camera 360x640x3 uint8)."""
        for _ in range(settle_steps):
            self.world_api.step(render=True)
        self.sonar.make_sonar_data(intensity_gain=self.sonar_gain)
        polar = self.sonar.get_sonar_image()
        polar = polar.numpy() if hasattr(polar, "numpy") else np.asarray(polar)
        fan = polar_to_fan(polar, self.p, flip_azimuth=self.flip_azimuth, gamma=self._gamma)
        self.cam.render()
        frame = getattr(self.cam, "_uw_frame", None)
        if frame is None:
            cam = np.zeros((CAM_RES[1], CAM_RES[0], 3), np.uint8)
        else:
            cam = np.asarray(frame)[..., :3].astype(np.uint8)
        return fan, self._apply_murk(cam)

    def _apply_murk(self, cam: np.ndarray) -> np.ndarray:
        if self.murk <= 0:
            return cam
        from PIL import Image, ImageFilter

        a = min(self.murk, 0.95)
        img = Image.fromarray(cam).filter(ImageFilter.GaussianBlur(radius=1 + 4 * a))
        out = np.asarray(img, np.float32) * (1 - 0.85 * a) + WATER_RGB * 0.85 * a
        out += np.random.normal(0, 6 * a, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------- mid-run changes ----------
    def apply_change(self, event: dict):
        """Change the scene mid-run. Updates self.world (ground truth) to match.

        {"type": "add_object", "object": {world.json object}}
        {"type": "remove_object", "id": "o3"}
        {"type": "set_burial", "id": "o3", "burial": 0.2}
        {"type": "set_murk", "level": 0.0..1.0}
        """
        from pxr import Gf

        kind = event["type"]
        if kind == "add_object":
            obj = dict(event["object"])
            self.world["objects"].append(obj)
            self._add_object(obj)
        elif kind == "remove_object":
            self.stage.RemovePrim(f"/World/Objects/{event['id']}")
            self._prims.pop(event["id"], None)
            self.world["objects"] = [o for o in self.world["objects"] if o["id"] != event["id"]]
        elif kind == "set_burial":
            obj = next(o for o in self.world["objects"] if o["id"] == event["id"])
            obj["burial"] = float(event["burial"])
            obj["xyz"][2] = -obj["burial"] * CLASS_SPECS[obj["cls"]]["height"]
            self._prims[obj["id"]].GetOrderedXformOps()[0].Set(Gf.Vec3d(*obj["xyz"]))
        elif kind == "set_murk":
            self.murk = float(event["level"])
        else:
            raise ValueError(f"unknown change type {kind}")
        print(f"[session] change applied: {event}", flush=True)

    def close(self):
        self.app.close()
