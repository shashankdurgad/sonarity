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
CHASE_BACK, CHASE_UP, CHASE_PITCH = 4.5, 2.2, 28.0  # chase camera: metres behind / above the rig, deg down
WATER_RGB = np.array([18, 62, 74], dtype=np.float32)  # murk colour (dark teal)
REFLECTIVITY = {"seabed": 0.1, "aerial_bomb": 1.0, "sphere_mine": 1.0, "rock": 0.6, "tyre": 0.5}
COLOURS = {
    "aerial_bomb": (0.28, 0.22, 0.16), "sphere_mine": (0.20, 0.22, 0.20),
    "rock": (0.42, 0.41, 0.38), "tyre": (0.05, 0.05, 0.05),
}


DEFAULT_SEABED = (-30.0, 110.0, -30.0, 90.0)  # covers the placeholder area and P3's x 20-70 layout


def seabed_bounds_for(world: dict, margin: float = 25.0) -> tuple[float, float, float, float]:
    """Seabed extent for a world: objects bbox + survey margin (4) + lead-in (6) + sonar range (10) + spare."""
    xs = [o["xyz"][0] for o in world["objects"]] or [16.0]
    ys = [o["xyz"][1] for o in world["objects"]] or [16.0]
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


class SimSession:
    def __init__(self, sonar: SonarParams = SonarParams(), flip_azimuth: bool = False, warmup: int = 20,
                 sun: float = 300.0, dome: float = 80.0, sonar_norm: str = "all",
                 sonar_gamma: float = 1.0, sonar_gain: float = 8.0, chase: bool = True,
                 seabed_bounds: tuple | None = None):
        self._init_seabed_bounds = tuple(seabed_bounds) if seabed_bounds else DEFAULT_SEABED
        self._chase_enabled = chase
        self.chase_cam = None
        self._auv = None
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
        self._prims = {}  # object id -> prim path
        self._scene = None
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
        self._build_seabed(bounds=self._init_seabed_bounds)
        self._make_sensors()

    # ---------- scene ----------
    def _label(self, prim, reflectivity: float, cls: str):
        from isaacsim.core.utils.semantics import add_labels

        add_labels(prim, labels=[f"{reflectivity}"], instance_name="reflectivity")
        add_labels(prim, labels=[cls], instance_name="class")

    def _build_seabed(self, bounds=DEFAULT_SEABED, max_verts: int = 300, seed: int = 0):
        """Seabed mesh over bounds = (x0, x1, y0, y1). Must cover everything the sonar can see:
        if the sonar looks past the edge it gets an empty point cloud and OceanSim crashes."""
        from pxr import Gf, UsdGeom, Vt

        if self.stage.GetPrimAtPath("/World/Seabed"):
            self.stage.RemovePrim("/World/Seabed")
        self._seabed_bounds = bounds
        rng = np.random.default_rng(seed)
        x0, x1, y0, y1 = bounds
        res = max(0.5, max(x1 - x0, y1 - y0) / (max_verts - 1))
        xs = np.arange(x0, x1 + 1e-6, res)
        ys = np.arange(y0, y1 + 1e-6, res)
        nx, n = len(xs), len(ys)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
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
        for i in range(nx - 1):
            for j in range(n - 1):
                a = i * n + j
                idx += [a, a + n, a + n + 1, a + 1]
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/Seabed")
        mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * ((nx - 1) * (n - 1))))
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
        self._prims[oid] = root

    def _scene_labeler(self, prim, cls, oid):
        """Callback for P3's sim/scene.py: the sonar only sees prims with a reflectivity label."""
        from pxr import UsdGeom

        # A single displayColor on a mesh without interpolation='constant' is read as per-vertex and
        # corrupts the pointcloud annotator, crashing the sonar when the object enters view.
        gp = UsdGeom.Gprim(prim)
        if gp:
            pv = gp.GetDisplayColorPrimvar()
            if pv and pv.HasValue() and len(pv.Get() or []) == 1:
                pv.SetInterpolation(UsdGeom.Tokens.constant)
        self._label(prim, REFLECTIVITY[cls], cls)

    def _scene_add(self, obj: dict):
        """Add one object via P3's scene.add_object (if scene.py exists) or the placeholder builder."""
        if self._scene is None:
            self._add_object(obj)
        else:
            self._prims[obj["id"]] = self._scene.add_object(
                self.stage, obj, parent_path="/World/SurveySite/Objects", semantic_labeler=self._scene_labeler)

    def load_world(self, world: dict):
        """Build all objects from a world.json dict. Uses P3's sim/scene.py if present."""
        self.world = {**world, "objects": [dict(o, xyz=list(o["xyz"])) for o in world["objects"]]}
        need, b = seabed_bounds_for(self.world), self._seabed_bounds
        if need[0] < b[0] or need[1] > b[1] or need[2] < b[2] or need[3] > b[3]:
            # Rebuilding the seabed after the sonar starts breaks OceanSim's point cloud (all frames go
            # blank), so we only warn. Pass seabed_bounds=seabed_bounds_for(world) to SimSession instead.
            print(f"[session] WARNING world needs seabed {need} but it is {b}; sonar may go blank near "
                  f"the edges. Create SimSession(seabed_bounds=seabed_bounds_for(world)).", flush=True)
        try:
            from sim import scene  # P3
        except ImportError:
            scene = None
        self._scene = scene
        if scene is None:
            for obj in self.world["objects"]:
                self._add_object(obj)
            print(f"[session] placeholder scene: {len(self.world['objects'])} objects", flush=True)
        else:
            try:  # our textured, sonar-labelled seabed replaces scene.py's flat box
                paths = scene.build_scene(self.stage, self.world, semantic_labeler=self._scene_labeler,
                                          include_seabed=False)
            except TypeError:  # older scene.py without include_seabed
                paths = scene.build_scene(self.stage, self.world, semantic_labeler=self._scene_labeler)
                self.stage.RemovePrim("/World/SurveySite/Seabed")
            self._prims.update(paths)
            print(f"[session] scene built by sim/scene.py: {len(paths)} objects", flush=True)
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
        if self._chase_enabled:
            self._build_auv()
            self.chase_cam = UW_Camera(
                prim_path="/World/chase_cam", resolution=CAM_RES,
                position=np.array([-CHASE_BACK, 0, p.altitude + CHASE_UP]),
                orientation=self._q(np.array([0.0, CHASE_PITCH, 0.0]), degrees=True))
        self.world_api.reset()
        # "all" = normalise over the whole image; the default "range" flattened object/seabed contrast.
        self.sonar.sonar_initialize(normalizing_method=self._sonar_norm, viewport=False, include_unlabelled=False)
        # OceanSim's default water params (custom ones washed the image out to flat teal).
        self.cam.initialize(viewport=False)
        if self.chase_cam is not None:
            self.chase_cam.initialize(viewport=False)
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
        if self._auv is not None:
            from pxr import Gf

            ops = self._auv.GetOrderedXformOps()
            ops[0].Set(Gf.Vec3d(x, y, alt - 0.1))
            ops[1].Set(deg)
        if self.chase_cam is not None:
            bx, by = x - CHASE_BACK * math.cos(yaw), y - CHASE_BACK * math.sin(yaw)
            self.chase_cam.set_world_pose(position=np.array([bx, by, alt + CHASE_UP]),
                                          orientation=self._q(np.array([0.0, CHASE_PITCH, deg]), degrees=True))

    def _build_auv(self):
        """Yellow torpedo AUV whose nose sits at the sensor rig. Not labelled, so the sonar ignores it."""
        from pxr import Gf, UsdGeom

        root = "/World/AUV"
        self._auv = UsdGeom.Xform.Define(self.stage, root)
        self._auv.AddTranslateOp().Set(Gf.Vec3d(0, 0, self.p.altitude - 0.1))
        self._auv.AddRotateZOp().Set(0.0)
        yellow = self._material("auv_yellow", rgb=(0.95, 0.72, 0.08), roughness=0.4)
        dark = self._material("auv_dark", rgb=(0.08, 0.08, 0.09), roughness=0.5)

        def part(kind, name, mat, t, s=(1, 1, 1), **attrs):
            g = getattr(UsdGeom, kind).Define(self.stage, f"{root}/{name}")
            for k, v in attrs.items():
                getattr(g, f"Create{k}Attr")(v)
            g.AddTranslateOp().Set(Gf.Vec3d(*t))
            g.AddScaleOp().Set(Gf.Vec3f(*s))
            self._bind(g.GetPrim(), mat)

        part("Cylinder", "hull", yellow, (-0.9, 0, 0), Axis="X", Radius=0.17, Height=1.4)
        part("Sphere", "nose", dark, (-0.2, 0, 0), s=(1.5, 1, 1), Radius=0.17)
        part("Cone", "tail", yellow, (-1.75, 0, 0), s=(-1, 1, 1), Axis="X", Radius=0.17, Height=0.3)
        part("Cube", "fin_v", dark, (-1.65, 0, 0), s=(0.22, 0.015, 0.55), Size=1.0)
        part("Cube", "fin_h", dark, (-1.65, 0, 0), s=(0.22, 0.55, 0.015), Size=1.0)
        part("Cylinder", "prop", dark, (-1.95, 0, 0), Axis="X", Radius=0.1, Height=0.05)
        part("Cylinder", "mast", dark, (-0.5, 0, 0.2), Axis="Z", Radius=0.025, Height=0.15)

    def chase(self) -> np.ndarray | None:
        """Third-person view of the AUV (360x640x3 uint8) from the latest render; None if disabled."""
        if self.chase_cam is None:
            return None
        self.chase_cam.render()
        frame = getattr(self.chase_cam, "_uw_frame", None)
        if frame is None:
            return np.zeros((CAM_RES[1], CAM_RES[0], 3), np.uint8)
        return self._apply_murk(np.asarray(frame)[..., :3].astype(np.uint8))

    def observe(self, settle_steps: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Render and return (sonar 512x512 uint8 fan, camera 360x640x3 uint8)."""
        for _ in range(settle_steps):
            self.world_api.step(render=True)
        try:
            self.sonar.make_sonar_data(intensity_gain=self.sonar_gain)
            polar = self.sonar.get_sonar_image()
            polar = polar.numpy() if hasattr(polar, "numpy") else np.asarray(polar)
            fan = polar_to_fan(polar, self.p, flip_azimuth=self.flip_azimuth, gamma=self._gamma)
        except KeyError as e:  # empty point cloud (nothing labelled in view): OceanSim's annotator throws
            print(f"[session] sonar saw nothing ({e}); returning a blank frame", flush=True)
            fan = np.zeros((512, 512), np.uint8)
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
    def reserve(self, counts: dict, park_xy=(0.0, 0.0)):
        """Pre-create hidden objects for mid-run 'reveal' events, e.g. reserve({"aerial_bomb": 2}).

        Call right after load_world and BEFORE the first observe(). Objects created after rendering
        starts never get their semantic labels registered, so OceanSim sees them as unlabelled: they
        cast a sonar shadow but give no echo. Reserved objects are parked 3 m under the seabed (hidden
        from sonar and camera) and are not in self.world until revealed.
        """
        self._reserve = getattr(self, "_reserve", {})
        nid = max([int(o["id"][1:]) for o in self.world["objects"]] + [0]) + 1
        for cls, n in counts.items():
            for _ in range(n):
                obj = {"id": f"o{nid}", "cls": cls, "xyz": [float(park_xy[0]), float(park_xy[1]), -3.0],
                       "yaw": 0.0, "burial": 0.0}
                nid += 1
                self._scene_add(obj)
                self._reserve.setdefault(cls, []).append(obj)
        print(f"[session] reserved hidden objects: { {k: len(v) for k, v in self._reserve.items()} }", flush=True)

    def apply_change(self, event: dict):
        """Change the scene mid-run. Updates self.world (ground truth) to match.

        {"type": "reveal_object", "cls": "aerial_bomb", "xy": [x, y], "yaw": 0.3, "burial": 0.2}  # needs reserve()
        {"type": "add_object", "object": {world.json object}}   # WARNING: sonar won't echo it (see reserve)
        {"type": "remove_object", "id": "o3"}
        {"type": "set_burial", "id": "o3", "burial": 0.2}
        {"type": "set_murk", "level": 0.0..1.0}
        Returns the affected world object (or None).
        """
        from pxr import Gf, UsdGeom

        kind = event["type"]
        if kind == "reveal_object":
            pool = getattr(self, "_reserve", {}).get(event["cls"], [])
            if not pool:
                raise ValueError(f"no reserved {event['cls']} left; call reserve() before the first observe()")
            obj = pool.pop(0)
            obj["burial"] = float(event.get("burial", 0.0))
            obj["yaw"] = float(event.get("yaw", 0.0))
            obj["xyz"] = [float(event["xy"][0]), float(event["xy"][1]),
                          -obj["burial"] * CLASS_SPECS[obj["cls"]]["height"]]
            ops = UsdGeom.Xformable(self.stage.GetPrimAtPath(self._prims[obj["id"]])).GetOrderedXformOps()
            ops[0].Set(Gf.Vec3d(*obj["xyz"]))
            ops[1].Set(math.degrees(obj["yaw"]))
            self.world["objects"].append(obj)
            print(f"[session] change applied: {event} -> {obj['id']}", flush=True)
            return obj
        if kind == "add_object":
            obj = dict(event["object"], xyz=list(event["object"]["xyz"]))
            self.world["objects"].append(obj)
            self._scene_add(obj)
        elif kind == "remove_object":
            path = self._prims.pop(event["id"], None)
            if path:
                self.stage.RemovePrim(path)
            self.world["objects"] = [o for o in self.world["objects"] if o["id"] != event["id"]]
        elif kind == "set_burial":
            obj = next(o for o in self.world["objects"] if o["id"] == event["id"])
            obj["burial"] = float(event["burial"])
            obj["xyz"][2] = -obj["burial"] * CLASS_SPECS[obj["cls"]]["height"]
            xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(self._prims[obj["id"]]))
            xf.GetOrderedXformOps()[0].Set(Gf.Vec3d(*obj["xyz"]))
        elif kind == "set_murk":
            self.murk = float(event["level"])
        else:
            raise ValueError(f"unknown change type {kind}")
        print(f"[session] change applied: {event}", flush=True)

    def close(self):
        self.app.close()
