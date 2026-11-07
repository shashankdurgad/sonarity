"""P3 scene data + OpenUSD geometry; no Isaac imports at module-import time.

Laptop: python -m sim.scene --seed 1 --scenario clear_sand --out runs/p3_scene/world.json
Isaac:  build_scene(stage, world) AFTER P2 starts SimulationApp.

Coordinates: metres, Z-up, yaw radians about +Z. xyz[2] is the object's
BOTTOM-anchor offset from a flat seabed, NOT water-surface depth. Burial is
metadata; it is converted to z by make_world and is not applied a second time.
No sensors, lighting, hydrodynamics, sonar labels or camera turbidity are created.
"""
import argparse
import json
import math
from pathlib import Path
import random

CLASSES = ("aerial_bomb", "sphere_mine", "rock", "tyre")
SCENARIOS = ("clear_sand", "half_buried", "murky")
# Internal geometry settings; these do not add fields to world.json.
HEIGHTS_M = {"aerial_bomb": 0.5, "sphere_mine": 0.75, "rock": 0.7, "tyre": 0.24}


def validate_world(world):
    """Reject malformed scene data before constructing USD prims."""
    if not isinstance(world, dict) or set(world) != {"seed", "scenario", "objects"}:
        raise ValueError("world must have exactly seed, scenario, objects")
    if type(world["seed"]) is not int or world["scenario"] not in SCENARIOS:
        raise ValueError("seed must be an integer and scenario must be supported")
    if not isinstance(world["objects"], list) or not world["objects"]:
        raise ValueError("objects must be a nonempty list")
    seen = set()
    for obj in world["objects"]:
        if not isinstance(obj, dict) or set(obj) != {"id", "cls", "xyz", "yaw", "burial"}:
            raise ValueError("object must have id, cls, xyz, yaw, burial")
        oid = obj["id"]
        if not isinstance(oid, str) or not oid.startswith("o") or not oid[1:].isdigit() or oid in seen:
            raise ValueError("IDs must be unique o1, o2, ...")
        seen.add(oid)
        if obj["cls"] not in CLASSES:
            raise ValueError(f"Unsupported class: {obj['cls']}")
        xyz = obj["xyz"]
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError("xyz must be a three-number list")
        values = xyz + [obj["yaw"], obj["burial"]]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
            raise ValueError("Coordinates, yaw and burial must be finite numbers")
        if not 0 <= obj["burial"] <= 1:
            raise ValueError("burial must be between 0 and 1")


def make_world(seed=1, scenario="clear_sand"):
    """Seed 1 clear_sand matches the supplied sample object's layout exactly.

    Other seeds jitter each anchor by up to 3 m and vary yaw. Every scene has
    four objects. Sensor range is ~10 m: P2 must actually pass near these anchors.
    murky changes metadata only; P2 owns applying camera turbidity.
    """
    if type(seed) is not int or scenario not in SCENARIOS:
        raise ValueError("Invalid seed or scenario")
    rng = random.Random(seed)
    anchors = [(20.0, 30.0), (40.0, 15.0), (55.0, 45.0), (70.0, 25.0)]
    classes = ["aerial_bomb", "rock", "tyre", "sphere_mine"]
    yaws = [0.5, 0.0, 1.2, 0.0]
    base_burial = [0.2, 0.0, 0.0, 0.4]
    objects = []
    for i, (cls, (x, y)) in enumerate(zip(classes, anchors)):
        yaw = yaws[i]
        if seed != 1:
            x += rng.uniform(-3, 3)
            y += rng.uniform(-3, 3)
            yaw = rng.uniform(-math.pi, math.pi)
        burial = rng.uniform(0.25, 0.55) if scenario == "half_buried" else base_burial[i]
        z = -HEIGHTS_M[cls] * burial
        objects.append({"id": f"o{i+1}", "cls": cls,
                        "xyz": [round(x, 6), round(y, 6), round(z, 6)],
                        "yaw": round(yaw, 6), "burial": round(burial, 6)})
    world = {"seed": seed, "scenario": scenario, "objects": objects}
    validate_world(world)
    return world


def save_world(world, path, *, overwrite=False):
    validate_world(world)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(world, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _usd():
    # Delayed intentionally: ordinary laptop commands need only Python stdlib.
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdShade
    except ImportError as exc:
        raise RuntimeError("Geometry needs P2's Isaac/OpenUSD runtime. Start SimulationApp first.") from exc
    return Gf, Sdf, UsdGeom, UsdShade


def _tag(prim, obj):
    """GT hooks only; these custom attributes are NOT a sonar annotation API."""
    _, Sdf, _, _ = _usd()
    prim.CreateAttribute("p3:object_id", Sdf.ValueTypeNames.String).Set(obj["id"])
    prim.CreateAttribute("p3:class", Sdf.ValueTypeNames.String).Set(obj["cls"])


def _paint(stage, shape, color):
    Gf, Sdf, _, UsdShade = _usd()
    shape.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mat = UsdShade.Material.Define(stage, str(shape.GetPath()) + "/Material")
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)


def _box(stage, path, size, centre, color):
    Gf, _, UsdGeom, _ = _usd()
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*centre))
    box.AddScaleOp().Set(Gf.Vec3f(*size))
    _paint(stage, box, color)
    return box


def _sphere(stage, path, radius, centre, color, scale=None):
    Gf, _, UsdGeom, _ = _usd()
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.AddTranslateOp().Set(Gf.Vec3d(*centre))
    if scale is not None:
        sphere.AddScaleOp().Set(Gf.Vec3f(*scale))
    _paint(stage, sphere, color)
    return sphere


def _tyre_mesh(stage, path, color):
    Gf, _, UsdGeom, _ = _usd()
    points, faces = [], []
    n, m, major, minor = 32, 12, 0.42, 0.12
    for i in range(n):
        u = 2 * math.pi * i / n
        for j in range(m):
            v = 2 * math.pi * j / m
            r = major + minor * math.cos(v)
            points.append(Gf.Vec3f(r*math.cos(u), r*math.sin(u), minor*(1+math.sin(v))))
    for i in range(n):
        for j in range(m):
            faces.extend([i*m+j, ((i+1)%n)*m+j, ((i+1)%n)*m+(j+1)%m, i*m+(j+1)%m])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4]*(n*m))
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    _paint(stage, mesh, color)
    return mesh


def add_object(stage, obj, *, parent_path="/World/P3Scene/Objects", seabed_z=0.0,
               scale_xyz=(1.0, 1.0, 1.0), semantic_labeler=None):
    """Create one visual proxy and return its root prim path.

    P5 may call this with a HYPOTHETICAL obj, rather than a real world.json row.
    semantic_labeler is an optional callback (prim, class_name, object_id) owned
    by P2; it must use the actual renderer's supported semantic-label mechanism.
    Scaling changes proxy dimensions; P5 computes scale from detected size.
    """
    validate_world({"seed": 1, "scenario": "clear_sand", "objects": [obj]})
    if len(scale_xyz) != 3 or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in scale_xyz):
        raise ValueError("scale_xyz must contain three positive finite numbers")
    if not math.isfinite(seabed_z):
        raise ValueError("seabed_z must be finite")
    Gf, _, UsdGeom, _ = _usd()
    UsdGeom.Xform.Define(stage, parent_path)
    path = f"{parent_path}/{obj['id']}"
    if stage.GetPrimAtPath(path):
        raise ValueError(f"Object already exists at {path}; rebuild the owned scene or use another parent")
    root = UsdGeom.Xform.Define(stage, path)
    x, y, z = obj["xyz"]
    root.AddTranslateOp().Set(Gf.Vec3d(x, y, seabed_z + z))
    root.AddRotateZOp().Set(math.degrees(obj["yaw"]))
    root.AddScaleOp().Set(Gf.Vec3f(*scale_xyz))
    _tag(root.GetPrim(), obj)
    color = (0.28, 0.30, 0.29)
    parts = []
    if obj["cls"] == "aerial_bomb":
        body = UsdGeom.Cylinder.Define(stage, path + "/Body")
        body.CreateAxisAttr("X")
        body.CreateRadiusAttr(0.25)
        body.CreateHeightAttr(1.5)
        body.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.25))
        _paint(stage, body, color)
        parts.append(body)
        parts.append(_box(stage, path+"/FinHorizontal", (0.4, 0.7, 0.04), (-0.65, 0, 0.25), color))
        parts.append(_box(stage, path+"/FinVertical", (0.4, 0.04, 0.5), (-0.65, 0, 0.25), color))
    elif obj["cls"] == "sphere_mine":
        parts.append(_sphere(stage, path+"/Body", 0.375, (0, 0, 0.375), color))
    elif obj["cls"] == "rock":
        # Intentionally simple smooth proxy; replace after first sensor render.
        parts.append(_sphere(stage, path+"/Body", 1.0, (0, 0, 0.35), color, (0.6, 0.45, 0.35)))
    else:
        parts.append(_tyre_mesh(stage, path+"/Body", color))
    for part in parts:
        _tag(part.GetPrim(), obj)
        if semantic_labeler is not None:
            semantic_labeler(part.GetPrim(), obj["cls"], obj["id"])
    return path


def build_scene(stage, world, *, root_path="/World/P3Scene", seabed_z=0.0, semantic_labeler=None):
    """Build a flat seabed + objects in a metre-scale, Z-up stage supplied by P2.

    The root must be dedicated to P3 assets. Rebuild removes only that subtree;
    put cameras/sensors elsewhere. Existing origin/lighting/camera setup is P2's.
    No rigid-body physics is added because this plan uses static scene objects.
    """
    validate_world(world)
    if not root_path.startswith("/World/") or root_path == "/World/" or root_path.endswith("/"):
        raise ValueError("Use a dedicated path such as /World/P3Scene")
    if not math.isfinite(seabed_z):
        raise ValueError("seabed_z must be finite")
    _, _, UsdGeom, _ = _usd()
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("P3 expects a Z-up stage; agree this with P2")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("P3 expects metresPerUnit=1; agree this with P2")
    if stage.GetPrimAtPath(root_path):
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)
    xs = [o["xyz"][0] for o in world["objects"]]
    ys = [o["xyz"][1] for o in world["objects"]]
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    sx, sy = max(100.0, max(xs)-min(xs)+40), max(80.0, max(ys)-min(ys)+40)
    _box(stage, root_path+"/Seabed", (sx, sy, 0.2), (cx, cy, seabed_z-0.1), (0.45, 0.39, 0.27))
    return {obj["id"]: add_object(stage, obj, parent_path=root_path+"/Objects",
                                  seabed_z=seabed_z, semantic_labeler=semantic_labeler)
            for obj in world["objects"]}


def main():
    parser = argparse.ArgumentParser(description="Write P3 world data without Isaac or GPU dependencies")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenario", choices=SCENARIOS, default="clear_sand")
    parser.add_argument("--out", type=Path, default=Path("runs/p3_scene/world.json"))
    parser.add_argument("--force", action="store_true", help="Overwrite this output file if it exists")
    args = parser.parse_args()
    world = make_world(args.seed, args.scenario)
    try:
        save_world(world, args.out, overwrite=args.force)
    except FileExistsError:
        parser.error(f"{args.out} already exists; use another output path or --force intentionally")
    print(f"Wrote {args.out} ({len(world['objects'])} objects, {world['scenario']}, seed {world['seed']})")
    print("Scene data only. P2 must render sonar/camera/poses; this is not a complete survey run.")


if __name__ == "__main__":
    main()
