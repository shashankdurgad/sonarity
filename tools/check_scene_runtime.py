"""Run check_runtime(stage, labeler) inside an ALREADY INITIALISED Isaac session.

labeler must be the real P2 semantic callback (prim, class_name, object_id).
This check creates a dedicated temporary subtree; it does not start Isaac.
"""
import math
from sim.scene import DIMENSIONS_M, add_object, build_scene, make_world


def check_runtime(stage, semantic_labeler, update=None):
    from pxr import Usd, UsdGeom
    if not callable(semantic_labeler):
        raise ValueError("Pass the actual simulator's semantic label callback")
    root = "/World/SceneIntegrationCheck"
    if stage.GetPrimAtPath(root):
        raise ValueError("Diagnostic subtree already exists; choose a clean stage or remove it deliberately")
    later_path = "/World/SurveySite/Objects/o7"
    if stage.GetPrimAtPath(later_path):
        raise ValueError("o7 already exists in the live scene; do not replace it for this diagnostic")
    existing_parent = stage.GetPrimAtPath("/World/SurveySite/Objects")
    existing_paths = [str(p.GetPath()) for p in existing_parent.GetChildren()] if existing_parent else []
    calls = []
    def label(prim, cls, oid):
        semantic_labeler(prim, cls, oid)
        calls.append(oid)
    try:
        world = make_world(10)
        paths = build_scene(stage, world, root_path=root, semantic_labeler=label, include_seabed=False)
        if stage.GetPrimAtPath(root+"/Seabed"):
            raise AssertionError("include_seabed=False still created a floor")
        # Deliberately add after initial construction, including an ID beyond o6.
        later = {"id":"o7", "cls":"tyre", "xyz":[31.0,31.0,0.0], "yaw":0.0, "burial":0.0}
        path = add_object(stage, later, parent_path="/World/SurveySite/Objects", semantic_labeler=label)
        if not stage.GetPrimAtPath(path) or "o7" not in calls:
            raise AssertionError("Later object or actual label callback failed")
        if any(not stage.GetPrimAtPath(p) for p in list(paths.values())+existing_paths):
            raise AssertionError("Adding o7 removed prior objects")
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.IsA(UsdGeom.Gprim):
                color = UsdGeom.Gprim(prim).GetDisplayColorPrimvar()
                if color.GetInterpolation() != UsdGeom.Tokens.constant or len(color.Get()) != 1:
                    raise AssertionError(f"Invalid displayColor at {prim.GetPath()}")
        # Check the authored bounds for every object class with zero rotation.
        for i, cls in enumerate(DIMENSIONS_M):
            obj = {"id":f"o{20+i}","cls":cls,"xyz":[0.0,0.0,0.0],"yaw":0.0,"burial":0.0}
            p = add_object(stage, obj, parent_path=root+"/Bounds", semantic_labeler=label)
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
            bound = cache.ComputeWorldBound(stage.GetPrimAtPath(p)).ComputeAlignedRange()
            size = bound.GetSize()
            expected = DIMENSIONS_M[cls]
            for actual, key in zip(size, ("length","width","height")):
                if not math.isclose(actual, expected[key], abs_tol=1e-4):
                    raise AssertionError(f"{cls} {key}: USD {actual} != table {expected[key]}")
        if update is not None:
            update()
        print("PASS: USD constant color, seabed opt-out, geometry bounds and mid-session o7 with real label callback")
        print("A real sonar capture is still needed to verify visibility/pointcloud output.")
    finally:
        if stage.GetPrimAtPath(later_path):
            stage.RemovePrim(later_path)
        stage.RemovePrim(root)
