"""CPU contract checks for compact training scenes; not a sonar runtime test."""
import inspect
import json
import math
import sys
from sim.scene import CLASSES, DIMENSIONS_M, HEIGHTS_M, MIN_SPACING_M, add_object, build_scene, make_world, validate_world


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_checks():
    counts, class_orders = set(), set()
    for seed in range(1, 501):
        worlds = {scenario: make_world(seed, scenario) for scenario in ("clear_sand", "half_buried", "murky")}
        for scenario, world in worlds.items():
            validate_world(world)
            require(world == make_world(seed, scenario), "Non-deterministic generation")
            objects = world["objects"]
            require(4 <= len(objects) <= 6, "Object count outside 4-6")
            classes = {obj["cls"] for obj in objects}
            require(bool(classes & {"aerial_bomb", "sphere_mine"}), "No mine-like object")
            require(bool(classes & {"rock", "tyre"}), "No other object")
            for obj in objects:
                require(math.isclose(obj["xyz"][2], -HEIGHTS_M[obj["cls"]]*obj["burial"], abs_tol=2e-6), "Burial/z disagreement")
                if seed != 1:
                    require(all(0 <= v <= 32 for v in obj["xyz"][:2]), "Outside compact survey area")
            for i, a in enumerate(objects):
                for b in objects[i+1:]:
                    require(math.hypot(a["xyz"][0]-b["xyz"][0], a["xyz"][1]-b["xyz"][1]) >= MIN_SPACING_M, "Object centres are less than 6 m apart")
        def layout(world):
            return [(o["id"], o["cls"], o["xyz"][:2], o["yaw"]) for o in world["objects"]]
        require(layout(worlds["clear_sand"]) == layout(worlds["half_buried"]) == layout(worlds["murky"]), "Scenario changed layout")
        require(worlds["clear_sand"]["objects"] == worlds["murky"]["objects"], "Murky object definitions differ")
        counts.add(len(worlds["clear_sand"]["objects"]))
        class_orders.add(tuple(o["cls"] for o in worlds["clear_sand"]["objects"]))
    require(counts == {4, 5, 6} and len(class_orders) > 10, "Insufficient generation variety")
    require(make_world(1)["objects"][0]["xyz"] == [20.0, 30.0, -0.1], "Legacy seed 1 changed")
    require(make_world(1)["objects"][3]["xyz"] == [70.0, 25.0, -0.3], "Legacy seed 1 changed")
    require(set(DIMENSIONS_M) == set(CLASSES), "Dimensions missing class")
    for cls, values in DIMENSIONS_M.items():
        require(set(values) == {"length", "width", "height"}, "Dimension keys changed")
        require(all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in values.values()), "Invalid dimension")
        require(HEIGHTS_M[cls] == values["height"], "Height constants inconsistent")
    require(inspect.signature(build_scene).parameters["include_seabed"].default is True, "Seabed opt-out missing")
    later = {"id": "o7", "cls": "tyre", "xyz": [5.0, 5.0, 0.0], "yaw": 0.0, "burial": 0.0}
    validate_world({"seed": 1, "scenario": "clear_sand", "objects": [later]})
    require(inspect.signature(add_object).parameters["parent_path"].default == "/World/SurveySite/Objects", "Object parent API changed")
    print("PASS: 500 seeds x 3 scenarios; compact bounds (except legacy seed 1), >=6m centre spacing, 4-6 varied objects, class balance, reproducibility and burial")
    print("PASS: dimensions/height tables, include_seabed default, unchanged object parent and o7 data acceptance")
    print("NOT CHECKED: actual USD geometry, live label callback, sonar rendering or training annotations")


if __name__ == "__main__":
    try:
        run_checks()
    except (ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
