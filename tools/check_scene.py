"""Focused local checks for scene generation; no GPU or third-party packages.

Run from repository root: python -m tools.check_scene
These checks do not exercise Isaac rendering, sonar labels, or model accuracy.
"""
import json
import math
import sys

from sim.scene import CLASSES, HEIGHTS_M, make_world, validate_world


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_checks():
    for seed in range(1, 51):
        worlds = {s: make_world(seed, s) for s in ("clear_sand", "half_buried", "murky")}
        for scenario, world in worlds.items():
            validate_world(world)
            require(world == make_world(seed, scenario), f"Non-reproducible seed {seed} / {scenario}")
            require({o["cls"] for o in world["objects"]} == set(CLASSES), "Missing object type")
            require(len(world["objects"]) == 4, "Expected the initial four-object scene")
            for obj in world["objects"]:
                require(math.isclose(obj["xyz"][2], -HEIGHTS_M[obj["cls"]]*obj["burial"], abs_tol=2e-6), "Burial and bottom-anchor z disagree")
            for i, a in enumerate(world["objects"]):
                for b in world["objects"][i+1:]:
                    distance = math.hypot(a["xyz"][0]-b["xyz"][0], a["xyz"][1]-b["xyz"][1])
                    require(distance > 3, "Initial scene objects are too close for independent inspection")
        def layout(world):
            return [(o["id"], o["cls"], o["xyz"][:2], o["yaw"]) for o in world["objects"]]
        require(layout(worlds["clear_sand"]) == layout(worlds["half_buried"]) == layout(worlds["murky"]), "Scenario changed positions/headings; comparison is confounded")
    sample = make_world(1)
    require(sample["objects"][0]["xyz"] == [20.0, 30.0, -0.1], "Sample anchor changed")
    require(sample["objects"][3]["xyz"] == [70.0, 25.0, -0.3], "Sample mine anchor changed")
    require(make_world(2) != make_world(3), "Different seeds did not produce variation")
    for mutate in (lambda w: w["objects"][0].update(id="o0"),
                   lambda w: w["objects"][1].update(id="o1"),
                   lambda w: w["objects"][0].update(xyz=[float("nan"), 0, 0]),
                   lambda w: w["objects"][0].update(burial=1.1)):
        invalid = json.loads(json.dumps(sample))
        mutate(invalid)
        try:
            validate_world(invalid)
        except ValueError:
            pass
        else:
            raise ValueError("Invalid scene was accepted")
    print("PASS: scene generation checked across 50 seeds and 3 scenarios")
    print("PASS: reproducibility, sample anchors, scenario-controlled layout, burial placement, spacing and invalid-input rejection")
    print("NOT CHECKED: Isaac geometry, acoustic realism, camera turbidity or sonar annotations")


if __name__ == "__main__":
    try:
        run_checks()
    except (ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
