"""Object specs, placeholder world generation, survey path and YOLO labels (owner: P2).

Pure Python + NumPy (no Isaac), so it runs and can be tested on a laptop.
`make_world` is a PLACEHOLDER until P3's sim/scene.py lands; it writes the world.json contract format.
"""
import math
import random

import numpy as np

from sim.sonar_geometry import SIZE, SonarParams, world_to_pixel

WATER_DEPTH = 20.0  # m; poses.csv depth = WATER_DEPTH - altitude

# Footprint (length along yaw, width) and height in metres; YOLO class 0 = mine_like, 1 = other.
CLASS_SPECS = {
    "aerial_bomb": {"length": 1.9, "width": 0.5, "height": 0.5, "yolo": 0},
    "sphere_mine": {"length": 1.0, "width": 1.0, "height": 1.0, "yolo": 0},
    "rock": {"length": 1.6, "width": 1.1, "height": 0.6, "yolo": 1},
    "tyre": {"length": 0.95, "width": 0.95, "height": 0.25, "yolo": 1},
}
YOLO_NAMES = ["mine_like", "other"]

AREA = (0.0, 32.0, 0.0, 32.0)  # x_min, x_max, y_min, y_max of the object field


def make_world(seed: int, scenario: str = "clear_sand", n_extra: int = 0) -> dict:
    """One of each class, spread out, plus `n_extra` random objects. world.json format (PLAN §11)."""
    rng = random.Random(seed)
    classes = ["aerial_bomb", "sphere_mine", "rock", "tyre"] + [rng.choice(list(CLASS_SPECS)) for _ in range(n_extra)]
    rng.shuffle(classes)
    objects, placed = [], []
    for i, cls in enumerate(classes, start=1):
        for _ in range(200):
            x = rng.uniform(AREA[0] + 3, AREA[1] - 3)
            y = rng.uniform(AREA[2] + 2, AREA[3] - 2)
            if all(math.hypot(x - px, y - py) > 6.0 for px, py in placed):
                break
        placed.append((x, y))
        if scenario == "half_buried":
            burial = rng.uniform(0.3, 0.6)
        else:
            burial = rng.uniform(0.0, 0.1)
        h = CLASS_SPECS[cls]["height"]
        objects.append({
            "id": f"o{i}", "cls": cls,
            "xyz": [round(x, 2), round(y, 2), round(-burial * h, 3)],
            "yaw": round(rng.uniform(-math.pi, math.pi), 3),
            "burial": round(burial, 2),
        })
    return {"seed": seed, "scenario": scenario, "objects": objects}


def lawnmower(lane_spacing: float = 8.0, step: float = 0.75, speed: float = 0.75,
              lead_in: float = 6.0, altitude: float = 2.5, n_lanes: int | None = None) -> list[dict]:
    """Boustrophedon path along ±x over AREA, with transit legs between lanes. Returns pose dicts."""
    x0, x1, y0, y1 = AREA
    lanes = list(np.arange(y0, y1 + 1e-6, lane_spacing))
    if n_lanes:
        lanes = lanes[:n_lanes]
    poses, t = [], 0.0
    dt = step / speed

    def add(x, y, yaw):
        nonlocal t
        poses.append({"t": round(t, 2), "x": round(float(x), 3), "y": round(float(y), 3),
                      "depth": round(WATER_DEPTH - altitude, 2), "yaw": round(float(yaw), 4)})
        t += dt

    for li, y in enumerate(lanes):
        fwd = li % 2 == 0
        xs = np.arange(x0 - lead_in, x1 + 1e-6, step) if fwd else np.arange(x1 + lead_in, x0 - 1e-6, -step)
        yaw = 0.0 if fwd else math.pi
        for x in xs:
            add(x, y, yaw)
        if li + 1 < len(lanes):
            for yy in np.arange(y + step, lanes[li + 1], step):
                add(xs[-1], yy, math.pi / 2)
    return poses


def object_corners(obj: dict) -> list[tuple[float, float]]:
    spec = CLASS_SPECS[obj["cls"]]
    x, y = obj["xyz"][0], obj["xyz"][1]
    c, s = math.cos(obj["yaw"]), math.sin(obj["yaw"])
    hl, hw = spec["length"] / 2, spec["width"] / 2
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]]


def yolo_labels(world: dict, pose: dict, p: SonarParams, pad_px: float = 3.0) -> list[str]:
    """YOLO lines ('cls cx cy w h', normalised) for objects whose centre is inside the sonar footprint."""
    near, far = p.ground_range_limits()
    half_fov = math.radians(p.hori_fov / 2)
    lines = []
    for obj in world["objects"]:
        if obj.get("burial", 0) >= 0.95:
            continue
        dx, dy = obj["xyz"][0] - pose["x"], obj["xyz"][1] - pose["y"]
        fwd = dx * math.cos(pose["yaw"]) + dy * math.sin(pose["yaw"])
        side = -dx * math.sin(pose["yaw"]) + dy * math.cos(pose["yaw"])
        ground = math.hypot(fwd, side)
        if fwd <= 0 or not (near <= ground <= far) or abs(math.atan2(side, fwd)) > half_fov:
            continue
        px = [world_to_pixel(cx, cy, pose, p) for cx, cy in object_corners(obj)]
        cols = [c for c, _ in px]
        rows = [r for _, r in px]
        c0, c1 = max(min(cols) - pad_px, 0), min(max(cols) + pad_px, SIZE - 1)
        r0, r1 = max(min(rows) - pad_px, 0), min(max(rows) + pad_px, SIZE - 1)
        if c1 - c0 < 2 or r1 - r0 < 2:
            continue
        cls = CLASS_SPECS[obj["cls"]]["yolo"]
        lines.append(f"{cls} {(c0 + c1) / 2 / SIZE:.6f} {(r0 + r1) / 2 / SIZE:.6f} "
                     f"{(c1 - c0) / SIZE:.6f} {(r1 - r0) / SIZE:.6f}")
    return lines


if __name__ == "__main__":
    w = make_world(3)
    path = lawnmower()
    p = SonarParams()
    n_labelled = sum(1 for pose in path if yolo_labels(w, pose, p))
    print(f"{len(w['objects'])} objects, {len(path)} poses, {n_labelled} frames with labels, "
          f"ground range {p.ground_range_limits()}")
