"""Run the sonar detector on runs/<id>/sonar and write contacts.json + crops/ (PLAN §11).

Weights come from `weights`, else $UXO_WEIGHTS, else perception/weights/best.pt.
Detections from nearby frames are merged into one contact per object.
"""
import csv
import json
import math
import os

from sim.sonar_geometry import DEFAULT, SIZE, pixel_to_world

NAMES = ["mine_like", "other"]
MERGE_RADIUS_M = 2.5
CROP_MARGIN_PX = 12


def merge_detections(detections: list[dict], radius: float = MERGE_RADIUS_M, min_hits: int = 2) -> list[dict]:
    """Group detections within `radius` metres; keep each group's best frame and its summed-confidence class."""
    groups: list[list[dict]] = []
    for det in sorted(detections, key=lambda d: -d["conf"]):
        for group in groups:
            if math.dist(det["xy"], group[0]["xy"]) <= radius:
                group.append(det)
                break
        else:
            groups.append([det])
    contacts = []
    for group in groups:
        if len(group) < min_hits:
            continue
        votes = {name: sum(d["conf"] for d in group if d["cls"] == name) for name in NAMES}
        cls = max(votes, key=votes.get)
        best = next(d for d in group if d["cls"] == cls)
        n = len(group)
        contacts.append({**best, "cls": cls, "hits": n,
                         "xy": [round(float(sum(d["xy"][i] for d in group) / n), 2) for i in range(2)]})
    return sorted(contacts, key=lambda c: -c["conf"])


def run(run_dir: str, weights: str | None = None, conf: float = 0.25, device=None, min_hits: int = 2) -> list[dict]:
    from PIL import Image
    from ultralytics import YOLO

    weights = weights or os.environ.get("UXO_WEIGHTS") or os.path.join(os.path.dirname(__file__), "weights", "best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"detector weights not found: {weights} (set UXO_WEIGHTS)")
    model = YOLO(weights)
    with open(os.path.join(run_dir, "poses.csv")) as f:
        poses = list(csv.DictReader(f))

    detections = []
    for row in poses:
        pose = {k: float(row[k]) for k in ("x", "y", "yaw")}
        path = os.path.join(run_dir, row["frame"])
        result = model.predict(path, conf=conf, verbose=False, device=device, imgsz=640)[0]
        for box in result.boxes:
            x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
            w, h = x1 - x0, y1 - y0
            detections.append({
                "xy": [float(v) for v in pixel_to_world((x0 + x1) / 2, (y0 + y1) / 2, pose)],
                # Rough: long axis across the image means the object lies across the robot's heading.
                "yaw": round(pose["yaw"] if h >= w else pose["yaw"] + math.pi / 2, 3),
                "size_m": [round(max(w, h) * DEFAULT.m_per_px, 2), round(min(w, h) * DEFAULT.m_per_px, 2)],
                "cls": NAMES[int(box.cls)], "conf": round(float(box.conf), 3),
                "frame": row["frame"], "box": (x0, y0, x1, y1),
            })

    crops_dir = os.path.join(run_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    contacts = []
    for i, c in enumerate(merge_detections(detections, min_hits=min_hits), start=1):
        x0, y0, x1, y1 = c["box"]
        m = CROP_MARGIN_PX
        area = (max(int(x0) - m, 0), max(int(y0) - m, 0), min(int(x1) + m, SIZE), min(int(y1) + m, SIZE))
        Image.open(os.path.join(run_dir, c["frame"])).crop(area).save(os.path.join(crops_dir, f"c{i}.png"))
        contacts.append({"id": f"c{i}", "xy": c["xy"], "yaw": c["yaw"], "size_m": c["size_m"], "cls": c["cls"],
                         "conf": c["conf"], "frame": c["frame"], "crop": f"crops/c{i}.png"})
    with open(os.path.join(run_dir, "contacts.json"), "w") as f:
        json.dump(contacts, f, indent=2)
    print(f"[detect] {len(detections)} detections -> {len(contacts)} contacts", flush=True)
    return contacts
