"""Run YOLOv8 on sonar frames, output contacts.json + crops/."""

import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from ultralytics import YOLO

CLASS_MAP = {0: "mine_like", 1: "other"}


def run(run_dir: str, weights: str | None = None) -> list:
    weights = weights or os.environ.get("UXO_WEIGHTS", "perception/weights/best.pt")
    if not os.path.exists(weights):
        print(f"[detect] weights not found: {weights}", file=sys.stderr)
        return []

    model = YOLO(weights)
    sonar_dir = os.path.join(run_dir, "sonar")
    crops_dir = os.path.join(run_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # Load poses for world-coordinate mapping
    poses = {}
    poses_path = os.path.join(run_dir, "poses.csv")
    if os.path.exists(poses_path):
        with open(poses_path) as f:
            for row in csv.DictReader(f):
                poses[row["frame"]] = row

    frames = sorted(f for f in os.listdir(sonar_dir) if f.endswith(".png"))
    frame_paths = [os.path.join(sonar_dir, f) for f in frames]

    contacts = []
    contact_id = 1

    for frame_name, frame_path in zip(frames, frame_paths):
        results = model(frame_path, verbose=False)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.8:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx_px = (x1 + x2) / 2
                cy_px = (y1 + y2) / 2
                w_px = x2 - x1
                h_px = y2 - y1

                # Estimate world position from pose if available
                rel_frame = f"sonar/{frame_name}"
                pose = poses.get(rel_frame)
                if pose:
                    img = Image.open(frame_path)
                    iw, ih = img.size
                    # Rough mapping: pixel center to world offset (assume 10m sonar range)
                    sonar_range = 10.0
                    wx = float(pose["x"]) + (cx_px / iw - 0.5) * sonar_range
                    wy = float(pose["y"]) + (cy_px / ih - 0.5) * sonar_range
                    size_x = (w_px / iw) * sonar_range
                    size_y = (h_px / ih) * sonar_range
                    yaw = float(pose["yaw"])
                else:
                    wx, wy = cx_px, cy_px
                    size_x, size_y = w_px, h_px
                    yaw = 0.0

                # Save crop
                img = Image.open(frame_path)
                crop = img.crop((max(0, int(x1)), max(0, int(y1)),
                                 min(img.width, int(x2)), min(img.height, int(y2))))
                # Ensure crop fits within 256x256
                if crop.width > 256 or crop.height > 256:
                    crop.thumbnail((256, 256))
                cid = f"c{contact_id}"
                crop_path = f"crops/{cid}.png"
                crop.save(os.path.join(run_dir, crop_path))

                cls_name = CLASS_MAP.get(cls_idx, "other")

                contacts.append({
                    "id": cid,
                    "xy": [round(wx, 1), round(wy, 1)],
                    "yaw": round(yaw, 2),
                    "size_m": [round(size_x, 2), round(size_y, 2)],
                    "cls": cls_name,
                    "conf": round(conf, 3),
                    "frame": rel_frame,
                    "crop": crop_path,
                })
                contact_id += 1

    out_path = os.path.join(run_dir, "contacts.json")
    with open(out_path, "w") as f:
        json.dump(contacts, f, indent=2)

    print(f"[detect] {len(contacts)} contacts from {len(frames)} frames → {out_path}")
    return contacts


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/sample"
    run(run_dir)
