"""Scripted lawnmower survey in OceanSim -> runs/<id>/ files (PLAN §11) + preview videos (owner: P2).

Runs under Isaac's Python:
    /isaac-sim/python.sh sim/survey.py --out /tmp/run --seed 3 --scenario clear_sand [--quick]

Writes: world.json, sonar/*.png, cam/*.png, poses.csv, labels/*.txt,
        survey.mp4 (sonar | camera) and survey_labels.mp4 (with ground-truth boxes, for checking).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--scenario", default="clear_sand", choices=["clear_sand", "half_buried", "murky"])
parser.add_argument("--quick", action="store_true", help="2 lanes, coarse steps, for testing")
parser.add_argument("--flip-azimuth", action="store_true")
args, _ = parser.parse_known_args()

from sim.session import SimSession  # noqa: E402  (boots nothing until constructed)

import numpy as np  # noqa: E402

from sim.objects import YOLO_NAMES, lawnmower, make_world, yolo_labels  # noqa: E402
from sim.sonar_geometry import SonarParams  # noqa: E402

t0 = time.time()
p = SonarParams()
sim = SimSession(sonar=p, flip_azimuth=args.flip_azimuth)
print(f"[survey] Isaac + sensors ready in {time.time() - t0:.0f}s", flush=True)

from PIL import Image, ImageDraw  # noqa: E402

out = args.out
for d in ("sonar", "cam", "chase", "labels", "_preview", "_preview_labels"):
    os.makedirs(os.path.join(out, d), exist_ok=True)

world = make_world(args.seed, args.scenario)
sim.load_world(world)
with open(os.path.join(out, "world.json"), "w") as f:
    json.dump(world, f, indent=2)

poses = lawnmower(altitude=p.altitude, **({"n_lanes": 2, "step": 1.5} if args.quick else {}))
print(f"[survey] {len(poses)} poses, scenario={args.scenario}, seed={args.seed}", flush=True)


def preview(i, sonar, cam, chase, pose, lines, path):
    s = Image.fromarray(sonar).convert("RGB")
    if lines:
        d = ImageDraw.Draw(s)
        for ln in lines:
            c, cx, cy, w, h = ln.split()
            cx, cy, w, h = (float(v) * 512 for v in (cx, cy, w, h))
            col = (255, 80, 80) if c == "0" else (80, 220, 255)
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=col, width=2)
            d.text((cx - w / 2, cy - h / 2 - 12), YOLO_NAMES[int(c)], fill=col)
    # Layout 1152x720: sonar left (centred), camera top-right, chase view bottom-right.
    canvas = Image.new("RGB", (512 + 640, 720), (0, 0, 0))
    canvas.paste(s, (0, 104))
    canvas.paste(Image.fromarray(cam), (512, 0))
    if chase is not None:
        canvas.paste(Image.fromarray(chase), (512, 360))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), f"SIMULATED  frame {i:06d}  x={pose['x']:.1f} y={pose['y']:.1f} "
                   f"yaw={np.degrees(pose['yaw']):.0f}  {args.scenario}", fill=(255, 255, 255))
    d.text((8, 90), "forward-looking sonar", fill=(200, 200, 200))
    d.text((520, 8), "underwater camera", fill=(255, 255, 255))
    d.text((520, 368), "chase view", fill=(255, 255, 255))
    canvas.save(path)


rows = []
t1 = time.time()
for i, pose in enumerate(poses):
    sim.move_to(pose["x"], pose["y"], pose["yaw"])
    sonar, cam = sim.observe()
    chase = sim.chase()
    name = f"{i:06d}"
    Image.fromarray(sonar, mode="L").save(os.path.join(out, "sonar", f"{name}.png"))
    Image.fromarray(cam).save(os.path.join(out, "cam", f"{name}.png"))
    if chase is not None:
        Image.fromarray(chase).save(os.path.join(out, "chase", f"{name}.png"))
    lines = yolo_labels(sim.world, pose, p)
    with open(os.path.join(out, "labels", f"{name}.txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    rows.append({**pose, "frame": f"sonar/{name}.png"})
    preview(i, sonar, cam, chase, pose, [], os.path.join(out, "_preview", f"{name}.png"))
    preview(i, sonar, cam, chase, pose, lines, os.path.join(out, "_preview_labels", f"{name}.png"))
    if i % 25 == 0:
        print(f"[survey] {i}/{len(poses)} ({(time.time() - t1) / (i + 1):.2f}s/frame)", flush=True)

with open(os.path.join(out, "poses.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["t", "x", "y", "depth", "yaw", "frame"])
    w.writeheader()
    w.writerows(rows)

for src, dst in (("_preview", "survey.mp4"), ("_preview_labels", "survey_labels.mp4")):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "8",
                    "-i", os.path.join(out, src, "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    os.path.join(out, dst)], check=False)
    subprocess.run(["rm", "-rf", os.path.join(out, src)], check=False)

n_lab = sum(1 for r in rows if os.path.getsize(os.path.join(out, "labels", os.path.basename(r["frame"])[:-4] + ".txt")))
print(f"[survey] DONE {len(rows)} frames, {n_lab} with labels, total {time.time() - t0:.0f}s", flush=True)
sim.close()
