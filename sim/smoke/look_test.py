"""Tune the look: park the rig right in front of the sphere mine and render under several light levels.

/isaac-sim/python.sh sim/smoke/look_test.py --out /tmp/out
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/tmp/out")
args, _ = parser.parse_known_args()
os.makedirs(args.out, exist_ok=True)

from sim.session import SimSession  # noqa: E402

import numpy as np  # noqa: E402

from sim.objects import make_world  # noqa: E402

sim = SimSession()
world = make_world(3)
sim.load_world(world)
from PIL import Image  # noqa: E402

for obj in world["objects"]:
    if obj["cls"] in ("sphere_mine", "aerial_bomb"):
        x, y = obj["xyz"][:2]
        # Camera looks ~2.7 m ahead (40 deg down from 2.3 m); sonar sees 3-9.7 m ahead.
        for tag, back in (("cam", 2.7), ("sonar", 6.0)):
            sim.move_to(x - back, y, 0.0)
            for sun, dome in ((900, 250), (300, 80), (100, 25), (30, 8)):
                sim.set_lights(sun, dome)
                sonar, cam = sim.observe(settle_steps=6)
                raw = sim.cam._rgba_annot.get_data()
                raw = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
                stem = f"{obj['cls']}_{tag}_sun{sun}"
                Image.fromarray(cam).save(os.path.join(args.out, f"{stem}_uw.png"))
                if raw is not None and raw.size:
                    Image.fromarray(raw[..., :3].astype(np.uint8)).save(os.path.join(args.out, f"{stem}_raw.png"))
                if sun == 900:
                    Image.fromarray(sonar).save(os.path.join(args.out, f"{obj['cls']}_{tag}_sonar.png"))
                print(f"[look] {stem}: cam mean={cam.mean():.0f} std={cam.std():.1f} "
                      f"raw mean={raw.mean() if raw is not None and raw.size else -1:.0f} "
                      f"sonar mean={sonar.mean():.1f} max={sonar.max()}", flush=True)
sim.close()
