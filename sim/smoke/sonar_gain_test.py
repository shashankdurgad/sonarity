"""Sweep OceanSim sonar intensity_gain at the bomb and the mine. /isaac-sim/python.sh sim/smoke/sonar_gain_test.py --out /tmp/out"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/tmp/out")
args, _ = parser.parse_known_args()
os.makedirs(args.out, exist_ok=True)

from sim.session import SimSession  # noqa: E402
from sim.objects import make_world  # noqa: E402

sim = SimSession()
world = make_world(3)
sim.load_world(world)
from PIL import Image  # noqa: E402

for obj in world["objects"]:
    x, y = obj["xyz"][:2]
    sim.move_to(x - 6.0, y + 1.5, 0.0)
    for g in (2, 4, 8, 16):
        sim.sonar_gain = g
        sonar, cam = sim.observe(settle_steps=4)
        Image.fromarray(sonar).save(os.path.join(args.out, f"{obj['cls']}_gain{g}.png"))
        print(f"[gain] {obj['cls']} gain={g}: mean={sonar.mean():.1f}", flush=True)
sim.close()
