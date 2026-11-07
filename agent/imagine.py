"""Imagination: pick the most informative inspection angle for each contact.

Owner: P5.

For each contact in contacts.json:
  1. imagine  - render "bomb" and "rock" hypotheses of the detected size and heading in a
                clean scene, from north/east/south/west.
  2. score    - how different the two hypotheses look from each angle (0..1).
  3. choose   - the angle where they differ most: that's where the real view settles it.
  4. inspect  - render the real scene from the chosen angle.

Uses ONLY the contact's xy, yaw and size_m for imagining. Never reads world.json.

Writes imagined/*.png, inspections/*.png and viewpoints.json (PLAN.md §11).
"""

from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import visit  # noqa: E402

HYPOTHESES = {"bomb": "aerial_bomb", "rock": "rock"}
DIRECTIONS = ["north", "east", "south", "west"]
SCORE_BLUR = 7      # compare shapes, not exact pixels
SCORE_SCALE = 600.0  # absolute difference that maps to score ~0.63 (1 - 1/e)


def _seed(*parts) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def _signature(view, background):
    """What the object adds to the sonar view: highlights (+) and shadows (-)."""
    return visit._box_blur(view - background, SCORE_BLUR, SCORE_BLUR)


def difference_score(sig_a, sig_b) -> float:
    """0 = hypotheses look identical, towards 1 = easy to tell apart.

    Absolute, not relative: a side-on view where both objects cast big, different
    shadows should beat an end-on view where both are small blobs.
    """
    diff = float(np.abs(sig_a - sig_b).sum())
    return 1.0 - float(np.exp(-diff / SCORE_SCALE))


def imagine_contact(contact, backend="proxy"):
    """Return (scores, display views) for one contact, without touching ground truth."""
    cid, xy = contact["id"], contact["xy"]
    yaw = contact.get("yaw", 0.0)
    rock_seed = _seed(cid, "rock")

    bomb_scenes = [visit.hypothesis_scene(contact, "aerial_bomb", yaw=yaw),
                   # nose/tail can't be told apart from a detection: consider both
                   visit.hypothesis_scene(contact, "aerial_bomb", yaw=yaw + np.pi)]
    rock_scene = visit.hypothesis_scene(contact, "rock", yaw=yaw, seed=rock_seed)
    empty = visit.hypothesis_scene(contact, None)

    scores, views = {}, {}
    for d in DIRECTIONS:
        bg = visit.render_view(empty, xy, d, backend)
        bombs = [visit.render_view(s, xy, d, backend) for s in bomb_scenes]
        rock = visit.render_view(rock_scene, xy, d, backend)
        sig_rock = _signature(rock, bg)
        scores[d] = round(float(np.mean([difference_score(_signature(b, bg), sig_rock)
                                         for b in bombs])), 3)
        views[d] = {"bomb": bombs[0], "rock": rock}
    return scores, views


def _save(polar, path, seed):
    Image.fromarray(visit.to_image(visit.add_speckle(polar, seed)), mode="L").save(path)


def run(run_dir: str, backend: str | None = None) -> list:
    """Main entry point (called by run_demo.py). Writes viewpoints.json."""
    backend = backend or os.environ.get("UXO_RENDER", "proxy")
    with open(os.path.join(run_dir, "contacts.json")) as f:
        contacts = json.load(f)
    os.makedirs(os.path.join(run_dir, "imagined"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "inspections"), exist_ok=True)

    viewpoints = []
    for c in contacts:
        cid = c["id"]
        if "yaw" not in c:
            print(f"[imagine] {cid}: no yaw in contacts.json, assuming 0", file=sys.stderr)
        scores, views = imagine_contact(c, backend)
        chosen = max(scores, key=scores.get)

        imagined = {}
        for d in DIRECTIONS:
            imagined[d] = []
            for hyp in HYPOTHESES:
                rel = f"imagined/{cid}_{d}_{hyp}.png"
                _save(views[d][hyp], os.path.join(run_dir, rel), _seed(cid, d, hyp))
                imagined[d].append(rel)

        # The sensor actually goes there: render the real scene from the chosen angle.
        real = visit.real_scene(run_dir, c, seed=_seed(cid, "seabed"))
        inspection = f"inspections/{cid}_{chosen}.png"
        _save(visit.render_view(real, c["xy"], chosen, backend),
              os.path.join(run_dir, inspection), _seed(cid, chosen, "real"))

        viewpoints.append({"contact_id": cid, "chosen": chosen, "scores": scores,
                           "imagined": imagined, "inspection": inspection})
        ranked = ", ".join(f"{d} {s:.2f}" for d, s in sorted(scores.items(), key=lambda kv: -kv[1]))
        print(f"[imagine] {cid} ({c['cls']}): {ranked} -> {chosen}")

    with open(os.path.join(run_dir, "viewpoints.json"), "w") as f:
        json.dump(viewpoints, f, indent=2)
    print(f"[imagine] Wrote {len(viewpoints)} viewpoints to {run_dir}/viewpoints.json")
    return viewpoints


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "runs/sample")
