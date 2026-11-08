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

Two entry points:
  run(run_dir)               - batch pipeline (run_demo.py): imagine + render the inspection.
  imagine_contact(contact,…) - live mission (sim/mission.py): imagine and return the pose to
                               drive to; the mission loop takes the real OceanSim inspection.
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


UNKNOWN_YAWS = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # tried when the detector gives no heading
INSPECT_MARGIN_M = 2.5  # inspection pose sits this far outside the keep-out circle (matches
                        # sim/mission_logic.inspection_pose: INSPECT_EXTRA_M 1.5 + 1.0)
TIE_MARGIN = 0.02       # scores this close count as a tie -> take the side nearest the robot


def _score_views(contact, backend="proxy", range_m=visit.INSPECT_RANGE_M, yaws=None):
    """Imagine bomb vs rock from each side. Returns (scores, display views).

    Uses only the contact's xy, size_m and yaw. If yaw is unknown, averages over `yaws`.
    """
    cid, xy = contact["id"], contact["xy"]
    if yaws is None:
        yaws = [contact.get("yaw", 0.0)]
    # nose/tail can't be told apart from a detection: consider both for every heading
    bomb_scenes = [visit.hypothesis_scene(contact, "aerial_bomb", yaw=y + flip)
                   for y in yaws for flip in (0.0, np.pi)]
    rock_scenes = [visit.hypothesis_scene(contact, "rock", yaw=y, seed=_seed(cid, "rock"))
                   for y in yaws]
    empty = visit.hypothesis_scene(contact, None)

    scores, views = {}, {}
    for d in DIRECTIONS:
        bg = visit.render_view(empty, xy, d, backend, range_m)
        bombs = [visit.render_view(s, xy, d, backend, range_m) for s in bomb_scenes]
        rocks = [visit.render_view(s, xy, d, backend, range_m) for s in rock_scenes]
        per_yaw = []
        for k, rock in enumerate(rocks):
            sig_rock = _signature(rock, bg)
            per_yaw += [difference_score(_signature(b, bg), sig_rock) for b in bombs[2 * k:2 * k + 2]]
        scores[d] = round(float(np.mean(per_yaw)), 3)
        views[d] = {"bomb": bombs[0], "rock": rocks[0]}
    return scores, views


def _pose_clear(x, y, circles, margin=0.3):
    """Same rule as sim.mission_logic.pose_clear: outside every keep-out circle (+ margin)."""
    return all(np.hypot(x - c["xy"][0], y - c["xy"][1]) >= c["r"] + margin for c in circles)


def imagine_contact(contact, keep_out_m=None, robot_pose=None, out_dir=None, backend="proxy",
                    circles=()):
    """Live-mission entry point (LIVE_MISSION.md). Pick the side to inspect a contact from.

    contact:    contacts.json-style dict (id, xy, size_m, optional yaw). Never world.json.
    keep_out_m: keep-out radius; the inspection pose is placed keep_out_m + 2.5 m from the
                contact, and imagination renders from that same distance.
    robot_pose: {x, y, yaw}; breaks near-ties in favour of the shorter move.
    out_dir:    run folder; if given, saves imagined/<cid>_<dir>_<hyp>.png for the video.
    circles:    keep-out circles [{xy, r}]; sides whose pose falls inside one are skipped
                (if every side is blocked, the best side is returned and the caller backs off).

    Returns the same dict as sim.mission_logic.choose_view_standin, plus "imagined":
      {"contact_id", "chosen", "scores", "pose": {x, y, yaw}, "reason", "imagined"}
    """
    cid = contact["id"]
    range_m = visit.INSPECT_RANGE_M if keep_out_m is None else keep_out_m + INSPECT_MARGIN_M
    yaw_known = contact.get("yaw") is not None and contact.get("yaw_known", True)
    scores, views = _score_views(contact, backend, range_m, None if yaw_known else UNKNOWN_YAWS)

    open_sides = [d for d in DIRECTIONS
                  if _pose_clear(*visit.sensor_pose(contact["xy"], d, range_m)[:2], circles)] or list(DIRECTIONS)
    blocked = [d for d in DIRECTIONS if d not in open_sides]
    best = max(scores[d] for d in open_sides)
    tied = [d for d in open_sides if scores[d] >= best - TIE_MARGIN]
    if robot_pose is not None and len(tied) > 1:
        def dist(d):
            sx, sy, _ = visit.sensor_pose(contact["xy"], d, range_m)
            return np.hypot(sx - robot_pose["x"], sy - robot_pose["y"])
        chosen = min(tied, key=dist)
        why_tie = f"; {', '.join(tied)} about equal, took the nearest"
    else:
        chosen = max(open_sides, key=scores.get)
        why_tie = ""
    sx, sy, heading = visit.sensor_pose(contact["xy"], chosen, range_m)

    imagined = {}
    if out_dir:
        os.makedirs(os.path.join(out_dir, "imagined"), exist_ok=True)
        for d in DIRECTIONS:
            imagined[d] = []
            for hyp in HYPOTHESES:
                rel = f"imagined/{cid}_{d}_{hyp}.png"
                _save(views[d][hyp], os.path.join(out_dir, rel), _seed(cid, d, hyp))
                imagined[d].append(rel)

    reason = (f"imagined bomb vs rock from 4 sides at {range_m:.1f} m: {chosen} differs most "
              f"({scores[chosen]:.2f}){why_tie}")
    if blocked:
        reason += f"; {', '.join(blocked)} blocked by a keep-out zone"
    if not yaw_known:
        reason += "; heading unknown, averaged over 4 headings"
    return {"contact_id": cid, "chosen": chosen, "scores": scores,
            "pose": {"x": round(float(sx), 3), "y": round(float(sy), 3), "yaw": round(heading, 4)},
            "reason": reason, "imagined": imagined}


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
        scores, views = _score_views(c, backend)
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
