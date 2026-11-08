"""Live mission run in one Isaac session (owner: P2). See LIVE_MISSION.md.

    /isaac-sim/python.sh sim/mission.py --out /tmp/run --seed 3 [--lanes 2]
        [--events-file F] [--status-file F] [--auto-event uncover_bomb@60 ...]

Every step: move -> observe -> detect -> track -> (maybe) hold / re-plan / inspect -> log.
Environment changes arrive as JSON lines in --events-file (written by infra/mission.py from the
live trigger) or from --auto-event NAME@STEP for unattended tests. The robot is never told about a
change; it only finds out through its sonar.

Writes the PLAN §11 contract (sonar/, cam/, poses.csv, contacts.json, crops/, viewpoints.json,
inspections/, targets.json, world.json at the END) plus chase/, mission_log.jsonl and mission.mp4.
"""
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import deque

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--scenario", default="clear_sand", choices=["clear_sand", "half_buried", "murky"])
parser.add_argument("--lanes", type=int, default=0, help="limit survey lanes (0 = all)")
parser.add_argument("--events-file", default="")
parser.add_argument("--status-file", default="")
parser.add_argument("--auto-event", action="append", default=[], help="NAME@STEP, e.g. uncover_bomb@60")
parser.add_argument("--max-steps", type=int, default=2000)
args, _ = parser.parse_known_args()

from sim.session import SimSession, seabed_bounds_for  # noqa: E402

import numpy as np  # noqa: E402

from sim import mission_logic as ml  # noqa: E402
from sim.objects import CLASS_SPECS, get_world, lawnmower, survey_area  # noqa: E402
from sim.sonar_geometry import SonarParams  # noqa: E402

# Pluggable parts: real implementations replace the stand-ins as they land (LIVE_MISSION.md).
if os.environ.get("UXO_DETECT_URL"):  # P1 YOLO via the deployed uxo-detector service (infra/mission.py)
    from perception.detect_bridge import bridge_detect_frame as detect_frame
else:
    try:
        import ultralytics  # noqa: F401  P1's detect_frame needs torch; stock Isaac 6.0.1 has none
        from perception.detect import detect_frame  # P1
    except Exception:
        detect_frame = ml.detect_frame_standin
try:
    from agent.imagine import imagine_contact  # P5: NumPy sonar "imagination", runs on the robot
except Exception as _e:
    print(f"[mission] imagine_contact unavailable ({_e}); using stand-in view choice", flush=True)
    imagine_contact = None
try:
    from agent.planner import classify_contact  # P4
except Exception:
    classify_contact = None
DETECTOR = "P1 detect_frame" if detect_frame is not ml.detect_frame_standin else "stand-in blob detector"

t0 = time.time()
p = SonarParams()
world = get_world(args.seed, args.scenario)
# World + hidden event objects are built BEFORE the sensors start (see SimSession docstring).
sim = SimSession(sonar=p, seabed_bounds=seabed_bounds_for(world), world=world,
                 reserve={"aerial_bomb": 2, "sphere_mine": 1, "rock": 1, "tyre": 1})
print(f"[mission] Isaac ready in {time.time() - t0:.0f}s; detector: {DETECTOR}", flush=True)

from PIL import Image, ImageDraw  # noqa: E402

out = args.out
for d in ("sonar", "cam", "chase", "crops", "inspections", "_preview"):
    os.makedirs(os.path.join(out, d), exist_ok=True)

area = survey_area(world)
map_bounds = (area[0] - 8, area[1] + 8, area[2] - 6, area[3] + 6)
survey = lawnmower(altitude=p.altitude, area=area, **({"n_lanes": args.lanes} if args.lanes else {}))


def item(pose, state, banner, **kw):
    return {"pose": {"x": pose["x"], "y": pose["y"], "yaw": pose["yaw"]}, "state": state, "banner": banner, **kw}


queue = deque(item(q, "SURVEY", "lawnmower survey") for q in survey)
auto = {}
for spec in args.auto_event:
    name, _, at = spec.partition("@")
    auto[int(at)] = {"type": name}

tracker = ml.ContactTracker()
circles, targets, viewpoints, old_paths, driven, rows = [], [], [], [], [], []
pending, handled = [], set()
caption, caption_until = "", -1
events_offset = 0
log = open(os.path.join(out, "mission_log.jsonl"), "w")


def next_obj_id():
    return f"o{max(int(o['id'][1:]) for o in sim.world['objects']) + 1}"


def point_ahead(dist_lo=11.0, dist_hi=13.0):
    """A point on the robot's remaining SURVEY route, 11-13 m ahead (placed by the environment, not the robot)."""
    target = random.uniform(dist_lo, dist_hi)
    acc, prev = 0.0, (pose["x"], pose["y"])
    for it in queue:
        q = it["pose"]
        acc += math.dist(prev, (q["x"], q["y"]))
        prev = (q["x"], q["y"])
        if acc >= target:
            return q
    return None  # not enough route left ahead


def handle_event(ev):
    """Apply an environment change. Returns the on-screen caption."""
    kind = ev.get("type")
    if kind in ("uncover_bomb", "drop_object"):
        cls = "aerial_bomb" if kind == "uncover_bomb" else ev.get("cls", "aerial_bomb")
        q = point_ahead()
        if q is None:
            return "event skipped: less than ~13 m of route left ahead of the robot"
        # 1-1.6 m off the centreline (well inside the 4 m keep-out, so the route must change) and angled
        # 35-85 deg across the route: end-on bombs give almost no echo and dead-ahead sits in OceanSim's
        # centre streak. Timing and exact spot stay random; a person triggers it.
        lat = random.choice((-1, 1)) * random.uniform(1.0, 1.6)
        x, y = q["x"] - lat * math.sin(q["yaw"]), q["y"] + lat * math.cos(q["yaw"])
        yaw = q["yaw"] + random.choice((-1, 1)) * random.uniform(0.6, 1.5)
        burial = 0.2 if kind == "uncover_bomb" else 0.0
        try:
            sim.apply_change({"type": "reveal_object", "cls": cls, "xy": [round(x, 2), round(y, 2)],
                              "yaw": round(yaw, 3), "burial": burial})
        except ValueError as e:
            return f"event ignored: {e}"
        what = "seabed scour uncovers a bomb" if kind == "uncover_bomb" else f"a {cls} appears"
        return f"ENVIRONMENT CHANGE (robot not told): {what} on the route ahead"
    if kind == "murk_on":
        sim.apply_change({"type": "set_murk", "level": float(ev.get("level", 0.75))})
        return "ENVIRONMENT CHANGE (robot not told): visibility drops"
    if kind == "murk_off":
        sim.apply_change({"type": "set_murk", "level": 0.0})
        return "ENVIRONMENT CHANGE: water clears"
    return f"unknown event {kind}"


def read_events():
    global events_offset
    evs = []
    if args.events_file and os.path.exists(args.events_file):
        with open(args.events_file) as f:
            f.seek(events_offset)
            for line in f:
                if line.strip():
                    evs.append(json.loads(line))
            events_offset = f.tell()
    return evs


def remaining_survey():
    return [it["pose"] for it in queue if it["state"] in ("SURVEY", "RESUME")]


def choose_view(c, here):
    """P5's imagine_contact picks the side; any side inside ANOTHER keep-out zone is rejected and the
    next-best clear side (by P5's scores) is used. Falls back to the stand-in if imagine fails."""
    if imagine_contact is not None:
        try:
            v = imagine_contact(c, keep_out_m=ml.KEEP_OUT_M, robot_pose=here, out_dir=out)
            if ml.pose_clear(v["pose"], circles):
                return v
            from sim import visit
            rng = math.dist((v["pose"]["x"], v["pose"]["y"]), c["xy"])
            for d in sorted(v["scores"], key=v["scores"].get, reverse=True):
                sx, sy, hd = visit.sensor_pose(c["xy"], d, rng)
                cand = {"x": sx, "y": sy, "yaw": hd}
                if ml.pose_clear(cand, circles):
                    return {**v, "chosen": d, "pose": cand,
                            "reason": v["reason"] + f"; {v['chosen']} side is inside another keep-out zone, using {d}"}
        except Exception as e:  # never let imagination stop the robot
            print(f"[mission] imagine_contact failed ({e}); using stand-in", flush=True)
    return ml.choose_view_standin(c, here, circles=circles)


reinspect = []  # contacts whose inspection was interrupted by a newer, closer contact


def plan_inspection(c, here, hold_steps=6, prefix="CONTACT"):
    """Hold, add a keep-out circle, re-plan the rest of the survey around it, inspect from outside, resume."""
    old = remaining_survey()
    clearance = ml.min_clearance(old, c["xy"])
    circles.append({"id": c["id"], "xy": c["xy"], "r": ml.KEEP_OUT_M})
    old_paths.append([here] + old)
    new_rest = ml.detour(old, circles)
    view = choose_view(c, here)
    insp = view["pose"]
    if not ml.pose_clear(insp, circles):  # every side blocked: back off further on the chosen side
        dx, dy = ml.DIRECTIONS[view["chosen"]]
        for extra in (1.5, 3.0, 4.5):
            cand = {**insp, "x": insp["x"] + dx * extra, "y": insp["y"] + dy * extra}
            if ml.pose_clear(cand, circles):
                insp = view["pose"] = cand
                break
    # Rejoin where the route FIRST passes near the contact (its detour). The globally nearest point can be
    # on a later lane, which would silently skip the rest of the current lane.
    near_r = ml.KEEP_OUT_M + ml.CLEARANCE_M + 2.5
    k = next((i for i, q in enumerate(new_rest) if math.dist((q["x"], q["y"]), c["xy"]) <= near_r), 0)
    backoff = []
    if not ml.pose_clear(here, circles, margin=0.0):
        # Confirmed too late: retrace the driven path until clear of every keep-out zone.
        for q in reversed(driven):
            backoff.append({"x": q["x"], "y": q["y"], "yaw": q["yaw"]})
            if ml.pose_clear(q, circles, margin=0.3):
                break
    start = backoff[-1] if backoff else here
    to_insp = ml.route(start, insp, circles)
    if not to_insp and math.dist((start["x"], start["y"]), (insp["x"], insp["y"])) > 0.5:
        insp = None  # no safe path to any inspection side: skip the inspection, still avoid the zone
    back = ml.route(insp or start, new_rest[k], circles) if new_rest else []
    if new_rest and not back:
        print(f"[mission] no safe route back to the survey around {c['id']}; holding", flush=True)
        new_rest, k = [], 0
    why = (f"{c['id']} {'mine-like' if c['cls'] == 'mine_like' else 'unidentified'} contact, "
           f"{clearance:.1f} m from planned route (keep-out {ml.KEEP_OUT_M:.0f} m)")
    done_ids = {t["contact_id"] for t in targets}
    for it_ in queue:  # an inspection we were on our way to: remember it and come back afterwards
        if it_.get("capture") and it_["contact"] not in done_ids and it_["contact"] != c["id"]:
            prev = next((x for x in tracker.confirmed() if x["id"] == it_["contact"]), None)
            if prev is not None and all(prev["id"] != r["id"] for r in reinspect):
                reinspect.append(prev)
    queue.clear()
    # Already inside the zone (contact confirmed late): retreat FIRST, then hold outside it.
    for q in backoff:
        queue.append(item(q, "BACKOFF", f"inside {c['id']} keep-out zone: backing off", contact=c["id"]))
    for _ in range(hold_steps):
        queue.append(item(start, "HOLD", f"{prefix} {c['id']}: stopping. {why}", contact=c["id"]))
    for _ in range(4):
        queue.append(item(start, "REPLAN", f"re-planning route around {c['id']} keep-out zone", contact=c["id"]))
    if insp is not None:
        for q in to_insp:
            queue.append(item(q, "INSPECT", f"moving to inspect {c['id']} from the {view['chosen']}", contact=c["id"]))
        for n in range(4):
            queue.append(item(insp, "INSPECT", f"inspecting {c['id']} from the {view['chosen']}", contact=c["id"],
                              capture=(n == 3), view=view))
    if not new_rest:
        for _ in range(4):
            queue.append(item(insp or start, "HOLD", f"no safe route onward past {c['id']}: holding for operator"))
    for q in back:
        queue.append(item(q, "RESUME", f"rejoining survey around {c['id']}"))
    for q in new_rest[k:]:
        queue.append(item(q, "SURVEY", "lawnmower survey (re-planned)"))
    return why


def preview(i, sonar, chase, dets, it, path_now):
    s = Image.fromarray(sonar).convert("RGB")
    d = ImageDraw.Draw(s)
    for det in dets:
        d.rectangle(det["box"], outline=(255, 220, 0), width=2)
    m = ml.draw_map((640, 360), map_bounds, driven, path_now, old_paths, circles, tracker.confirmed(),
                    targets, it["pose"], it["state"], it["banner"], caption if i <= caption_until else "")
    canvas = Image.new("RGB", (1152, 720), (0, 0, 0))
    canvas.paste(s, (0, 104))
    if chase is not None:
        canvas.paste(Image.fromarray(chase), (512, 0))
    canvas.paste(m, (512, 360))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), f"SIMULATED  step {i:05d}  live mission  seed {args.seed}  {args.scenario}", fill=(255, 255, 255))
    d.text((8, 90), f"forward-looking sonar ({DETECTOR})", fill=(200, 200, 200))
    d.text((520, 8), "chase view", fill=(255, 255, 255))
    canvas.save(os.path.join(out, "_preview", f"{i:06d}.png"))


step = 0
pose = queue[0]["pose"]
while queue and step < args.max_steps:
    for ev in read_events() + ([auto.pop(step)] if step in auto else []):
        caption, caption_until = handle_event(ev), step + 30
        print(f"[mission] step {step}: {caption}", flush=True)
    it = queue.popleft()
    pose = it["pose"]
    look = None
    if it.get("view"):  # at the inspection pose: point the camera at the contact
        lc = next((x for x in tracker.confirmed() if x["id"] == it["contact"]), None)
        look = tuple(lc["xy"]) if lc else None
    sim.move_to(pose["x"], pose["y"], pose["yaw"], cam_look_at=look)
    sonar, cam = sim.observe()
    chase = sim.chase()
    name = f"{step:06d}"
    Image.fromarray(sonar, mode="L").save(os.path.join(out, "sonar", f"{name}.png"))
    Image.fromarray(cam).save(os.path.join(out, "cam", f"{name}.png"))
    if chase is not None:
        Image.fromarray(chase).save(os.path.join(out, "chase", f"{name}.png"))
    rows.append({"t": round(step * 1.0, 2), "x": pose["x"], "y": pose["y"], "depth": 17.5, "yaw": pose["yaw"],
                 "frame": f"sonar/{name}.png"})
    driven.append(pose)

    try:
        dets = detect_frame(sonar, pose) if sonar.max() > 0 else []
    except Exception as e:  # service down or slow: keep the robot going on the stand-in
        print(f"[mission] detector failed ({e!r}); stand-in this frame", flush=True)
        dets = ml.detect_frame_standin(sonar, pose)
    for c in tracker.update(dets, f"sonar/{name}.png"):
        pending.append(c)

    decision = ""
    if it.get("capture"):
        c = next(c for c in tracker.confirmed() if c["id"] == it["contact"])
        view = it["view"]
        ip = f"inspections/{c['id']}_{view['chosen']}.png"
        Image.fromarray(sonar, mode="L").save(os.path.join(out, ip))
        Image.fromarray(cam).save(os.path.join(out, ip[:-4] + "_cam.png"))  # camera aimed at the contact
        if classify_contact is not None:
            try:
                tgt = classify_contact(c, os.path.join(out, ip), view)
            except Exception as e:  # never let the planner stop the robot
                print(f"[mission] classify_contact failed ({e}); using stand-in", flush=True)
                tgt = ml.classify_standin(c, sonar)
        else:
            tgt = ml.classify_standin(c, sonar)
        targets.append(tgt)
        viewpoints.append({"contact_id": c["id"], "chosen": view["chosen"], "scores": view["scores"],
                           "imagined": view.get("imagined", {}), "inspection": ip, "reason": view.get("reason", "")})
        decision = f"verdict {c['id']}: {tgt['class']} ({tgt['reason']})"

    # Safety rule: act on a new contact in ANY moving state (a contact seen while resuming or transiting
    # to an inspection must still stop the robot). Only HOLD/REPLAN/BACKOFF defer it (robot not advancing).
    if it["state"] in ("SURVEY", "RESUME", "INSPECT") and pending:
        while pending:
            c = pending.pop(0)
            if c["id"] in handled:
                continue
            handled.add(c["id"])
            route_ahead = [pose] + [q["pose"] for q in queue]
            clearance = ml.min_clearance(route_ahead, c["xy"])
            # Any contact near the route is a hazard until inspected: YOLO sometimes labels a real mine
            # "other" (m2yolo seed 3: the uncovered bomb was c2 "other" and the robot passed 0.68 m from it).
            if clearance < ml.KEEP_OUT_M + ml.CLEARANCE_M:
                decision = "HOLD: " + plan_inspection(c, pose)
                break
            circles.append({"id": c["id"], "xy": c["xy"], "r": ml.KEEP_OUT_M})
            decision = f"{c['id']} logged, {clearance:.1f} m from route: no route change needed"

    # Come back to an inspection that a newer contact interrupted, once we're moving on the survey again.
    if (not decision and reinspect and it["state"] in ("SURVEY", "RESUME")
            and not any(q.get("capture") for q in queue)):
        c = reinspect.pop(0)
        if c["id"] not in {t["contact_id"] for t in targets}:
            decision = "RE-INSPECT: " + plan_inspection(c, pose, hold_steps=2, prefix="RETURNING TO")

    path_now = [it2["pose"] for it2 in queue]
    preview(step, sonar, chase, dets, it, path_now)
    log.write(json.dumps({
        "step": step, "pose": pose, "state": it["state"], "banner": it["banner"],
        "event": caption if step <= caption_until else "", "dets": [{k: d[k] for k in ("xy", "cls", "conf")} for d in dets],
        "contacts": [{k: c[k] for k in ("id", "xy", "cls", "conf", "hits")} for c in tracker.confirmed()],
        "keep_out": circles, "decision": decision,
        "planned_path": [[round(q["x"], 2), round(q["y"], 2)] for q in path_now[::4]],
    }) + "\n")
    if decision:
        print(f"[mission] step {step}: {decision}", flush=True)
    if args.status_file:
        with open(args.status_file, "w") as f:
            json.dump({"step": step, "state": it["state"], "banner": it["banner"], "pose": pose,
                       "contacts": len(tracker.confirmed()), "remaining": len(queue)}, f)
    if step % 25 == 0:
        print(f"[mission] step {step} {it['state']} remaining {len(queue)} contacts {len(tracker.confirmed())}", flush=True)
    step += 1
log.close()

# ---- contract outputs (world.json written at the END so injected objects are included)
contacts = []
for c in tracker.confirmed():
    x0, y0, x1, y1 = c["box"]
    pad = 12
    crop = Image.open(os.path.join(out, c["frame"])).crop((max(x0 - pad, 0), max(y0 - pad, 0),
                                                           min(x1 + pad, 512), min(y1 + pad, 512)))
    crop.save(os.path.join(out, "crops", f"{c['id']}.png"))
    contacts.append({"id": c["id"], "xy": c["xy"], "yaw": 0.0, "size_m": c["size_m"], "cls": c["cls"],
                     "conf": c["conf"], "frame": c["frame"], "crop": f"crops/{c['id']}.png"})
done = {t["contact_id"] for t in targets}
for c in contacts:
    if c["id"] not in done:
        t = ml.classify_standin(c, None)
        t["reason"] = "Not inspected: off the planned route; flagged for review."
        targets.append(t)
for name, obj in (("contacts.json", contacts), ("targets.json", targets), ("viewpoints.json", viewpoints),
                  ("world.json", sim.world)):
    with open(os.path.join(out, name), "w") as f:
        json.dump(obj, f, indent=2)
with open(os.path.join(out, "poses.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["t", "x", "y", "depth", "yaw", "frame"])
    w.writeheader()
    w.writerows(rows)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "8", "-i", os.path.join(out, "_preview", "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out, "mission.mp4")], check=False)
subprocess.run(["rm", "-rf", os.path.join(out, "_preview")], check=False)
print(f"[mission] DONE {step} steps, {len(contacts)} contacts, {len(circles)} keep-out zones, "
      f"{len(viewpoints)} inspections, total {time.time() - t0:.0f}s", flush=True)
sim.close()
