"""Mission video: one continuous live run, composed from mission_log.jsonl (LIVE_MISSION.md, M4).

Owner: P5. Reads a folder written by sim/mission.py:
  mission_log.jsonl, sonar/, cam/, chase/, contacts.json, targets.json, viewpoints.json,
  imagined/, inspections/

Every frame is one logged mission step (no cuts). Survey steps play fast, decisions play slowly;
the speed is shown on screen. Layout (1920x1080):

  header: title | state banner | SIMULATED tag | speed
  red bar when an environment change is happening
  left:   sonar (detections marked) over robot camera
  middle: site map (driven path, planned route, old route greyed with clearance, keep-out circles)
  right:  chase view over the robot's reasoning (decision feed / imagination / verdict)

    python viz/mission_compose.py runs/<mission_id>                     # writes mission_demo.mp4
    python viz/mission_compose.py runs/<mission_id> --preview <out_dir> # key stills only
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.sonar_geometry import world_to_pixel  # noqa: E402
from viz.compose import (ACCENT, BG, CLASS_COLOURS, DIRECTIONS, EDGE, FPS, H, MUTED, PANEL,  # noqa: E402
                         TEXT, W, _font, _panel, _paste_fit, _wrap, _writer)

STATE_COLOURS = {"SURVEY": (34, 197, 94), "RESUME": (34, 197, 94), "HOLD": (239, 68, 68),
                 "BACKOFF": (239, 68, 68), "REPLAN": (245, 158, 11), "INSPECT": ACCENT}
STATE_LABELS = {"SURVEY": "SURVEYING", "RESUME": "RESUMING SURVEY", "HOLD": "CONTACT AHEAD: HOLDING",
                "BACKOFF": "BACKING OFF", "REPLAN": "RE-PLANNING", "INSPECT": "INSPECTING"}
OLD_ROUTE = (100, 110, 125)
WARN = (248, 113, 113)

# Seconds on screen per logged step, by state. Scaled down to fit MAX_SECONDS (survey first).
SECONDS = {"SURVEY": 0.12, "RESUME": 0.15, "INSPECT": 0.3, "BACKOFF": 0.3, "HOLD": 0.6, "REPLAN": 0.8}
PAUSE = {"event": 1.5, "hold": 1.5, "verdict": 4.0}  # extra seconds on key steps
TITLE_S, END_S = 3.5, 6.0
MAX_SECONDS = 58.0
MISSION_STEP_S = 1.0  # mission time per logged step (poses.csv t = step)
VERDICT_SHOW_STEPS = 25  # verdict panel stays up this many steps after the verdict


# =====================================================================================
# Data
# =====================================================================================

class Mission:
    def __init__(self, run_dir: str):
        self.dir = run_dir
        with open(os.path.join(run_dir, "mission_log.jsonl")) as f:
            self.log = [json.loads(line) for line in f if line.strip()]
        self.contacts = {c["id"]: c for c in self._json("contacts.json", [])}
        self.targets = {t["contact_id"]: t for t in self._json("targets.json", [])}
        self.viewpoints = {v["contact_id"]: v for v in self._json("viewpoints.json", [])}
        world = self._json("world.json", {})
        self.scenario, self.seed = world.get("scenario", ""), world.get("seed", "")

        # Key moments, all read from the robot's own log
        self.hold_at, self.verdict_at, self.old_routes = {}, {}, {}
        for e in self.log:
            dec, i = e.get("decision", ""), e["step"]
            m = re.search(r"\b(c\d+)\b", dec)
            if not m:
                continue
            cid = m.group(1)
            if dec.startswith("HOLD") and cid not in self.hold_at:
                self.hold_at[cid] = i
                before = self.log[i - 1] if i > 0 else e   # route as planned just before the hold
                pose = e["pose"]
                self.old_routes[cid] = [(pose["x"], pose["y"])] + [tuple(p) for p in before["planned_path"]]
            elif dec.startswith("verdict") and cid not in self.verdict_at:
                self.verdict_at[cid] = i
        self.event_starts = [e["step"] for k, e in enumerate(self.log)
                             if e.get("event") and (k == 0 or not self.log[k - 1].get("event"))]

        pts = [(e["pose"]["x"], e["pose"]["y"]) for e in self.log]
        pts += [tuple(p) for e in self.log[:1] for p in e["planned_path"]]
        pts += [tuple(c["xy"]) for c in self.contacts.values()]
        self.bounds = (min(p[0] for p in pts) - 6, max(p[0] for p in pts) + 6,
                       min(p[1] for p in pts) - 6, max(p[1] for p in pts) + 6)

    def _json(self, name, default):
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            return default
        with open(path) as f:
            return json.load(f)

    def image(self, rel, mode="L"):
        path = os.path.join(self.dir, rel) if rel else ""
        return Image.open(path).convert(mode) if rel and os.path.exists(path) else None

    def frame(self, folder, i, mode="RGB"):
        return self.image(f"{folder}/{i:06d}.png", mode)

    def contact_xy(self, cid, i=None):
        """Robot's estimate of a contact's position (from the log at step i, else contacts.json)."""
        if i is not None:
            for c in self.log[i]["contacts"]:
                if c["id"] == cid:
                    return c["xy"]
        return self.contacts.get(cid, {}).get("xy")

    def clearances(self, cid):
        """(old route closest approach, actual closest approach) to the contact, metres."""
        xy = self.contact_xy(cid)
        old = min(math.dist(p, xy) for p in self.old_routes[cid])
        after = [math.dist((e["pose"]["x"], e["pose"]["y"]), xy) for e in self.log[self.hold_at[cid]:]]
        return old, min(after)

    def focus(self, i):
        """Contact the robot is dealing with at step i, and what to show: imagine / verdict / None."""
        for cid, v in self.verdict_at.items():
            if v <= i < v + VERDICT_SHOW_STEPS:
                return cid, "verdict"
        state = self.log[i]["state"]
        if state in ("HOLD", "BACKOFF", "REPLAN", "INSPECT"):
            m = re.search(r"\b(c\d+)\b", self.log[i]["banner"])
            if m:
                return m.group(1), "imagine"
        return None, None


# =====================================================================================
# Panels
# =====================================================================================

def _header(canvas, m: Mission, e: dict, speed: float):
    d = ImageDraw.Draw(canvas)
    d.text((24, 16), "Look Before You Dive", font=_font(28, True), fill=TEXT)
    state = e["state"]
    label = STATE_LABELS.get(state, state)
    if state == "INSPECT":
        side = next((s for s in DIRECTIONS if s in e["banner"]), "")
        label = f"INSPECTING FROM THE {side.upper()}" if side else label
    colour = STATE_COLOURS.get(state, TEXT)
    lw = d.textlength(label, font=_font(22, True))
    d.rounded_rectangle((330, 14, 330 + lw + 32, 52), radius=8, fill=colour)
    d.text((346, 21), label, font=_font(22, True), fill=BG)
    tag = f"SIMULATED  ·  live mission  ·  {m.scenario}  ·  seed {m.seed}  ·  t {e['step'] * MISSION_STEP_S:.0f} s"
    tw = d.textlength(tag, font=_font(16, True))
    sp = f"speed ×{speed:.0f}" if speed >= 1.5 else "speed ×1"
    spw = d.textlength(sp, font=_font(16, True))
    d.rounded_rectangle((W - tw - spw - 76, 18, W - spw - 52, 50), radius=8, outline=EDGE, fill=PANEL)
    d.text((W - tw - spw - 64, 25), tag, font=_font(16, True), fill=MUTED)
    d.text((W - spw - 36, 25), sp, font=_font(16, True), fill=TEXT)


def _event_bar(canvas, text):
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 60, W, 100), fill=(127, 29, 29))
    d.text((24, 68), text, font=_font(20, True), fill=(254, 226, 226))


def _sonar(canvas, m: Mission, e: dict, box):
    d = ImageDraw.Draw(canvas)
    _panel(d, box, "Forward-looking sonar")
    inner = (box[0] + 20, box[1] + 40, box[2] - 20, box[3] - 36)
    img = m.frame("sonar", e["step"], "L")
    _paste_fit(canvas, img, inner)
    if img is None:
        return
    s = min((inner[2] - inner[0]) / img.width, (inner[3] - inner[1]) / img.height)
    ox = inner[0] + ((inner[2] - inner[0]) - img.width * s) / 2
    oy = inner[1] + ((inner[3] - inner[1]) - img.height * s) / 2
    placed = []  # label positions so far, to stack labels of overlapping detections
    for det in e.get("dets", []):
        col, row = world_to_pixel(det["xy"][0], det["xy"][1], e["pose"])
        if not (0 <= col < img.width and 0 <= row < img.height):
            continue
        x, y, r = ox + col * s, oy + row * s, 22
        colour = (250, 204, 21)
        d.rectangle((x - r, y - r, x + r, y + r), outline=colour, width=3)
        label = f"{det['cls']} {det['conf']:.2f}"
        lw = d.textlength(label, font=_font(15, True))
        lx = x + r + 6 if x + r + 6 + lw < box[2] - 8 else x - r - 6 - lw   # keep inside the panel
        ly = y - r
        while any(abs(ly - py) < 18 and abs(lx - px) < 120 for px, py in placed):
            ly += 18
        placed.append((lx, ly))
        d.text((lx, ly), label, font=_font(15, True), fill=colour)
    n = len(e.get("dets", []))
    d.text((box[0] + 20, box[3] - 28), f"{n} detection{'s' if n != 1 else ''} this frame (YOLO)",
           font=_font(15), fill=MUTED)


def _map(canvas, m: Mission, i: int, box):
    _panel(ImageDraw.Draw(canvas), box, "Site map (robot's view)")
    inner = (box[0] + 16, box[1] + 44, box[2] - 16, box[3] - 64)
    x0, x1, y0, y1 = m.bounds
    s = min((inner[2] - inner[0]) / (x1 - x0), (inner[3] - inner[1]) / (y1 - y0))
    ox = inner[0] + ((inner[2] - inner[0]) - (x1 - x0) * s) / 2
    oy = inner[3] - ((inner[3] - inner[1]) - (y1 - y0) * s) / 2

    def P(x, y):
        return ox + (x - x0) * s, oy - (y - y0) * s

    e = m.log[i]
    over = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    for c in e["keep_out"]:                                   # keep-out zones (filled)
        cx, cy = P(*c["xy"])
        r = c["r"] * s
        od.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(239, 68, 68, 45))
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), over).convert("RGB"))
    d = ImageDraw.Draw(canvas)

    for gx in range(int(math.ceil(x0 / 10)) * 10, int(x1) + 1, 10):
        d.line([P(gx, y0), P(gx, y1)], fill=(30, 41, 59))
    for gy in range(int(math.ceil(y0 / 10)) * 10, int(y1) + 1, 10):
        d.line([P(x0, gy), P(x1, gy)], fill=(30, 41, 59))
    d.text((inner[2] - 22, inner[1]), "N", font=_font(16, True), fill=MUTED)

    for cid, k in m.hold_at.items():                          # old route, greyed + dashed
        if k > i:
            continue
        pts = [P(*p) for p in m.old_routes[cid]]
        for a, b in list(zip(pts, pts[1:]))[::2]:
            d.line([a, b], fill=OLD_ROUTE, width=2)
    if len(e["planned_path"]) > 1:                            # route still to drive
        d.line([P(e["pose"]["x"], e["pose"]["y"])] + [P(*p) for p in e["planned_path"]],
               fill=(34, 197, 94), width=2)
    driven = [P(q["pose"]["x"], q["pose"]["y"]) for q in m.log[: i + 1]]
    if len(driven) > 1:
        d.line(driven, fill=ACCENT, width=3)

    for c in e["keep_out"]:
        cx, cy = P(*c["xy"])
        r = c["r"] * s
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(239, 68, 68), width=2)
    for c in e["contacts"]:
        cx, cy = P(*c["xy"])
        t = m.targets.get(c["id"]) if m.verdict_at.get(c["id"], 1 << 30) <= i else None
        colour = CLASS_COLOURS.get(t["class"], TEXT) if t else TEXT
        d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=colour, width=3)
        d.text((cx + 10, cy - 10), c["id"] + (f"  {t['class']}" if t else ""), font=_font(16, True), fill=colour)

    p = e["pose"]                                             # robot
    px, py = P(p["x"], p["y"])
    hx, hy = math.cos(p["yaw"]), -math.sin(p["yaw"])
    d.polygon([(px + 16 * hx, py + 16 * hy), (px - 9 * hx - 8 * hy, py - 9 * hy + 8 * hx),
               (px - 9 * hx + 8 * hy, py - 9 * hy - 8 * hx)], fill=(250, 204, 21))

    y = box[3] - 56                                           # legend / warning
    warned = [cid for cid, k in m.hold_at.items() if k <= i]
    if warned:
        cid = warned[-1]
        old, _ = m.clearances(cid)
        d.text((box[0] + 20, y), f"Old route (grey) would have passed {old:.1f} m from {cid}",
               font=_font(17, True), fill=WARN)
    else:
        d.text((box[0] + 20, y), "blue: driven   green: planned route", font=_font(16), fill=MUTED)
    if e["keep_out"]:
        d.text((box[0] + 20, y + 26), f"red: {e['keep_out'][0]['r']:.0f} m keep-out (scaled to this 32 m site; "
               "real standoffs are far larger)", font=_font(14), fill=MUTED)


def _feed(canvas, m: Mission, i: int, box):
    d = ImageDraw.Draw(canvas)
    _panel(d, box, "Robot's decisions")
    entries = [(e["step"], e["decision"]) for e in m.log[: i + 1] if e.get("decision")][-5:]
    y = box[1] + 48
    if not entries:
        d.text((box[0] + 20, y), "Surveying. No contacts near the route yet.", font=_font(18), fill=MUTED)
    for step, text in reversed(entries):
        d.text((box[0] + 20, y), f"t {step * MISSION_STEP_S:.0f} s", font=_font(15, True), fill=MUTED)
        lines = _wrap(text, _font(17), box[2] - box[0] - 110, d)[:3]
        for line in lines:
            d.text((box[0] + 90, y), line, font=_font(17), fill=TEXT)
            y += 24
        y += 14
        if y > box[3] - 40:
            break


def _imagine(canvas, m: Mission, cid: str, i: int, box):
    d = ImageDraw.Draw(canvas)
    _panel(d, box, f"Imagining {cid}: bomb vs rock from each side")
    vp = m.viewpoints.get(cid)
    if not vp:
        d.text((box[0] + 20, box[1] + 60), "no viewpoint recorded for this contact", font=_font(18), fill=MUTED)
        return
    reveal = m.log[i]["state"] not in ("HOLD", "BACKOFF")      # scores + choice once re-planning starts
    tile, gap = 116, 14
    gx0 = (box[0] + box[2]) // 2 - (4 * tile + 3 * gap) // 2 + 24
    gy0 = box[1] + 80
    d.text((gx0 - 50, gy0 + tile // 2 - 10), "bomb", font=_font(15, True), fill=MUTED)
    d.text((gx0 - 50, gy0 + tile + 6 + tile // 2 - 10), "rock", font=_font(15, True), fill=MUTED)
    has_imgs = any(vp.get("imagined", {}).values())
    for k, side in enumerate(DIRECTIONS):
        x = gx0 + k * (tile + gap)
        chosen = reveal and side == vp["chosen"]
        score = vp["scores"].get(side)
        label = side + (f" {score:.2f}" if reveal and score is not None else "")
        d.text((x, gy0 - 28), label, font=_font(16, chosen), fill=ACCENT if chosen else TEXT)
        for j, rel in enumerate(vp.get("imagined", {}).get(side, [])[:2]):
            ty = gy0 + j * (tile + 6)
            _paste_fit(canvas, m.image(rel), (x, ty, x + tile, ty + tile), outline=ACCENT if chosen else None)
        if reveal and score is not None:
            by = gy0 + 2 * tile + 18
            d.rectangle((x, by, x + tile, by + 8), fill=(30, 41, 59))
            d.rectangle((x, by, x + int(tile * score), by + 8), fill=ACCENT if chosen else MUTED)
    ty = gy0 + 2 * tile + 46
    if not has_imgs:
        d.text((box[0] + 20, gy0 + 40), "(imagined pictures not saved in this run)", font=_font(16), fill=MUTED)
    msg = (f"Going to the {vp['chosen']} side: that's where a bomb and a rock would look most different."
           if reveal else "Picturing a bomb and a rock at this spot, seen from north, east, south and west…")
    for line in _wrap(msg, _font(18), box[2] - box[0] - 40, d):
        d.text((box[0] + 20, ty), line, font=_font(18), fill=ACCENT if reveal else TEXT)
        ty += 26
    d.text((box[0] + 20, box[3] - 30), "Imagined on board in under a second, before moving.",
           font=_font(14), fill=MUTED)


def _verdict(canvas, m: Mission, cid: str, box):
    d = ImageDraw.Draw(canvas)
    vp, t = m.viewpoints.get(cid, {}), m.targets.get(cid)
    _panel(d, box, f"Inspection of {cid} from the {vp.get('chosen', '?')}")
    _paste_fit(canvas, m.image(vp.get("inspection"), "L"), (box[0] + 20, box[1] + 44, box[0] + 260, box[1] + 284))
    if not t:
        d.text((box[0] + 280, box[1] + 60), "no verdict", font=_font(22), fill=MUTED)
        return
    x = box[0] + 280
    d.text((x, box[1] + 50), t["class"], font=_font(48, True), fill=CLASS_COLOURS.get(t["class"], TEXT))
    d.text((x, box[1] + 112), f"confidence {t['confidence']:.2f}", font=_font(20), fill=TEXT)
    d.text((x, box[1] + 140), f"priority {t['priority']}", font=_font(20), fill=TEXT)
    y = box[1] + 300
    for line in _wrap(f"“{t['reason']}”", _font(19), box[2] - box[0] - 40, d)[:6]:
        d.text((box[0] + 20, y), line, font=_font(19), fill=MUTED)
        y += 27
    d.text((box[0] + 20, box[3] - 30), "Detection and prioritisation only. Humans dispose.",
           font=_font(14), fill=MUTED)


# =====================================================================================
# Frames
# =====================================================================================

def mission_frame(m: Mission, i: int, speed: float = 1.0) -> Image.Image:
    e = m.log[i]
    canvas = Image.new("RGB", (W, H), BG)
    top = 112 if e.get("event") else 72
    _header(canvas, m, e, speed)
    if e.get("event"):
        _event_bar(canvas, e["event"])
    _sonar(canvas, m, e, (24, top, 644, 692))
    d = ImageDraw.Draw(canvas)
    _panel(d, (24, 708, 644, 1056), "Robot camera")
    _paste_fit(canvas, m.frame("cam", i), (44, 746, 624, 1044))
    _map(canvas, m, i, (664, top, 1268, 1056))
    _panel(d, (1288, top, 1896, top + 360), "Chase view")
    _paste_fit(canvas, m.frame("chase", i), (1304, top + 40, 1880, top + 352))
    cid, mode = m.focus(i)
    right = (1288, top + 376, 1896, 1056)
    if mode == "verdict":
        _verdict(canvas, m, cid, right)
    elif mode == "imagine":
        _imagine(canvas, m, cid, i, right)
    else:
        _feed(canvas, m, i, right)
    return canvas


def title_frame(m: Mission) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    lines = [("One continuous run, no cuts.", 48, TEXT),
             ("Mid-run, a person changes the seabed. The robot is not told.", 32, TEXT),
             ("It has to notice with its own sonar, stop, re-plan and check.", 32, MUTED)]
    y = 400
    for text, size, colour in lines:
        tw = d.textlength(text, font=_font(size, size > 40))
        d.text(((W - tw) / 2, y), text, font=_font(size, size > 40), fill=colour)
        y += size + 30
    tag = "SIMULATED  ·  survey steps sped up, decisions shown at slower speed"
    tw = d.textlength(tag, font=_font(18, True))
    d.text(((W - tw) / 2, 980), tag, font=_font(18, True), fill=MUTED)
    return canvas


def end_frame(m: Mission) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((24, 16), "Look Before You Dive", font=_font(28, True), fill=TEXT)
    d.text((330, 22), "What the robot did", font=_font(22), fill=ACCENT)
    # Lead with the closest call (smallest old-route clearance); others get one line each.
    held = sorted(m.hold_at, key=lambda c: m.clearances(c)[0])
    for n, cid in enumerate(held[:3]):
        old, new = m.clearances(cid)
        evs = [s for s in m.event_starts if s <= m.hold_at[cid]]
        if n == 0:
            d.text((80, 100), f"{cid}: closest approach (distances scaled to a 32 m test site)",
                   font=_font(26, True), fill=TEXT)
            d.text((80, 140), f"{old:.1f} m", font=_font(84, True), fill=WARN)
            d.text((80, 240), "on the old route", font=_font(22), fill=MUTED)
            d.text((400, 160), "to", font=_font(48, True), fill=MUTED)
            d.text((540, 140), f"{new:.1f} m", font=_font(84, True), fill=(34, 197, 94))
            d.text((540, 240), "what it actually did", font=_font(22), fill=MUTED)
            if evs:
                d.text((80, 284), f"change at t {evs[-1] * MISSION_STEP_S:.0f} s, stopped at "
                       f"t {m.hold_at[cid] * MISSION_STEP_S:.0f} s, seen by its own sonar",
                       font=_font(20), fill=TEXT)
        else:
            d.text((80, 300 + 34 * n), f"{cid}: old route {old:.1f} m, actually {new:.1f} m",
                   font=_font(20), fill=MUTED)
    _map(canvas, m, len(m.log) - 1, (24, 400, 980, 1056))
    _panel(d, (1000, 90, 1896, 1000), "Target list for the disposal team")
    yy = 140
    for t in sorted(m.targets.values(), key=lambda t: t["priority"]):
        colour = CLASS_COLOURS.get(t["class"], TEXT)
        d.text((1024, yy), f"#{t['priority']}  {t['contact_id']}  {t['class']}", font=_font(26, True), fill=colour)
        yy += 36
        for line in _wrap(t["reason"], _font(17), 840, d)[:3]:
            d.text((1024, yy), line, font=_font(17), fill=MUTED)
            yy += 24
        yy += 18
        if yy > 940:
            break
    d.text((1024, 1020), f"SIMULATED · {len(m.contacts)} contact{'s' if len(m.contacts) != 1 else ''} found · "
           f"keep-out zones scaled to a 32 m test site", font=_font(18), fill=MUTED)
    return canvas


def _durations(m: Mission) -> list:
    """Seconds per step, plus pauses on key steps, scaled so the video fits MAX_SECONDS."""
    pauses = {}
    for s in m.event_starts:
        pauses[s] = pauses.get(s, 0) + PAUSE["event"]
    for s in m.hold_at.values():
        pauses[s] = pauses.get(s, 0) + PAUSE["hold"]
    for s in m.verdict_at.values():
        pauses[s] = pauses.get(s, 0) + PAUSE["verdict"]
    base = [SECONDS.get(e["state"], 0.15) for e in m.log]
    fixed = TITLE_S + END_S + sum(pauses.values())
    slow = sum(b for b, e in zip(base, m.log) if e["state"] not in ("SURVEY", "RESUME"))
    fast = sum(base) - slow
    budget = MAX_SECONDS - fixed - slow
    k = min(1.0, max(budget, 1.0) / fast) if fast else 1.0
    out = [b * (k if e["state"] in ("SURVEY", "RESUME") else 1.0) for b, e in zip(base, m.log)]
    return out, pauses


def run(run_dir: str, out_path: str | None = None) -> str:
    import numpy as np

    m = Mission(run_dir)
    out_path = out_path or os.path.join(run_dir, "mission_demo.mp4")
    durs, pauses = _durations(m)
    write, close = _writer(out_path)
    total = 0.0

    def emit(img, secs):
        nonlocal total
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        for _ in range(max(1, round(secs * FPS))):
            write(arr)
        total += secs

    emit(title_frame(m), TITLE_S)
    for i, secs in enumerate(durs):
        speed = MISSION_STEP_S / max(secs, 1e-6)
        emit(mission_frame(m, i, speed), secs)
        if i in pauses:
            emit(mission_frame(m, i, 1.0), pauses[i])
    emit(end_frame(m), END_S)
    close()
    print(f"[mission_compose] Wrote {out_path} ({total:.0f} s, {len(m.log)} steps)")
    return out_path


def preview(run_dir: str, out_dir: str) -> list:
    """Stills at the key moments: start, change, hold, re-plan, inspect, verdict, resume, end."""
    m = Mission(run_dir)
    os.makedirs(out_dir, exist_ok=True)
    picks = {"start": 0}
    if m.event_starts:
        picks["event"] = m.event_starts[0] + 2
    for cid, k in list(m.hold_at.items())[:1]:
        picks["hold"] = k + 1
        steps = [e["step"] for e in m.log[k:] if e["state"] in ("REPLAN", "INSPECT")]
        if steps:
            picks["replan"] = steps[0]
        if cid in m.verdict_at:
            picks["verdict"] = m.verdict_at[cid] + 1
    picks["late"] = len(m.log) - 1
    written = []
    for name, i in picks.items():
        path = os.path.join(out_dir, f"{name}_{i:06d}.png")
        mission_frame(m, min(i, len(m.log) - 1), 8.0).save(path)
        written.append(path)
    for name, img in (("title", title_frame(m)), ("end", end_frame(m))):
        path = os.path.join(out_dir, f"{name}.png")
        img.save(path)
        written.append(path)
    return written


if __name__ == "__main__":
    args = sys.argv[1:]
    run_dir = args[0] if args and not args[0].startswith("--") else "runs/mission"
    if "--preview" in args:
        for p in preview(run_dir, args[args.index("--preview") + 1]):
            print(p)
    else:
        run(run_dir)
