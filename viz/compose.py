"""Demo output view: sonar | camera | map | imagined views | inspection + verdict.

Owner: P5.

Builds 1920x1080 frames from a run folder (PLAN.md §11 files only):
  * survey frame - sonar + camera at one pose, map with the path so far and contacts found
  * hero frame   - one contact: imagined bomb vs rock from 4 angles with scores, the chosen
                   angle, the real inspection and the planner's verdict

  * title, numbers (metrics.json only) and master target list frames

    python viz/compose.py runs/<id>                      # writes runs/<id>/demo.mp4 (~48 s)
    python viz/compose.py runs/<id> --preview <out_dir>  # PNG stills to <out_dir> instead

Video order and timings are in TIMING. Encodes with ffmpeg if installed, else OpenCV.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (10, 16, 26)
PANEL = (20, 29, 44)
EDGE = (48, 62, 86)
TEXT = (226, 232, 240)
MUTED = (140, 152, 172)
ACCENT = (56, 189, 248)
CLASS_COLOURS = {"UXO": (239, 68, 68), "non-UXO": (34, 197, 94), "uncertain": (245, 158, 11)}
DIRECTIONS = ["north", "east", "south", "west"]

_FONTS: dict = {}


def _font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONTS:
        candidates = (["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"]
                      + ["/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf",
                         "DejaVuSans.ttf"])
        font = None
        for path in candidates:
            try:
                font = ImageFont.truetype(path, size, index=1 if bold and path.endswith(".ttc") else 0)
                break
            except OSError:
                continue
        _FONTS[key] = font or ImageFont.load_default(size=size)
    return _FONTS[key]


# =====================================================================================
# Run data
# =====================================================================================

class Run:
    def __init__(self, run_dir: str):
        self.dir = run_dir
        with open(os.path.join(run_dir, "poses.csv")) as f:
            self.poses = [{k: (v if k == "frame" else float(v)) for k, v in row.items()}
                          for row in csv.DictReader(f)]
        self.contacts = self._json("contacts.json", [])
        self.viewpoints = {v["contact_id"]: v for v in self._json("viewpoints.json", [])}
        self.targets = {t["contact_id"]: t for t in self._json("targets.json", [])}
        world = self._json("world.json", {})
        self.scenario = world.get("scenario", "")
        self.seed = world.get("seed", "")
        frame_index = {p["frame"]: i for i, p in enumerate(self.poses)}
        self.found_at = {c["id"]: frame_index.get(c["frame"], 0) for c in self.contacts}

    def _json(self, name, default):
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            return default
        with open(path) as f:
            return json.load(f)

    def image(self, rel, mode="L"):
        path = os.path.join(self.dir, rel)
        if not os.path.exists(path):
            return None
        return Image.open(path).convert(mode)

    def cam_for(self, sonar_rel):
        return self.image(sonar_rel.replace("sonar/", "cam/", 1), "RGB")


# =====================================================================================
# Drawing helpers
# =====================================================================================

def _panel(draw, box, title=None):
    draw.rounded_rectangle(box, radius=10, fill=PANEL, outline=EDGE, width=1)
    if title:
        draw.text((box[0] + 14, box[1] + 10), title.upper(), font=_font(15, True), fill=MUTED)


def _paste_fit(canvas, img, box, outline=None):
    """Paste img scaled to fit inside box (keeps aspect), centred."""
    if img is None:
        d = ImageDraw.Draw(canvas)
        d.text((box[0] + 10, box[1] + 10), "missing", font=_font(16), fill=MUTED)
        return
    bw, bh = box[2] - box[0], box[3] - box[1]
    s = min(bw / img.width, bh / img.height)
    im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
    x, y = box[0] + (bw - im.width) // 2, box[1] + (bh - im.height) // 2
    canvas.paste(im.convert("RGB"), (x, y))
    if outline:
        ImageDraw.Draw(canvas).rectangle((x - 3, y - 3, x + im.width + 2, y + im.height + 2),
                                         outline=outline, width=3)


def _wrap(text, font, width, draw):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def _header(canvas, run, phase):
    d = ImageDraw.Draw(canvas)
    d.text((24, 18), "Look Before You Dive", font=_font(30, True), fill=TEXT)
    d.text((340, 27), f"{phase}", font=_font(20), fill=ACCENT)
    tag = f"SIMULATED  ·  {run.scenario}  ·  seed {run.seed}"
    tw = d.textlength(tag, font=_font(16, True))
    d.rounded_rectangle((W - tw - 48, 20, W - 24, 52), radius=8, outline=EDGE, fill=PANEL)
    d.text((W - tw - 36, 27), tag, font=_font(16, True), fill=MUTED)


# =====================================================================================
# Map
# =====================================================================================

def _draw_map(canvas, box, run, upto, focus=None, show_verdicts=False):
    """Top-down site map: path so far, AUV, contacts (coloured once classified)."""
    _panel(ImageDraw.Draw(canvas), box, "Site map")
    d = ImageDraw.Draw(canvas)
    pts = [(p["x"], p["y"]) for p in run.poses] + [tuple(c["xy"]) for c in run.contacts]
    x0, x1 = min(p[0] for p in pts) - 8, max(p[0] for p in pts) + 8
    y0, y1 = min(p[1] for p in pts) - 8, max(p[1] for p in pts) + 8
    inner = (box[0] + 20, box[1] + 44, box[2] - 20, box[3] - 20)
    s = min((inner[2] - inner[0]) / (x1 - x0), (inner[3] - inner[1]) / (y1 - y0))
    ox = inner[0] + ((inner[2] - inner[0]) - (x1 - x0) * s) / 2
    oy = inner[3] - ((inner[3] - inner[1]) - (y1 - y0) * s) / 2

    def to_px(x, y):
        return ox + (x - x0) * s, oy - (y - y0) * s   # +y (north) is up

    # grid every 10 m
    for gx in range(int(math.ceil(x0 / 10)) * 10, int(x1) + 1, 10):
        d.line([to_px(gx, y0), to_px(gx, y1)], fill=(30, 41, 59))
    for gy in range(int(math.ceil(y0 / 10)) * 10, int(y1) + 1, 10):
        d.line([to_px(x0, gy), to_px(x1, gy)], fill=(30, 41, 59))
    d.text((inner[2] - 22, inner[1]), "N", font=_font(16, True), fill=MUTED)
    nx, ny = inner[2] - 16, inner[1] + 24
    d.polygon([(nx, ny), (nx - 5, ny + 14), (nx + 5, ny + 14)], fill=MUTED)

    path = [to_px(p["x"], p["y"]) for p in run.poses[: upto + 1]]
    if len(path) > 1:
        d.line(path, fill=ACCENT, width=3)

    for c in run.contacts:
        if run.found_at[c["id"]] > upto:
            continue
        cx, cy = to_px(*c["xy"])
        # colour only once classified: the focus contact, or every contact on the final map
        verdict = run.targets.get(c["id"]) if (show_verdicts or focus == c["id"]) else None
        colour = CLASS_COLOURS.get(verdict["class"], TEXT) if verdict else TEXT
        r = 9 if focus == c["id"] else 7
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=colour, width=3)
        d.text((cx + 12, cy - 10), c["id"], font=_font(16, True), fill=colour)

    if focus is not None and focus in run.viewpoints:
        c = next(c for c in run.contacts if c["id"] == focus)
        from sim.visit import sensor_pose  # same geometry the renders used
        sx, sy, _ = sensor_pose(c["xy"], run.viewpoints[focus]["chosen"])
        a, b = to_px(sx, sy), to_px(*c["xy"])
        d.line([a, b], fill=TEXT, width=2)
        d.ellipse((a[0] - 6, a[1] - 6, a[0] + 6, a[1] + 6), fill=TEXT)
    elif run.poses:
        p = run.poses[min(upto, len(run.poses) - 1)]
        px, py = to_px(p["x"], p["y"])
        hx, hy = math.cos(p["yaw"]), -math.sin(p["yaw"])
        d.polygon([(px + 14 * hx, py + 14 * hy),
                   (px - 8 * hx - 7 * hy, py - 8 * hy + 7 * hx),
                   (px - 8 * hx + 7 * hy, py - 8 * hy - 7 * hx)], fill=TEXT)


# =====================================================================================
# Frames
# =====================================================================================

def _left_column(canvas, run, sonar_rel, caption):
    d = ImageDraw.Draw(canvas)
    _panel(d, (24, 72, 644, 692), "Forward-looking sonar")
    _paste_fit(canvas, run.image(sonar_rel), (44, 108, 624, 648))
    d.text((44, 660), caption, font=_font(15), fill=MUTED)
    _panel(d, (24, 708, 644, 1056), "Camera")
    _paste_fit(canvas, run.cam_for(sonar_rel), (44, 746, 624, 1044))


def survey_frame(run: Run, i: int) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    pose = run.poses[i]
    _header(canvas, run, "Survey")
    _left_column(canvas, run, pose["frame"],
                 f"t = {pose['t']:.1f} s   depth {pose['depth']:.1f} m   {pose['frame']}")
    _draw_map(canvas, (664, 72, 1896, 1056), run, upto=i)
    d = ImageDraw.Draw(canvas)
    found = [c for c in run.contacts if run.found_at[c["id"]] <= i]
    y = 120
    for c in found:
        d.text((1500, y), f"{c['id']}  {c['cls']}  conf {c['conf']:.2f}", font=_font(18), fill=TEXT)
        y += 28
    d.text((1500, 88), f"CONTACTS: {len(found)}", font=_font(15, True), fill=MUTED)
    return canvas


def hero_frame(run: Run, cid: str, stage: int = 3) -> Image.Image:
    """stage 0: imagined views, 1: + scores and chosen side, 2: + real inspection, 3: + verdict."""
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    c = next(c for c in run.contacts if c["id"] == cid)
    vp = run.viewpoints.get(cid)
    verdict = run.targets.get(cid)
    _header(canvas, run, f"Inspecting {cid}")
    _left_column(canvas, run, c["frame"], f"{cid} detected: {c['cls']}  conf {c['conf']:.2f}")

    # Imagination grid: bomb / rock x 4 directions
    _panel(d, (664, 72, 1896, 560), "Imagined before moving: bomb vs rock from each angle")
    tile, gap, gy0 = 170, 20, 150
    gx0 = (664 + 1896) // 2 - (4 * tile + 3 * gap) // 2
    d.text((gx0 - 64, gy0 + tile // 2 - 10), "bomb", font=_font(16, True), fill=MUTED)
    d.text((gx0 - 64, gy0 + tile + 8 + tile // 2 - 10), "rock", font=_font(16, True), fill=MUTED)
    if vp is None:
        d.text((gx0, gy0), "no viewpoints.json entry for this contact", font=_font(18), fill=MUTED)
    else:
        best = vp["chosen"]
        for k, direction in enumerate(DIRECTIONS):
            x = gx0 + k * (tile + gap)
            score = vp["scores"].get(direction) if stage >= 1 else None
            chosen = direction == best and stage >= 1
            label = f"{direction}" + (f"  {score:.2f}" if score is not None else "")
            d.text((x, gy0 - 30), label, font=_font(18, chosen), fill=ACCENT if chosen else TEXT)
            paths = vp["imagined"].get(direction, [])
            for j, rel in enumerate(paths[:2]):
                ty = gy0 + j * (tile + 8)
                _paste_fit(canvas, run.image(rel), (x, ty, x + tile, ty + tile),
                           outline=ACCENT if chosen else None)
            if score is not None:                                     # score bar
                by = gy0 + 2 * tile + 20
                d.rectangle((x, by, x + tile, by + 8), fill=(30, 41, 59))
                d.rectangle((x, by, x + int(tile * score), by + 8), fill=ACCENT if chosen else MUTED)
        if stage >= 1:
            d.text((gx0, gy0 + 2 * tile + 42), f"Chosen: {best}, where the two hypotheses differ most",
                   font=_font(18), fill=ACCENT)
        else:
            d.text((gx0, gy0 + 2 * tile + 42), "What would a bomb vs a rock look like from each side?",
                   font=_font(18), fill=TEXT)

    # Real inspection + verdict
    _panel(d, (664, 576, 1180, 1056), "Real inspection")
    if vp and stage >= 2:
        _paste_fit(canvas, run.image(vp["inspection"]), (684, 612, 1160, 1044))
    elif vp and stage == 1:
        d.text((690, 800), f"Moving to the {vp['chosen']} side…", font=_font(22), fill=MUTED)
    _panel(d, (1196, 576, 1896, 1056), "Verdict")
    if stage < 3:
        pass
    elif verdict:
        colour = CLASS_COLOURS.get(verdict["class"], TEXT)
        d.text((1220, 620), verdict["class"], font=_font(54, True), fill=colour)
        d.text((1220, 690), f"confidence {verdict['confidence']:.2f}   priority {verdict['priority']}",
               font=_font(22), fill=TEXT)
        if verdict.get("standoff_m"):
            d.text((1220, 724), f"exclusion radius {verdict['standoff_m']:g} m", font=_font(22), fill=TEXT)
        y = 780
        for line in _wrap(f"“{verdict['reason']}”", _font(22), 640, d):
            d.text((1220, y), line, font=_font(22), fill=MUTED)
            y += 32
        d.text((1220, 1010), "Detection and prioritisation only. Humans dispose.",
               font=_font(16), fill=MUTED)
    else:
        d.text((1220, 620), "no verdict yet", font=_font(22), fill=MUTED)
    return canvas


def target_list_frame(run: Run) -> Image.Image:
    """Final frame: site map with every verdict + the master target list for humans."""
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    _header(canvas, run, "Master target list")
    _draw_map(canvas, (24, 72, 1100, 1056), run, upto=len(run.poses) - 1, show_verdicts=True)
    _panel(d, (1120, 72, 1896, 1056), "For the disposal team")
    rows = sorted(run.targets.values(), key=lambda t: t["priority"])
    y = 120
    for t in rows:
        colour = CLASS_COLOURS.get(t["class"], TEXT)
        d.text((1144, y), f"#{t['priority']}", font=_font(26, True), fill=TEXT)
        d.text((1210, y), f"{t['contact_id']}  {t['class']}", font=_font(26, True), fill=colour)
        d.text((1210, y + 36), f"confidence {t['confidence']:.2f}"
               + (f"   exclusion {t['standoff_m']:g} m" if t.get("standoff_m") else ""),
               font=_font(18), fill=TEXT)
        yy = y + 64
        for line in _wrap(t["reason"], _font(18), 640, d):
            d.text((1210, yy), line, font=_font(18), fill=MUTED)
            yy += 26
        y = yy + 24
    d.text((1144, 1010), "Detection and prioritisation only. Humans dispose.", font=_font(16), fill=MUTED)
    return canvas


# =====================================================================================
# Entry points
# =====================================================================================

def preview(run_dir: str, out_dir: str) -> list:
    """Write a few survey frames and one hero frame per contact to out_dir."""
    run = Run(run_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    n = len(run.poses)
    for i in sorted({0, n // 2, n - 1}):
        path = os.path.join(out_dir, f"survey_{i:06d}.png")
        survey_frame(run, i).save(path)
        written.append(path)
    for c in run.contacts:
        path = os.path.join(out_dir, f"hero_{c['id']}.png")
        hero_frame(run, c["id"]).save(path)
        written.append(path)
    path = os.path.join(out_dir, "targets.png")
    target_list_frame(run).save(path)
    written.append(path)
    return written


def title_frame(run: Run) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    lines = [("Wartime munitions still lie under the North Sea,", 44, TEXT),
             ("right where offshore wind farms are being built.", 44, TEXT),
             ("Every site has to be surveyed before anyone digs.", 30, MUTED)]
    y = 400
    for text, size, colour in lines:
        tw = d.textlength(text, font=_font(size, size > 40))
        d.text(((W - tw) / 2, y), text, font=_font(size, size > 40), fill=colour)
        y += size + 30
    tag = "SIMULATED"
    tw = d.textlength(tag, font=_font(18, True))
    d.text(((W - tw) / 2, 980), tag, font=_font(18, True), fill=MUTED)
    return canvas


def numbers_frame(run: Run) -> Image.Image | None:
    """Measured numbers from metrics.json only. None if the run has no metrics."""
    m = run._json("metrics.json", None)
    if not m:
        return None
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    _header(canvas, run, "Results from this run")
    stats = [("contacts found", f"{m.get('contacts_detected', 0)} / {m.get('gt_objects', 0)}"),
             ("classification accuracy", f"{100 * m.get('classification_accuracy', 0):.0f}%"),
             ("UXO missed", f"{m.get('missed_uxo', 0)}")]
    col = W // len(stats)
    for k, (label, value) in enumerate(stats):
        vw = d.textlength(value, font=_font(96, True))
        lw = d.textlength(label, font=_font(26))
        cx = col * k + col // 2
        d.text((cx - vw / 2, 400), value, font=_font(96, True), fill=TEXT)
        d.text((cx - lw / 2, 530), label, font=_font(26), fill=MUTED)
    note = f"One simulated run · {m.get('scenario', '')} · seed {m.get('seed', '')}"
    nw = d.textlength(note, font=_font(20))
    d.text(((W - nw) / 2, 700), note, font=_font(20), fill=MUTED)
    return canvas


# Seconds each part is on screen. The hero contact gets the long, staged sequence.
FPS = 30
TIMING = {"title": 5.0, "survey_total": 10.0, "hero_stage": [4.0, 5.0, 4.0, 6.0],
          "other_contact": 3.0, "numbers": 5.0, "targets": 6.0}


def hero_contact(run: Run) -> str | None:
    """Top-priority UXO if there is one, else the first contact with a viewpoint."""
    ranked = sorted(run.targets.values(), key=lambda t: (t["class"] != "UXO", t["priority"]))
    for t in ranked:
        if t["contact_id"] in run.viewpoints:
            return t["contact_id"]
    return next(iter(run.viewpoints), None)


def timeline(run: Run):
    """Yield (frame image, seconds on screen) in video order."""
    yield title_frame(run), TIMING["title"]
    n = max(len(run.poses), 1)
    for i in range(len(run.poses)):
        yield survey_frame(run, i), TIMING["survey_total"] / n
    hero = hero_contact(run)
    if hero:
        for stage, secs in enumerate(TIMING["hero_stage"]):
            yield hero_frame(run, hero, stage), secs
    for c in run.contacts:
        if c["id"] != hero:
            yield hero_frame(run, c["id"], 3), TIMING["other_contact"]
    numbers = numbers_frame(run)
    if numbers is not None:
        yield numbers, TIMING["numbers"]
    yield target_list_frame(run), TIMING["targets"]


def _writer(path: str):
    """Return (write(np_rgb_frame), close()) using ffmpeg if installed, else OpenCV."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "20", path], stdin=subprocess.PIPE)
        return (lambda f: proc.stdin.write(f.tobytes())), (lambda: (proc.stdin.close(), proc.wait()))

    import cv2
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), FPS, (W, H))
        if vw.isOpened():
            print(f"[compose] encoding with OpenCV ({fourcc})")
            return (lambda f: vw.write(f[:, :, ::-1])), vw.release
    raise RuntimeError("no video encoder: install ffmpeg or opencv-python")


def run(run_dir: str, out_path: str | None = None) -> str:
    """Called by run_demo.py. Writes demo.mp4 into the run folder."""
    import numpy as np

    r = Run(run_dir)
    out_path = out_path or os.path.join(run_dir, "demo.mp4")
    write, close = _writer(out_path)
    total = 0.0
    for frame, secs in timeline(r):
        arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        for _ in range(max(1, round(secs * FPS))):
            write(arr)
        total += secs
    close()
    print(f"[compose] Wrote {out_path} ({total:.0f} s, {FPS} fps, {W}x{H})")
    return out_path


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = sys.argv[1:]
    run_dir = args[0] if args and not args[0].startswith("--") else "runs/sample"
    if "--preview" in args:
        out = args[args.index("--preview") + 1]
        for p in preview(run_dir, out):
            print(p)
    else:
        run(run_dir)
