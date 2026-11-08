"""Game-style HUD composite over the beauty frames -> MP4.

    python render/hud.py --run /runs/<id> --frames /runs/<id>/beauty/frames

Overlays (all from the mission's own logs): mission state banner and clock, event alerts, the
robot's real sonar with detection boxes, a live minimap (route, keep-out zones, contacts), a contact
list with verdicts, floating 3D-anchored contact tags, and verdict cards with the inspection camera.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render.timeline import FPS, Timeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--workers", type=int, default=14)
args, _ = ap.parse_known_args()

TL = Timeline(args.run)
OUT = args.out or os.path.join(args.run, "beauty", f"{TL.run_id}_beauty.mp4")
COMP = os.path.join(args.run, "beauty", "comp")
os.makedirs(COMP, exist_ok=True)


def font(name, size):
    for p in (f"/fonts/{name}", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font("Rajdhani-Bold.ttf", 34)
F_BIG = font("Rajdhani-Bold.ttf", 46)
F_MED = font("Rajdhani-SemiBold.ttf", 28)
F_SM = font("Rajdhani-SemiBold.ttf", 22)
F_MONO = font("ShareTechMono.ttf", 20)
F_MONO_S = font("ShareTechMono.ttf", 16)

CYAN = (60, 230, 255)
AMBER = (255, 176, 20)
RED = (255, 55, 40)
GREEN = (40, 255, 120)
YELLOW = (255, 220, 30)
WHITE = (235, 245, 250)
PANEL = (6, 16, 24, 175)
STATE_COL = {"SURVEY": GREEN, "RESUME": GREEN, "HOLD": AMBER, "REPLAN": CYAN, "INSPECT": (190, 120, 255),
             "BACKOFF": RED}
VERDICT_COL = {"UXO": RED, "uncertain": YELLOW, "non-UXO": GREEN, None: AMBER}


def panel(d, box, accent=CYAN, alpha=175):
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=(6, 16, 24, alpha))
    c = 14
    for (ax, ay, bx, by) in ((x0, y0, x0 + c, y0), (x0, y0, x0, y0 + c), (x1 - c, y0, x1, y0), (x1, y0, x1, y0 + c),
                             (x0, y1, x0 + c, y1), (x0, y1 - c, x0, y1), (x1 - c, y1, x1, y1), (x1, y1 - c, x1, y1)):
        d.line((ax, ay, bx, by), fill=accent + (255,), width=3)


def text(d, xy, s, f, fill=WHITE, anchor="la", shadow=True):
    if shadow:
        d.text((xy[0] + 2, xy[1] + 2), s, font=f, fill=(0, 0, 0, 200), anchor=anchor)
    d.text(xy, s, font=f, fill=fill + (255,) if len(fill) == 3 else fill, anchor=anchor)


def sonar_inset(f, size=330):
    e = TL.entry(f)
    step = TL.step_of(f)
    path = os.path.join(TL.run_dir, "sonar", f"{step:06d}.png")
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("L").resize((size, size))
    g = np.asarray(img).astype(np.float32) / 255
    rgb = np.stack([g * 0.35, g * 1.0, g * 0.75], -1)
    im = Image.fromarray((np.clip(rgb * 1.15, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    try:
        from sim.sonar_geometry import world_to_pixel

        pose = e["pose"]
        for det in e.get("dets") or []:
            u, v = world_to_pixel(det["xy"][0], det["xy"][1], pose)
            u, v = u * size / 512, v * size / 512
            col = RED if det["cls"] == "mine_like" else AMBER
            r = 14
            d.rectangle((u - r, v - r, u + r, v + r), outline=col + (255,), width=2)
            d.text((u + r + 3, v - r), f"{det['cls'].replace('_', '-')} {det['conf']:.2f}", font=F_MONO_S, fill=col + (255,))
    except Exception:  # noqa: BLE001
        pass
    return im


def minimap(f, size=330):
    x0, x1, y0, y1 = TL.bounds
    span = max(x1 - x0, y1 - y0) + 4
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = size / span

    def P(x, y):
        return (size / 2 + (x - cx) * s, size / 2 - (y - cy) * s)
    im = Image.new("RGBA", (size, size), (4, 20, 30, 210))
    d = ImageDraw.Draw(im)
    for gx in range(int(x0) - 4, int(x1) + 5, 4):
        d.line((*P(gx, y0 - 2), *P(gx, y1 + 2)), fill=(40, 90, 110, 70))
    for gy in range(int(y0) - 4, int(y1) + 5, 4):
        d.line((*P(x0 - 2, gy), *P(x1 + 2, gy)), fill=(40, 90, 110, 70))
    e = TL.entry(f)
    step = TL.step_of(f)
    for z in e.get("keep_out") or []:
        (zx, zy), r = z["xy"], z["r"] * s
        c = P(zx, zy)
        d.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=(255, 40, 30, 45), outline=RED + (255,), width=2)
    pp = e.get("planned_path") or []
    if pp:
        d.line([P(e["pose"]["x"], e["pose"]["y"])] + [P(*q) for q in pp], fill=CYAN + (220,), width=2)
    driven = [P(s_["pose"]["x"], s_["pose"]["y"]) for s_ in TL.log[:step + 1]]
    if len(driven) > 1:
        d.line(driven, fill=(255, 255, 255, 200), width=3)
    for c in e.get("contacts") or []:
        v = TL.verdict_at(c["id"], f)
        col = VERDICT_COL.get(v, AMBER)
        q = P(*c["xy"])
        d.rectangle((q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5), fill=col + (255,))
        d.text((q[0] + 7, q[1] - 9), c["id"], font=F_MONO_S, fill=col + (255,))
    rp, ry = TL.robot_pos[f], TL.robot_yaw[f]
    q = P(rp[0], rp[1])
    tri = [(q[0] + 11 * math.cos(-ry), q[1] + 11 * math.sin(-ry)),
           (q[0] + 7 * math.cos(-ry + 2.5), q[1] + 7 * math.sin(-ry + 2.5)),
           (q[0] + 7 * math.cos(-ry - 2.5), q[1] + 7 * math.sin(-ry - 2.5))]
    d.polygon(tri, fill=(255, 200, 30, 255), outline=(0, 0, 0, 255))
    return im


def wrap(s, n):
    return textwrap.wrap(s, n)


def compose_frame(f):
    src = os.path.join(args.frames, f"{f:05d}.png")
    dst = os.path.join(COMP, f"{f:05d}.png")
    if not os.path.exists(src):
        return False
    base = Image.open(src).convert("RGBA")
    W, H = base.size
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    e = TL.entry(f)
    step = TL.step_of(f)
    t = f / FPS
    state = e["state"] if f < TL.n_mission_frames else "COMPLETE"
    scol = STATE_COL.get(state, CYAN)

    # post: teal grade + depth gradient + bloom on bright/emissive areas (game look)
    arr = np.asarray(base.convert("RGB")).astype(np.float32) / 255
    yy = np.linspace(0, 1, H)[:, None, None]
    arr = arr * (0.86 + 0.14 * yy) + np.array([0.0, 0.035, 0.06]) * (1 - yy) * 0.6
    mk = TL.murk(f)
    if mk > 0.01:  # turbid water: scatter blur + distance haze (stronger toward the far, upper frame)
        soft = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1 + 4 * mk))
                          ).astype(np.float32) / 255
        arr = arr + (soft - arr) * (0.6 * mk)
        haze = np.clip(mk * (0.30 + 0.55 * (1 - yy) ** 1.5), 0, 0.85)
        arr = arr + (np.array([0.10, 0.27, 0.28]) - arr) * haze
    lum = arr.max(-1, keepdims=True)
    bright = np.clip((lum - 0.72) / 0.28, 0, 1) * arr
    small = Image.fromarray((bright * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    bl = np.asarray(small.filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255
    bl2 = np.asarray(small.filter(ImageFilter.GaussianBlur(18)).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255
    arr = 1 - (1 - arr) * (1 - 0.55 * bl) * (1 - 0.45 * bl2)  # screen blend
    base = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    # vignette
    vig = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse((-W * 0.15, -H * 0.25, W * 1.15, H * 1.25), fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(120))
    dark = Image.new("RGBA", (W, H), (0, 8, 14, 150))
    base = Image.composite(base, Image.alpha_composite(base, dark), vig)

    # top bar
    info = f"MISSION {TL.run_id.upper()}   T+{int(t // 60):02d}:{int(t % 60):02d}   STEP {step:03d}"
    panel(d, (30, 26, 520, 112))  # banner starts at x=540
    text(d, (52, 34), "AUV-01  //  MINE SURVEY", F_TITLE, CYAN)
    text(d, (52, 76), info, F_MONO if 52 + d.textlength(info, font=F_MONO) < 505 else F_MONO_S)
    # state chip + banner
    bw = 840
    panel(d, (W / 2 - bw / 2, 26, W / 2 + bw / 2, 128), accent=scol)
    dim = state in ("HOLD", "BACKOFF") and int(t * 2.5) % 2 == 1
    text(d, (W / 2, 32), state, F_BIG, tuple(int(c * 0.45) for c in scol) if dim else scol, anchor="ma")
    ban = (e.get("banner") or "").upper() if state != "COMPLETE" else "SURVEY COMPLETE - TARGET LIST READY"
    ban = ban.replace(" FROM PLANNED ROUTE", " FROM ROUTE").replace("(KEEP-OUT 4 M)", "· KEEP-OUT 4 M")
    text(d, (W / 2, 88), ban[:84], F_SM, WHITE, anchor="ma")
    # event alert
    for ev in TL.events:
        f0 = ev["step"] * TL.F
        if f0 <= f < f0 + 4.5 * FPS:
            a = 255 if (int((f - f0) / 6) % 2 == 0 or f - f0 > 40) else 120
            msg = "ENVIRONMENT CHANGE: " + ev["caption"].split(":", 1)[-1].strip().upper()
            bw2 = max(900, len(msg) * 17)
            d.rectangle((W / 2 - bw2 / 2, 150, W / 2 + bw2 / 2, 206), fill=(120, 10, 5, 170 * a // 255),
                        outline=RED + (a,), width=3)
            text(d, (W / 2, 160), "ALERT  //  " + msg, F_MED, (255, 210, 200), anchor="ma")
    # murk notice
    if TL.murk(f) - TL.murk_base > 0.3 and f < TL.n_mission_frames:
        text(d, (W - 52, 150), "VISIBILITY DEGRADED  -  CAMERA UNRELIABLE", F_SM, YELLOW, anchor="ra")

    # contact list (right)
    cons = e.get("contacts") or []
    if f >= TL.n_mission_frames:
        cons = TL.log[-1].get("contacts") or []
    if cons:
        h = 56 + 46 * len(cons)
        panel(d, (W - 470, 200, W - 30, 200 + h))
        text(d, (W - 450, 210), "CONTACTS", F_MED, CYAN)
        for i, c in enumerate(cons):
            v = TL.verdict_at(c["id"], f)
            col = VERDICT_COL.get(v, AMBER)
            y = 252 + 46 * i
            d.rectangle((W - 450, y + 6, W - 438, y + 30), fill=col + (255,))
            tgt = TL.targets.get(c["id"], {})
            ot = (tgt.get("object_type") or "").replace("_", " ") if v else ""
            label = f"{c['id'].upper()}  {c['cls'].replace('_', '-').upper():<10}"
            text(d, (W - 428, y + 2), label, F_MONO, WHITE)
            text(d, (W - 50, y + 2), (v or "INSPECT PENDING").upper() + (f" · {ot}" if ot and ot != "unknown" else ""),
                 F_MONO_S if len(ot) > 6 else F_MONO, col, anchor="ra")

    # floating tags anchored to the 3D contacts
    if cons:
        pts = np.array([[c["xy"][0], c["xy"][1], 2.0] for c in cons])
        uv, z = TL.project(f, pts, W, H)
        for c, (u, v_), zz in zip(cons, uv, z):
            if zz < 1 or not (40 < u < W - 40 and 140 < v_ < H - 60):
                continue
            vv = TL.verdict_at(c["id"], f)
            col = VERDICT_COL.get(vv, AMBER)
            label = f"{c['id'].upper()} · {(vv or 'CONTACT').upper()}"
            tw = d.textlength(label, font=F_SM) + 24
            d.line((u, v_, u + 30, v_ - 30), fill=col + (255,), width=2)
            d.rectangle((u + 30, v_ - 52, u + 30 + tw, v_ - 22), fill=(6, 16, 24, 190), outline=col + (255,), width=2)
            text(d, (u + 42, v_ - 50), label, F_SM, col)

    # verdict card with inspection camera
    for cid, (vstep, vcls) in TL.verdict_step.items():
        f0 = vstep * TL.F
        if f0 <= f < f0 + 4.0 * FPS:
            tgt = TL.targets.get(cid, {})
            col = VERDICT_COL.get(vcls, AMBER)
            cw, ch = 700, 250
            bx, by = 40, H - 380 - ch - 30
            panel(d, (bx, by, bx + cw, by + ch), accent=col, alpha=200)
            text(d, (bx + 20, by + 10), f"VERDICT {cid.upper()}", F_MED, WHITE)
            text(d, (bx + 20, by + 44), vcls.upper(), F_BIG, col)
            ot = (tgt.get("object_type") or "").replace("_", " ")
            if ot and ot != "unknown":
                text(d, (bx + 20, by + 96), ot.upper(), F_MED, WHITE)
            so = tgt.get("standoff_m")
            if so:
                text(d, (bx + 20, by + 130), f"STANDOFF {so} m", F_MONO, col)
            for k, line in enumerate(wrap(tgt.get("reason", ""), 34)[:3]):
                text(d, (bx + 20, by + 160 + 26 * k), line, F_MONO_S, WHITE)
            # inspection camera thumbnail
            insp = None
            for ext in ("north", "south", "east", "west"):
                p = os.path.join(TL.run_dir, "inspections", f"{cid}_{ext}_cam.png")
                if os.path.exists(p):
                    insp = p
                    break
            if insp:
                th = Image.open(insp).convert("RGBA").resize((300, 169))
                ov.alpha_composite(th, (bx + cw - 320, by + 60))
                d.rectangle((bx + cw - 320, by + 60, bx + cw - 20, by + 229), outline=col + (255,), width=2)
                text(d, (bx + cw - 318, by + 34), "ROBOT CAMERA", F_MONO_S, WHITE)

    # bottom-left minimap, bottom-right sonar
    panel(d, (36, H - 374, 40 + 334, H - 36))
    ov.alpha_composite(minimap(f), (40, H - 370))
    text(d, (52, H - 404), "SITE MAP", F_SM, CYAN)
    sn = sonar_inset(f)
    if sn is not None:
        panel(d, (W - 374, H - 374, W - 36, H - 36))
        ov.alpha_composite(sn, (W - 370, H - 370))
        text(d, (W - 52, H - 404), "FORWARD-LOOKING SONAR  ·  LIVE DATA", F_SM, CYAN, anchor="ra")
    # end credits (CC BY attribution for the Objaverse models)
    if f >= TL.n_mission_frames + FPS:
        cr = os.path.join("/runs/_assets/glb/credits.json")
        if os.path.exists(cr):
            c = json.load(open(cr))
            lines = ["3D MODELS (CC BY 4.0, via Objaverse / Sketchfab)"] + \
                    [f"{v['title']} by {v['author']}" for v in c.values()]
            for k, ln in enumerate(lines):
                text(d, (W / 2, H - 230 + 30 * k), ln, F_SM if k else F_MED, CYAN if not k else WHITE, anchor="ma")
    # footer
    text(d, (W / 2, H - 34), f"Rendered replay of logged mission {TL.run_id} · every robot decision is from the live run "
         "(Isaac Sim + OceanSim, YOLOv8 sonar detector, Gemma verdicts)", F_MONO_S, (170, 200, 210), anchor="ma")
    out = Image.alpha_composite(base, ov).convert("RGB")
    out.save(dst)
    return True


if __name__ == "__main__":
    frames = sorted(int(p[:5]) for p in os.listdir(args.frames) if p[:5].isdigit() and p.endswith(".png") and len(p) == 9)
    print(f"[hud] {len(frames)} frames of {TL.n_frames}", flush=True)
    ok = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(compose_frame, frames, chunksize=4)):
            ok += bool(r)
            if i % 100 == 0:
                print(f"[hud] {i}/{len(frames)}", flush=True)
    print(f"[hud] composed {ok}", flush=True)
    # contiguous numbering for ffmpeg
    seq = os.path.join(COMP, "seq")
    os.makedirs(seq, exist_ok=True)
    for i, f in enumerate(frames):
        dst = os.path.join(seq, f"{i:05d}.png")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.join(COMP, f"{f:05d}.png"), dst)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(seq, "%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    OUT], check=True)
    print(f"[hud] wrote {OUT}", flush=True)
    share = OUT.replace(".mp4", "_share720.mp4")
    dur = len(frames) / FPS
    kbps = int(26 * 8 * 1024 / max(dur, 1))  # ~26 MB target, fits chat upload limits
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", OUT, "-vf", "scale=1280:720",
                    "-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.3)}k",
                    "-bufsize", f"{kbps * 2}k", "-pix_fmt", "yuv420p", "-movflags", "+faststart", share], check=True)
    print(f"[hud] wrote {share}", flush=True)
