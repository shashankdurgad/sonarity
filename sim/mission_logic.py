"""Live mission logic (owner: P2). Pure NumPy/PIL, no Isaac, so it can be tested on a laptop.

Holds the STAND-INS used until the real parts arrive (see LIVE_MISSION.md):
  detect_frame_standin   -> replaced by perception.detect.detect_frame (P1)
  ContactTracker         -> may be replaced by P1's tracker (same interface)
  choose_view_standin    -> replaced by agent.imagine.imagine_contact (P5)
  classify_standin       -> replaced by agent.planner.classify_contact (P4)
Plus the parts P2 owns: keep-out circles, path re-planning, the mission map drawing.

Decision code here only ever sees sonar images, poses and its own contacts, never world.json.
"""
import math

import numpy as np

from sim.sonar_geometry import SIZE, SonarParams, pixel_to_world

STEP_M = 0.75          # path spacing (m)
KEEP_OUT_M = 4.0       # keep-out radius around a mine-like contact (scaled to the 32 m site)
CLEARANCE_M = 0.6      # extra margin when detouring around a circle
INSPECT_EXTRA_M = 1.5  # inspection pose sits this far outside the keep-out circle
DIRECTIONS = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}


# ---------------------------------------------------------------- detection (stand-in for P1)
def _score_map(a: np.ndarray, p: SonarParams, block: int = 4) -> np.ndarray:
    """Per-block brightness z-score against seabed at the same range. Masks the centre streak,
    the fan edges and the near seabed edge (all OceanSim artefacts)."""
    from numpy.lib.stride_tricks import sliding_window_view as sw

    n = SIZE // block
    a = a.astype(np.float32)
    blk = a.reshape(n, block, n, block).mean(axis=(1, 3))
    full = (a > 0).reshape(n, block, n, block).all(axis=(1, 3))
    valid = sw(np.pad(full, 2), (5, 5)).all(axis=(2, 3))
    yy, xx = np.mgrid[0:n, 0:n]
    rp = np.hypot((xx + .5) * block - (SIZE - 1) / 2, (SIZE - 1) - (yy + .5) * block)
    valid &= rp * p.m_per_px > 4.6
    ring = rp.astype(int) // 8
    bg = np.zeros_like(blk)
    sd = np.ones_like(blk)
    for k in np.unique(ring[valid]):
        m = valid & (ring == k)
        if m.sum() < 15:
            valid[ring == k] = False
            continue
        v = blk[m]
        med = np.median(v)
        bg[m] = med
        sd[m] = np.median(np.abs(v - med)) * 1.48 + 1
    z = (blk - bg) / sd
    z[~valid] = 0
    # OceanSim draws a bright vertical streak down the centre (= dead ahead), so mask it.
    # KNOWN LIMITATION: this makes the stand-in blind within about ±0.6 m of dead ahead
    # (column subtraction was tried: no better, and it added false contacts). Live events are
    # placed 1-1.6 m off the route centreline. Check P1's detector for the same blind spot.
    c = n // 2
    z[:, c - 4:c + 4] = 0
    return z


def detect_frame_standin(sonar: np.ndarray, pose: dict, p: SonarParams = SonarParams(),
                         thr: float = 6.0, block: int = 4, max_dets: int = 3) -> list[dict]:
    """STAND-IN detector: bright blobs vs seabed at the same range. Can't tell classes apart, so it
    reports every blob as `mine_like` (treat unknowns as dangerous). Same output format as P1's
    detect_frame: {xy, size_m, cls, conf, box}."""
    z = _score_map(sonar, p, block)
    dets = []
    for _ in range(max_dets):
        iy, ix = np.unravel_index(z.argmax(), z.shape)
        peak = float(z[iy, ix])
        if peak < thr:
            break
        y0, y1, x0, x1 = max(iy - 3, 0), iy + 4, max(ix - 3, 0), ix + 4
        win = z[y0:y1, x0:x1] > thr * 0.5
        ys, xs = np.nonzero(win)
        by0, by1 = (y0 + ys.min()) * block, (y0 + ys.max() + 1) * block
        bx0, bx1 = (x0 + xs.min()) * block, (x0 + xs.max() + 1) * block
        col, row = (bx0 + bx1) / 2, (by0 + by1) / 2
        x, y = pixel_to_world(col, row, pose, p)
        size = max(bx1 - bx0, by1 - by0) * p.m_per_px
        dets.append({"xy": [round(float(x), 2), round(float(y), 2)], "size_m": [round(size, 2), round(size * .6, 2)],
                     "cls": "mine_like", "conf": round(min(peak / 15.0, 0.99), 2),
                     "box": [int(bx0), int(by0), int(bx1), int(by1)]})
        z[y0:y1, x0:x1] = 0
    return dets


class ContactTracker:
    """Merge per-frame detections into contacts (contacts.json format). Confirmed after `min_hits`."""

    def __init__(self, radius: float = 2.5, min_hits: int = 2):
        self.radius, self.min_hits = radius, min_hits
        self.contacts: list[dict] = []

    def update(self, dets: list[dict], frame: str) -> list[dict]:
        """Returns contacts that became confirmed on this frame."""
        newly = []
        for d in dets:
            best = min(self.contacts, key=lambda c: math.dist(c["xy"], d["xy"]), default=None)
            if best is not None and math.dist(best["xy"], d["xy"]) <= self.radius:
                n = best["hits"]
                best["xy"] = [round(float((best["xy"][i] * n + d["xy"][i]) / (n + 1)), 2) for i in (0, 1)]
                best["hits"] += 1
                if d["conf"] > best["conf"]:
                    best.update(conf=d["conf"], frame=frame, size_m=d["size_m"], box=d["box"])
                if best["hits"] == self.min_hits:
                    best["id"] = f"c{sum(1 for c in self.contacts if c.get('id')) + 1}"
                    newly.append(best)
            else:
                self.contacts.append({**d, "hits": 1, "frame": frame, "id": None})
        return newly

    def confirmed(self) -> list[dict]:
        return [c for c in self.contacts if c.get("id")]


# ---------------------------------------------------------------- imagine / classify (stand-ins)
def inspection_pose(contact_xy, direction: str, keep_out: float = KEEP_OUT_M) -> dict:
    """Pose `direction` of the contact, outside the keep-out circle, facing the contact."""
    dx, dy = DIRECTIONS[direction]
    d = keep_out + INSPECT_EXTRA_M + 1.0
    x, y = contact_xy[0] + dx * d, contact_xy[1] + dy * d
    return {"x": x, "y": y, "yaw": math.atan2(contact_xy[1] - y, contact_xy[0] - x)}


def pose_clear(pose: dict, circles: list[dict], margin: float = 0.3) -> bool:
    """True if the pose is outside every keep-out circle (plus margin)."""
    return all(math.dist((pose["x"], pose["y"]), c["xy"]) >= c["r"] + margin for c in circles)


def choose_view_standin(contact: dict, robot_pose: dict, keep_out: float = KEEP_OUT_M,
                        circles: list[dict] = ()) -> dict:
    """STAND-IN for P5's imagine_contact: pick the side nearest the robot (shortest move), skipping
    sides whose inspection pose would sit inside ANY keep-out circle."""
    scores = {}
    for name in DIRECTIONS:
        ip = inspection_pose(contact["xy"], name, keep_out)
        if not pose_clear(ip, circles):
            scores[name] = 0.0
            continue
        scores[name] = round(1.0 / (1.0 + math.dist((ip["x"], ip["y"]), (robot_pose["x"], robot_pose["y"]))), 3)
    chosen = max(scores, key=scores.get)
    return {"contact_id": contact["id"], "chosen": chosen, "scores": scores,
            "pose": inspection_pose(contact["xy"], chosen, keep_out), "reason": "stand-in: nearest side"}


def classify_standin(contact: dict, sonar: np.ndarray) -> dict:
    """STAND-IN for P4's classify_contact: can't tell bomb from rock yet, so it says so."""
    return {"contact_id": contact["id"], "class": "uncertain", "confidence": 0.5, "priority": 1,
            "standoff_m": KEEP_OUT_M,
            "reason": "Stand-in classifier: strong compact sonar return; treated as a possible mine until verified."}


# ---------------------------------------------------------------- keep-out + path planning (P2)
def _yawed(points):
    out = []
    for i, (x, y) in enumerate(points):
        nx, ny = points[min(i + 1, len(points) - 1)]
        px, py = points[max(i - 1, 0)]
        out.append({"x": round(float(x), 3), "y": round(float(y), 3),
                    "yaw": round(math.atan2(ny - py, nx - px), 4) if (nx, ny) != (px, py) else 0.0})
    return out


def straight(a: dict, b: dict, step: float = STEP_M) -> list[dict]:
    n = max(int(math.dist((a["x"], a["y"]), (b["x"], b["y"])) / step), 1)
    pts = [(a["x"] + (b["x"] - a["x"]) * k / n, a["y"] + (b["y"] - a["y"]) * k / n) for k in range(1, n + 1)]
    return _yawed(pts)


def _blocked(xy, circles, inflate: float) -> bool:
    return any(math.dist(xy, c["xy"]) < c["r"] + inflate for c in circles)


def _seg_clear(p, q, circles, inflate: float, step: float = 0.25) -> bool:
    n = max(int(math.dist(p, q) / step), 1)
    return not any(_blocked((p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n), circles, inflate)
                   for k in range(n + 1))


def _astar(a, b, circles, inflate: float, res: float = 0.5, margin: float = 12.0):
    """Grid A* (8-connected) from a to b avoiding inflated keep-out circles; handles overlapping zones.
    Returns a smoothed list of (x, y) waypoints, or None."""
    import heapq

    xs = [a[0], b[0]] + [c["xy"][0] for c in circles]
    ys = [a[1], b[1]] + [c["xy"][1] for c in circles]
    x0, y0 = min(xs) - margin, min(ys) - margin
    nx, ny = int((max(xs) + margin - x0) / res) + 1, int((max(ys) + margin - y0) / res) + 1
    cell = lambda p: (min(max(int(round((p[0] - x0) / res)), 0), nx - 1), min(max(int(round((p[1] - y0) / res)), 0), ny - 1))
    pos = lambda c: (x0 + c[0] * res, y0 + c[1] * res)
    free = lambda c: not _blocked(pos(c), circles, inflate)
    s, g = cell(a), cell(b)
    openq, came, cost = [(0.0, s)], {s: None}, {s: 0.0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == g:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (cur[0] + dx, cur[1] + dy)
                if (dx or dy) and 0 <= nb[0] < nx and 0 <= nb[1] < ny and (free(nb) or nb == g):
                    nc = cost[cur] + math.hypot(dx, dy)
                    if nc < cost.get(nb, 1e18):
                        cost[nb], came[nb] = nc, cur
                        heapq.heappush(openq, (nc + math.dist(nb, g), nb))
    if g not in came:
        return None
    cells, c = [], g
    while c is not None:
        cells.append(pos(c))
        c = came[c]
    pts = [a] + cells[::-1][1:-1] + [b]
    smooth, i = [pts[0]], 0  # string-pulling: skip waypoints while the straight line stays clear
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _seg_clear(pts[i], pts[j], circles, inflate):
            j -= 1
        smooth.append(pts[j])
        i = j
    return smooth


def _resample(waypoints, step: float = STEP_M):
    out = []
    for p, q in zip(waypoints, waypoints[1:]):
        n = max(int(math.dist(p, q) / step), 1)
        out += [(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(1, n + 1)]
    return out


def route(a: dict, b: dict, circles: list[dict], step: float = STEP_M) -> list[dict]:
    """Transit a -> b avoiding every keep-out circle (+ CLEARANCE_M). Straight if clear, else grid A*.
    Final pose keeps b's yaw."""
    pa, pb = (a["x"], a["y"]), (b["x"], b["y"])
    if _seg_clear(pa, pb, circles, CLEARANCE_M * 0.5):
        pts = straight(a, b, step)
    else:
        wp = _astar(pa, pb, circles, CLEARANCE_M) or [pa, pb]
        pts = _yawed(_resample(wp, step))
    if pts:
        pts[-1] = {**pts[-1], "yaw": b.get("yaw", pts[-1]["yaw"])}
    return pts


def detour(path: list[dict], circles: list[dict], step: float = STEP_M) -> list[dict]:
    """Replace every run of poses that enters a keep-out circle with an A* route around it, rejoining at
    the next pose that is clear. Works with overlapping circles (their union is avoided)."""
    clear = lambda q: not _blocked((q["x"], q["y"]), circles, CLEARANCE_M * 0.5)
    out, i = [], 0
    while i < len(path):
        if clear(path[i]):
            out.append(path[i])
            i += 1
            continue
        j = i
        while j < len(path) and not clear(path[j]):
            j += 1
        start = out[-1] if out else path[i]
        if j >= len(path):  # path ends inside a zone: stop at the last clear pose
            break
        out.extend(route(start, path[j], circles, step)[:-1])
        i = j
    return out


def min_clearance(path: list[dict], xy) -> float:
    return min((math.dist((p["x"], p["y"]), xy) for p in path), default=float("inf"))


# ---------------------------------------------------------------- mission map (for previews)
STATE_COLOURS = {"SURVEY": (120, 220, 120), "HOLD": (255, 90, 90), "BACKOFF": (255, 60, 160), "REPLAN": (255, 170, 60),
                 "INSPECT": (90, 180, 255), "RESUME": (120, 220, 120), "DONE": (200, 200, 200)}


def draw_map(size, bounds, driven, remaining, old_paths, circles, contacts, targets, pose, state, banner, event):
    """Top-down site map (PIL image). bounds = (x0, x1, y0, y1) in metres."""
    from PIL import Image, ImageDraw

    W, H = size
    x0, x1, y0, y1 = bounds
    s = min((W - 20) / (x1 - x0), (H - 60) / (y1 - y0))

    def P(x, y):
        return 10 + (x - x0) * s, H - 10 - (y - y0) * s

    im = Image.new("RGB", (W, H), (12, 26, 38))
    d = ImageDraw.Draw(im)
    for old in old_paths:
        pts = [P(q["x"], q["y"]) for q in old]
        if len(pts) > 1:
            d.line(pts, fill=(110, 110, 110), width=1)
    if len(remaining) > 1:
        d.line([P(q["x"], q["y"]) for q in remaining], fill=(90, 200, 120), width=2)
    if len(driven) > 1:
        d.line([P(q["x"], q["y"]) for q in driven], fill=(240, 240, 240), width=2)
    for c in circles:
        cx, cy = P(*c["xy"])
        rr = c["r"] * s
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 70, 70), width=2)
    verdict = {t["contact_id"]: t["class"] for t in targets}
    for c in contacts:
        cx, cy = P(*c["xy"])
        col = {"UXO": (255, 60, 60), "non-UXO": (120, 220, 120), "uncertain": (255, 200, 60)}.get(
            verdict.get(c["id"]), (255, 255, 255))
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=col)
        d.text((cx + 6, cy - 6), c["id"], fill=col)
    rx, ry = P(pose["x"], pose["y"])
    yaw = pose["yaw"]
    tri = [(rx + 9 * math.cos(yaw), ry - 9 * math.sin(yaw)),
           (rx + 6 * math.cos(yaw + 2.5), ry - 6 * math.sin(yaw + 2.5)),
           (rx + 6 * math.cos(yaw - 2.5), ry - 6 * math.sin(yaw - 2.5))]
    d.polygon(tri, fill=(250, 200, 20))
    d.rectangle([0, 0, W, 22], fill=(0, 0, 0))
    d.text((8, 5), f"{state}: {banner}", fill=STATE_COLOURS.get(state, (255, 255, 255)))
    if event:
        d.rectangle([0, 24, W, 44], fill=(70, 0, 0))
        d.text((8, 28), event, fill=(255, 220, 220))
    return im
