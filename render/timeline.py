"""Mission replay timeline for the beauty render (pure Python + NumPy; used by render/beauty.py
inside Isaac and by render/hud.py for compositing, so both agree on every frame).

Reads runs/<id>/mission_log.jsonl (+ world.json, targets.json) and produces, per video frame:
robot pose, mission state, events, keep-out zones, planned route, contacts + verdicts, and a
cinematic camera (position, look-at target, field of view).
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

FPS = 24
FRAMES_PER_STEP = 6          # 4 mission steps per second of video
INTRO_STEPS = 10             # opening shot
REVEAL_S = 3.5               # bomb-reveal shot length
OUTRO_S = 5.0                # crane up to the site overview at the end
AUV_ALTITUDE = 2.4           # m above the seabed (sonar rig altitude 2.5 m)
SMOOTH_TAU_S = 0.55          # camera smoothing time constant


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


class Timeline:
    def __init__(self, run_dir: str, frames_per_step: int = FRAMES_PER_STEP):
        self.run_dir = run_dir
        self.F = frames_per_step
        with open(os.path.join(run_dir, "mission_log.jsonl")) as f:
            self.log = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(run_dir, "world.json")) as f:
            self.world = json.load(f)
        tpath = os.path.join(run_dir, "targets.json")
        self.targets = {t["contact_id"]: t for t in json.load(open(tpath))} if os.path.exists(tpath) else {}
        self.run_id = os.path.basename(os.path.normpath(run_dir))
        self.n_steps = len(self.log)
        self.n_mission_frames = (self.n_steps - 1) * self.F + 1
        self.n_frames = self.n_mission_frames + int(OUTRO_S * FPS)
        self._parse()
        self._robot_track()
        self._camera_track()

    # ------------------------------------------------------------------ parsing
    def _parse(self):
        L = self.log
        # events: first step of each caption
        self.events = []
        prev = ""
        for s in L:
            ev = s.get("event") or ""
            if ev and ev != prev:
                kind = ("bomb" if "bomb" in ev.lower() else "murk_on" if "visibility drops" in ev.lower()
                        else "murk_off" if "visibility" in ev.lower() else "drop" if "object" in ev.lower() else "other")
                self.events.append({"step": s["step"], "caption": ev, "kind": kind})
            prev = ev
        # objects revealed by events: ids not in the generated base world (needs sim.objects; optional)
        base_ids = None
        try:
            from sim.objects import get_world
            base = get_world(self.world.get("seed", 1), self.world.get("scenario", "clear_sand"))
            base_ids = {o["id"] for o in base["objects"]}
        except Exception:
            pass
        objs = self.world["objects"]
        extra = [o for o in objs if base_ids is not None and o["id"] not in base_ids]
        if base_ids is None:  # fallback: assume the last k objects are event objects
            k = sum(1 for e in self.events if e["kind"] in ("bomb", "drop"))
            extra = objs[len(objs) - k:] if k else []
        reveal_events = [e for e in self.events if e["kind"] in ("bomb", "drop")]
        self.reveal_step = {}  # object id -> step
        for obj, ev in zip(extra, reveal_events):
            self.reveal_step[obj["id"]] = ev["step"]
            ev["object"] = obj
        # verdicts
        self.verdict_step = {}
        for s in L:
            m = re.match(r"verdict (c\d+): (\S+)", s.get("decision") or "")
            if m and m.group(1) not in self.verdict_step:
                self.verdict_step[m.group(1)] = (s["step"], m.group(2))
        # first confirmed step per contact
        self.contact_first = {}
        self.contact_xy = {}
        for s in L:
            for c in s.get("contacts") or []:
                self.contact_first.setdefault(c["id"], s["step"])
                self.contact_xy[c["id"]] = c["xy"]
                self.contact_cls = getattr(self, "contact_cls", {})
                self.contact_cls[c["id"]] = c["cls"]
        self.contact_cls = getattr(self, "contact_cls", {})
        # murk level per step
        self.murk_base = 0.55 if self.world.get("scenario") == "murky" else 0.0  # murky site: hazy all run
        self.murk_steps = [(e["step"], 1.0 if e["kind"] == "murk_on" else self.murk_base)
                           for e in self.events if e["kind"] in ("murk_on", "murk_off")]
        # site bounds
        xs = [o["xyz"][0] for o in objs] + [s["pose"]["x"] for s in L]
        ys = [o["xyz"][1] for o in objs] + [s["pose"]["y"] for s in L]
        self.bounds = (min(xs) - 4, max(xs) + 4, min(ys) - 4, max(ys) + 4)
        self.centre = np.array([(self.bounds[0] + self.bounds[1]) / 2, (self.bounds[2] + self.bounds[3]) / 2, 0.0])

    # ------------------------------------------------------------- robot track
    def _robot_track(self):
        P = np.array([[s["pose"]["x"], s["pose"]["y"]] for s in self.log], float)
        Y = np.unwrap(np.array([s["pose"]["yaw"] for s in self.log], float))
        n = self.n_frames
        pos = np.zeros((n, 3))
        yaw = np.zeros(n)
        for f in range(n):
            fm = min(f, self.n_mission_frames - 1)
            i, t = divmod(fm, self.F)
            t /= self.F
            i0, i1, i2, i3 = max(i - 1, 0), i, min(i + 1, self.n_steps - 1), min(i + 2, self.n_steps - 1)
            pos[f, :2] = _catmull(P[i0], P[i1], P[i2], P[i3], t)
            yaw[f] = Y[i1] + (Y[i2] - Y[i1]) * (t * t * (3 - 2 * t))
        # heading follows motion direction when moving (looks natural), else logged yaw
        vel = np.gradient(pos[:, :2], axis=0)
        speed = np.linalg.norm(vel, axis=1)
        mot = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
        head = np.where(speed > 0.02, mot, np.nan)
        # fill stationary gaps with logged yaw, then smooth
        hy = np.where(np.isnan(head), yaw, head)
        hy = np.unwrap(hy)
        k = int(FPS * 0.4)
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        hy = np.convolve(np.pad(hy, k, mode="edge"), ker, mode="valid")
        self.robot_pos = pos
        self.robot_pos[:, 2] = AUV_ALTITUDE
        self.robot_yaw = hy
        self.robot_speed = speed

    def step_of(self, f: int) -> int:
        return min(f // self.F, self.n_steps - 1)

    def entry(self, f: int) -> dict:
        return self.log[self.step_of(f)]

    def time_s(self, f: int) -> float:
        return f / FPS

    def murk(self, f: int) -> float:
        lvl, t = self.murk_base, f / self.F
        for st, v in self.murk_steps:
            if t >= st:
                lvl = v if t >= st + 8 else lvl + (v - lvl) * (t - st) / 8.0
        return lvl

    def reveal_progress(self, oid: str, f: int) -> float:
        """0 = buried/hidden, 1 = fully uncovered (1.2 s rise)."""
        if oid not in self.reveal_step:
            return 1.0
        dt = (f - self.reveal_step[oid] * self.F) / FPS
        return float(np.clip(dt / 1.2, 0.0, 1.0)) if dt >= 0 else -1.0

    def verdict_at(self, cid: str, f: int):
        v = self.verdict_step.get(cid)
        if v and self.step_of(f) >= v[0]:
            return v[1]
        if f >= self.n_mission_frames and cid in self.targets:  # final target list (incl. not inspected)
            return self.targets[cid].get("class")
        return None

    # ------------------------------------------------------------ camera track
    def _contact_in_banner(self, s):
        m = re.search(r"\b(c\d+)\b", s.get("banner") or "")
        return m.group(1) if m and m.group(1) in self.contact_xy else None

    def _shot(self, f):
        # One consistent chase view for the whole mission (cutting between angles confused viewers),
        # then a single crane-up over the site once the mission is complete.
        return ("outro", None) if f >= self.n_mission_frames else ("chase", None)

    def _camera_track(self):
        n = self.n_frames
        cam = np.zeros((n, 3))
        tgt = np.zeros((n, 3))
        fov = np.zeros(n)
        shots = []
        shot_start = {}
        # One fixed viewing direction for the whole mission (looking north, like the minimap): the camera
        # tracks the robot's position but never rotates, so lane turns and inspections don't spin the view.
        FOLLOW = np.array([0.0, -9.0, 6.0])
        prev_shot = None
        orbit_a0 = 0.0
        for f in range(n):
            shot, arg = self._shot(f)
            key = (shot, arg if not isinstance(arg, dict) else arg["id"])
            if key != prev_shot:
                shot_start[f] = key
                start_f = f
            t = (f - start_f) / FPS
            r = self.robot_pos[f]
            yaw = self.robot_yaw[f]
            fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            up = np.array([0, 0, 1.0])
            fv = 50.0
            if shot == "intro":
                # low shot ahead of the robot, robot glides toward and past camera
                anchor = self.robot_pos[min(INTRO_STEPS * self.F, n - 1)]
                c = anchor + np.array([0.0, -3.2, -1.6]) + fwd * 2.0
                tg = r + np.array([0, 0, -0.2])
                fv = 42
            elif shot == "chase":
                c = r + FOLLOW
                tg = r + np.array([0.0, 2.0, -1.2])
                fv = 52
            elif shot == "high":
                c = r - fwd * 7.0 + left * 3.0 + up * 4.5
                tg = r + fwd * 2.0 - up * 1.5
            elif shot == "orbit":
                cxy = np.array(self.contact_xy[arg] + [0.0])
                mid = 0.5 * (r + cxy)
                if key != prev_shot:
                    d0 = cam[f - 1] - mid if f > 0 else -fwd
                    orbit_a0 = math.atan2(d0[1], d0[0])
                a = orbit_a0 + 0.22 * t
                rad = 8.5
                c = mid + np.array([math.cos(a) * rad, math.sin(a) * rad, 4.2])
                tg = mid + np.array([0, 0, 0.3])
                fv = 48
            elif shot == "top":
                cxy = np.array(self.contact_xy[arg] + [0.0]) if arg else r
                mid = 0.5 * (r + cxy)
                c = mid + np.array([0.0, -5.5, 15.0])
                tg = mid
                fv = 55
            elif shot == "side":
                cxy = np.array(self.contact_xy[arg] + [0.0])
                d = cxy - r
                d[2] = 0
                d /= (np.linalg.norm(d) + 1e-6)
                perp = np.array([-d[1], d[0], 0.0])
                mid = 0.5 * (r + cxy)
                c = mid + perp * 6.5 + up * 1.4 - d * 1.0
                tg = mid + up * 0.6
                fv = 46
            elif shot == "reveal":
                o = np.array(arg["xyz"][:2] + [0.0])
                d = o - r
                d[2] = 0
                d /= (np.linalg.norm(d) + 1e-6)
                perp = np.array([-d[1], d[0], 0.0])
                c = o + d * 3.6 + perp * 1.6 + up * 1.3
                if np.linalg.norm((c - r)[:2]) < 4.0:  # bomb close to the robot: frame both from the side
                    c = 0.5 * (o + r) + perp * 7.0 + up * 2.0
                tg = o + (r - o) * 0.35 + up * 0.5
                fv = 44
            else:  # outro: crane up over the site
                k = min(t / OUTRO_S, 1.0)
                k = k * k * (3 - 2 * k)
                c0 = r + FOLLOW
                c1 = self.centre + np.array([0.0, -(self.bounds[3] - self.bounds[2]) * 0.55, 30.0])
                c = c0 + (c1 - c0) * k
                tg = r + (self.centre - r) * k
                fv = 50 + 8 * k
            cam[f], tgt[f], fov[f] = c, tg, fv
            shots.append(shot)
            prev_shot = key
        # smoothing (hard cuts into/out of intro and reveal)
        a = 1.0 - math.exp(-1.0 / (FPS * SMOOTH_TAU_S))
        sc, st_, sf = cam.copy(), tgt.copy(), fov.copy()
        for f in range(1, n):
            cut = shots[f] != shots[f - 1] and ("reveal" in (shots[f], shots[f - 1]) or shots[f - 1] == "intro")
            if cut:
                continue
            sc[f] = sc[f - 1] + (cam[f] - sc[f - 1]) * a
            st_[f] = st_[f - 1] + (tgt[f] - st_[f - 1]) * a
            sf[f] = sf[f - 1] + (fov[f] - sf[f - 1]) * a
        for f in range(n):  # shot blends can swing the camera through the robot: keep >= 5.5 m away
            if shots[f] == "intro":
                continue
            d = sc[f] - self.robot_pos[f]
            dist = float(np.linalg.norm(d))
            if dist < 5.5:
                d = d / (dist + 1e-6) if dist > 1e-3 else np.array([0.0, -1.0, 0.3])
                sc[f] = self.robot_pos[f] + d * 5.5
        sc[:, 2] = np.maximum(sc[:, 2], 0.6)  # never below the sand
        self.cam_pos, self.cam_tgt, self.cam_fov, self.shots = sc, st_, sf, shots

    # -------------------------------------------------------------- projection
    def project(self, f: int, pts: np.ndarray, W: int, H: int):
        """World points (N,3) -> pixel coords (N,2) and depth (N,); camera as rendered."""
        c, t = self.cam_pos[f], self.cam_tgt[f]
        fwd = t - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(right, fwd)
        rel = pts - c
        x, y, z = rel @ right, rel @ up, rel @ fwd
        fx = (W / 2) / math.tan(math.radians(self.cam_fov[f]) / 2)
        u = W / 2 + fx * x / np.maximum(z, 1e-3)
        v = H / 2 - fx * y / np.maximum(z, 1e-3)
        return np.stack([u, v], 1), z
