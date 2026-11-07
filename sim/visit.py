"""Render forward-looking sonar views of a contact for imagination and inspection.

Owner: P5.

Two kinds of render, always with the same sensor geometry:
  * imagined   - a CLEAN scene: flat seabed + one hypothesis object built only from
                 the contact (xy, yaw, size_m). Never reads world.json.
  * inspection - the REAL scene around the contact, built from world.json (the sim's
                 truth, which is what the robot would actually see).

Backends:
  proxy    - NumPy sonar: heightfield + shadow casting + Lambertian returns + speckle.
             Runs on a laptop. Also the seed of the Python fallback sim (PLAN.md §9).
  oceansim - OceanSim ImagingSonarSensor on Modal. TODO once P2's image passes step 3.

Conventions (confirm with P2/P3):
  +x = east, +y = north, z up, seabed at z = 0 (metres).
  yaw = radians, counter-clockwise from +x (east). Object yaw points along its long axis
  (for a bomb, towards the nose).
  A view "from north" puts the sensor north of the contact, looking south.
"""

from __future__ import annotations

import json
import os

import numpy as np

from sim.objects import CLASS_SPECS                              # P2's object sizes
from sim.sonar_geometry import DEFAULT as SONAR, polar_to_fan   # P2's sonar + frame convention

# --- Inspection geometry: identical for imagined and real renders --------------------
INSPECT_RANGE_M = 6.0           # horizontal distance from contact to sensor (inside P2's beam)
INSPECT_ALT_M = SONAR.altitude  # sensor height above seabed
TILT_DEG = SONAR.pitch          # sensor pitched down

# --- Sonar model: everything follows sim/sonar_geometry.py ---------------------------
MAX_RANGE_M = SONAR.max_range
MIN_RANGE_M = SONAR.min_range
HFOV_DEG = SONAR.hori_fov
VFOV_DEG = SONAR.vert_fov
N_BEARINGS = 512
N_RANGE_BINS = 512
GAIN = 3.0                      # soft-saturation gain: flat sand at beam centre ~ 0.3

# --- Scene grid ----------------------------------------------------------------------
GRID_RES_M = 0.025
GRID_HALF_M = INSPECT_RANGE_M + MAX_RANGE_M + 0.5   # covers the whole fan from any side
SCENE_RADIUS_M = 14.0  # real objects within this distance of the contact are included

DIRECTIONS = {"north": (0.0, 1.0), "east": (1.0, 0.0), "south": (0.0, -1.0), "west": (-1.0, 0.0)}

REFLECTIVITY = {"seabed": 0.35, "aerial_bomb": 1.0, "sphere_mine": 1.0, "rock": 0.6, "tyre": 0.45}

# world.json has no sizes, so real objects use P2's CLASS_SPECS (length, width).
CANONICAL_SIZE_M = {cls: (s["length"], s["width"]) for cls, s in CLASS_SPECS.items()}


# =====================================================================================
# Object shapes: height above seabed on a local (u along yaw, v across) grid
# =====================================================================================

def _bomb_height(u, v, length, width):
    """Finned aerial bomb lying on the seabed. Tail at u = -L/2, nose at u = +L/2."""
    r = 0.35 * width                      # body radius (fins span the full width)
    nose_len, tail_len, fin_len = 0.22 * length, 0.25 * length, 0.15 * length
    u0, u1 = -length / 2, length / 2

    rad = np.full_like(u, r)
    tail = u < u0 + tail_len
    rad[tail] = r * (0.55 + 0.45 * (u[tail] - u0) / tail_len)
    nose = u > u1 - nose_len
    t = (u[nose] - (u1 - nose_len)) / nose_len
    rad[nose] = r * np.sqrt(np.clip(1 - t ** 2, 0, 1))
    inside = (u >= u0) & (u <= u1)

    h = np.where(inside & (np.abs(v) < rad), r + np.sqrt(np.clip(rad ** 2 - v ** 2, 0, None)), 0.0)

    fins = (u >= u0) & (u < u0 + fin_len)
    side_fin = fins & (np.abs(v) < width / 2)                      # horizontal plates
    top_fin = fins & (np.abs(v) < 0.02)                            # vertical plate
    h = np.where(side_fin, np.maximum(h, r + 0.015), h)
    h = np.where(top_fin, np.maximum(h, r + width / 2), h)
    return h


def _rock_height(u, v, length, width, rng):
    """Irregular, flattened boulder with a lumpy outline and rough surface."""
    a, b, c = length / 2, width / 2, min(0.55 * width, 0.6)
    phi = np.arctan2(v / b, u / a)
    wobble = np.zeros_like(u)
    for k in (2, 3, 5):
        wobble += rng.uniform(0.04, 0.09) * np.sin(k * phi + rng.uniform(0, 2 * np.pi))
    rr = np.sqrt((u / a) ** 2 + (v / b) ** 2) * (1 + wobble)
    h = np.where(rr < 1, c * np.clip(1 - rr ** 2, 0, None) ** 0.6, 0.0)
    rough = rng.normal(0, 0.012, size=u.shape)
    return np.where(h > 0, np.clip(h + rough, 0, None), 0.0)


def _sphere_height(u, v, length, width):
    rad = max(length, width) / 2
    d2 = u ** 2 + v ** 2
    return np.where(d2 < rad ** 2, rad + np.sqrt(np.clip(rad ** 2 - d2, 0, None)), 0.0)


def _tyre_height(u, v, length, width):
    outer = max(length, width) / 2
    tube = min(0.12, 0.3 * outer)
    ring = np.abs(np.sqrt(u ** 2 + v ** 2) - (outer - tube))
    return np.where(ring < tube, 0.6 * (tube + np.sqrt(np.clip(tube ** 2 - ring ** 2, 0, None))), 0.0)


def _object_height(cls, u, v, length, width, rng):
    if cls == "aerial_bomb":
        return _bomb_height(u, v, length, width)
    if cls == "rock":
        return _rock_height(u, v, length, width, rng)
    if cls == "sphere_mine":
        return _sphere_height(u, v, length, width)
    if cls == "tyre":
        return _tyre_height(u, v, length, width)
    raise ValueError(f"unknown object class {cls!r}")


# =====================================================================================
# Scene: heightfield + reflectivity grid
# =====================================================================================

class Scene:
    """Heightfield Z and reflectivity R on a square grid centred on (cx, cy)."""

    def __init__(self, cx, cy, seed=0, ripples=0.0):
        self.cx, self.cy = cx, cy
        n = int(round(2 * GRID_HALF_M / GRID_RES_M))
        self.n = n
        self.x0, self.y0 = cx - GRID_HALF_M, cy - GRID_HALF_M
        self.rng = np.random.default_rng(seed)
        self.Z = np.zeros((n, n), dtype=np.float32)        # Z[row=y, col=x]
        self.R = np.full((n, n), REFLECTIVITY["seabed"], dtype=np.float32)
        if ripples > 0:
            ys, xs = np.mgrid[0:n, 0:n] * GRID_RES_M
            ang = self.rng.uniform(0, np.pi)
            wave = np.sin((xs * np.cos(ang) + ys * np.sin(ang)) * 2 * np.pi / 0.8)
            self.Z += (ripples * wave + self.rng.normal(0, 0.004, (n, n))).astype(np.float32)

    def add(self, cls, xy, yaw, size, burial=0.0):
        length, width = size
        reach = 0.5 * np.hypot(length, width) + 0.1
        c0 = int((xy[0] - reach - self.x0) / GRID_RES_M)
        c1 = int((xy[0] + reach - self.x0) / GRID_RES_M) + 1
        r0 = int((xy[1] - reach - self.y0) / GRID_RES_M)
        r1 = int((xy[1] + reach - self.y0) / GRID_RES_M) + 1
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, self.n), min(r1, self.n)
        if c0 >= c1 or r0 >= r1:
            return
        rows, cols = np.mgrid[r0:r1, c0:c1]
        dx = self.x0 + cols * GRID_RES_M - xy[0]
        dy = self.y0 + rows * GRID_RES_M - xy[1]
        u = dx * np.cos(yaw) + dy * np.sin(yaw)
        v = -dx * np.sin(yaw) + dy * np.cos(yaw)
        h = _object_height(cls, u, v, length, width, self.rng)
        h = h - burial * h.max() if h.max() > 0 else h        # sink by a fraction of height
        top = h > 0.005
        patch_z = self.Z[r0:r1, c0:c1]
        patch_z[top] = np.maximum(patch_z[top], h[top])
        self.R[r0:r1, c0:c1][top] = REFLECTIVITY[cls]


# =====================================================================================
# Sonar rendering
# =====================================================================================

def _box_blur(a, ky, kx):
    """Separable box blur with edge padding (odd kernel sizes)."""
    out = a
    for axis, k in ((0, ky), (1, kx)):
        if k <= 1:
            continue
        pad = k // 2
        p = np.pad(out, [(pad, pad) if ax == axis else (0, 0) for ax in range(2)], mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float64)
        c = np.insert(c, 0, 0, axis=axis)
        hi = np.take(c, np.arange(k, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - k), axis=axis)
        out = ((hi - lo) / k).astype(np.float32)
    return out


def sensor_pose(contact_xy, direction):
    """Sensor x, y and heading (rad) for viewing the contact from a compass direction."""
    dx, dy = DIRECTIONS[direction]
    sx = contact_xy[0] + INSPECT_RANGE_M * dx
    sy = contact_xy[1] + INSPECT_RANGE_M * dy
    return sx, sy, float(np.arctan2(-dy, -dx))


def render_polar(scene, sx, sy, heading):
    """Noise-free sonar intensity in polar form [bearing, range_bin], values 0..1."""
    hfov, tilt = np.radians(HFOV_DEG), np.radians(TILT_DEG)
    sigma = np.radians(VFOV_DEG) / 2.355
    offsets = np.linspace(hfov / 2, -hfov / 2, N_BEARINGS)        # row 0 = sensor's left
    bearings = heading + offsets
    hs = np.arange(0.05, MAX_RANGE_M, GRID_RES_M * 0.8, dtype=np.float32)

    cb, sb = np.cos(bearings)[:, None], np.sin(bearings)[:, None]
    px, py = sx + cb * hs, sy + sb * hs
    col = np.clip(((px - scene.x0) / GRID_RES_M).astype(int), 0, scene.n - 1)
    row = np.clip(((py - scene.y0) / GRID_RES_M).astype(int), 0, scene.n - 1)

    gy, gx = np.gradient(scene.Z, GRID_RES_M)
    z = scene.Z[row, col]
    refl = scene.R[row, col]
    nx, ny = -gx[row, col], -gy[row, col]
    nnorm = np.sqrt(nx ** 2 + ny ** 2 + 1)

    dz = INSPECT_ALT_M - z
    slant = np.sqrt(hs ** 2 + dz ** 2)
    depression = np.arctan2(dz, hs)
    visible = depression <= np.minimum.accumulate(depression, axis=1) + 1e-6   # shadows
    # ray from sensor to point: (cb*h, sb*h, -dz) / slant ; returns ~ -n . ray
    cos_inc = (-(nx * cb * hs + ny * sb * hs) + dz) / (slant * nnorm)
    beam = np.exp(-0.5 * ((depression - tilt) / sigma) ** 2)
    intensity = refl * np.clip(cos_inc, 0, 1) * beam * visible

    rbin = ((slant - MIN_RANGE_M) / (MAX_RANGE_M - MIN_RANGE_M) * N_RANGE_BINS).astype(int)
    ok = (rbin >= 0) & (rbin < N_RANGE_BINS)
    polar = np.zeros((N_BEARINGS, N_RANGE_BINS), dtype=np.float32)
    rows_idx = np.broadcast_to(np.arange(N_BEARINGS)[:, None], rbin.shape)
    np.add.at(polar, (rows_idx[ok], rbin[ok]), intensity[ok])
    polar = _box_blur(polar, 3, 3)
    return 1.0 - np.exp(-GAIN * polar)


def add_speckle(polar, seed):
    """Multiplicative speckle + noise floor, for display images only."""
    rng = np.random.default_rng(seed)
    speckle = _box_blur(rng.exponential(1.0, polar.shape).astype(np.float32), 1, 3)
    noisy = polar * speckle + rng.normal(0.03, 0.015, polar.shape)
    return np.clip(noisy, 0, 1)


def to_image(polar):
    """Polar intensity [bearing (row 0 = port), range] in 0..1 -> 512x512 uint8 fan image.

    Uses P2's polar_to_fan, so imagined and inspection images share the survey frames'
    layout and metres-per-pixel scale.
    """
    return polar_to_fan(np.clip(polar, 0, 1).T * 255.0, SONAR)


# =====================================================================================
# Public API used by agent/imagine.py
# =====================================================================================

def hypothesis_scene(contact, cls, yaw=None, seed=0):
    """Clean scene: flat seabed + one object of class `cls` built from the contact only."""
    x, y = contact["xy"]
    scene = Scene(x, y, seed=seed)
    if cls is not None:
        scene.add(cls, (x, y), contact.get("yaw", 0.0) if yaw is None else yaw, contact["size_m"])
    return scene


def real_scene(run_dir, contact, seed=0):
    """Real scene around the contact from world.json (sim truth; never used for imagining)."""
    with open(os.path.join(run_dir, "world.json")) as f:
        world = json.load(f)
    x, y = contact["xy"]
    ripples = 0.012 if world.get("scenario") == "half_buried" else 0.006
    scene = Scene(x, y, seed=seed, ripples=ripples)
    for obj in world["objects"]:
        ox, oy = obj["xyz"][0], obj["xyz"][1]
        if np.hypot(ox - x, oy - y) <= SCENE_RADIUS_M:
            scene.add(obj["cls"], (ox, oy), obj.get("yaw", 0.0),
                      CANONICAL_SIZE_M[obj["cls"]], obj.get("burial", 0.0))
    return scene


def render_view(scene, contact_xy, direction, backend="proxy"):
    """Noise-free polar sonar view of `scene` from `direction`."""
    if backend != "proxy":
        raise NotImplementedError(
            f"backend {backend!r} not wired up yet; OceanSim lands after P2's image passes step 3")
    sx, sy, heading = sensor_pose(contact_xy, direction)
    return render_polar(scene, sx, sy, heading)
