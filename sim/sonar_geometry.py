"""Sonar frame geometry: OceanSim polar grid -> 512x512 fan image, and pixel <-> world (owner: P2).

Pure NumPy, no Isaac needed, so the detector / planner side can import it on a laptop.

Fan image convention (every runs/<id>/sonar/*.png):
  - 512 x 512 greyscale. The sensor sits at the bottom-centre pixel (col 255.5, row 511).
  - "Up" in the image = the sonar's forward direction (the robot's heading, `yaw` in poses.csv).
  - Distance from the sensor pixel = SLANT range in metres, at a fixed `m_per_px` scale.
  - Image right = starboard (robot's right).
  - Pixels outside the fan are 0.

To get a world position from a detection, call `pixel_to_world(col, row, pose)`; it converts
slant range to ground range using the sensor altitude and rotates by the pose yaw.
"""
from dataclasses import dataclass

import numpy as np

SIZE = 512


@dataclass(frozen=True)
class SonarParams:
    min_range: float = 0.2      # m
    max_range: float = 10.0     # m
    hori_fov: float = 130.0     # deg
    altitude: float = 2.5       # m above seabed (sensor height used for ground-range conversion)

    @property
    def m_per_px(self) -> float:
        half = np.radians(self.hori_fov / 2)
        width_m = 2 * self.max_range * np.sin(half) if half < np.pi / 2 else 2 * self.max_range
        return max(self.max_range / (SIZE - 1), width_m / SIZE)


DEFAULT = SonarParams()


def polar_to_fan(polar: np.ndarray, p: SonarParams = DEFAULT, flip_azimuth: bool = False) -> np.ndarray:
    """OceanSim sonar grid (n_range x n_azimuth[, C], row 0 = min_range) -> 512x512 uint8 fan.

    `flip_azimuth` swaps left/right if OceanSim's azimuth column 0 turns out to be starboard.
    """
    if polar.ndim == 3:
        polar = polar[..., :3].mean(axis=2)
    n_r, n_a = polar.shape
    rows, cols = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    dx = (cols - (SIZE - 1) / 2) * p.m_per_px          # starboard +
    dy = ((SIZE - 1) - rows) * p.m_per_px              # forward +
    r = np.hypot(dx, dy)
    theta = np.degrees(np.arctan2(dx, dy))             # 0 = forward, + = starboard
    ri = (r - p.min_range) / (p.max_range - p.min_range) * (n_r - 1)
    ai = (theta + p.hori_fov / 2) / p.hori_fov * (n_a - 1)
    if flip_azimuth:
        ai = (n_a - 1) - ai
    valid = (ri >= 0) & (ri <= n_r - 1) & (ai >= 0) & (ai <= n_a - 1)
    out = np.zeros((SIZE, SIZE), dtype=np.float32)
    out[valid] = polar[np.round(ri[valid]).astype(int), np.round(ai[valid]).astype(int)]
    return np.clip(out, 0, 255).astype(np.uint8)


def pixel_to_sensor(col: float, row: float, p: SonarParams = DEFAULT) -> tuple[float, float]:
    """Fan pixel -> (forward_m, starboard_m) on the seabed, relative to the sensor."""
    dx = (col - (SIZE - 1) / 2) * p.m_per_px
    dy = ((SIZE - 1) - row) * p.m_per_px
    slant = np.hypot(dx, dy)
    ground = np.sqrt(max(slant**2 - p.altitude**2, 0.0))
    scale = ground / slant if slant > 0 else 0.0
    return dy * scale, dx * scale


def pixel_to_world(col: float, row: float, pose: dict, p: SonarParams = DEFAULT) -> tuple[float, float]:
    """Fan pixel -> world (x, y). `pose` is a poses.csv row with x, y, yaw (radians, 0 = +x, CCW)."""
    fwd, stbd = pixel_to_sensor(col, row, p)
    yaw = float(pose["yaw"])
    # Starboard is the robot's right = -left; left unit vector is (-sin, cos).
    x = float(pose["x"]) + fwd * np.cos(yaw) + stbd * np.sin(yaw)
    y = float(pose["y"]) + fwd * np.sin(yaw) - stbd * np.cos(yaw)
    return x, y


def world_to_pixel(x: float, y: float, pose: dict, p: SonarParams = DEFAULT) -> tuple[float, float]:
    """World (x, y) on the seabed -> fan pixel (col, row). Used to make YOLO labels from world.json."""
    yaw = float(pose["yaw"])
    ddx, ddy = x - float(pose["x"]), y - float(pose["y"])
    fwd = ddx * np.cos(yaw) + ddy * np.sin(yaw)
    stbd = ddx * np.sin(yaw) - ddy * np.cos(yaw)
    ground = np.hypot(fwd, stbd)
    slant = np.hypot(ground, p.altitude)
    scale = slant / ground if ground > 0 else 0.0
    col = (SIZE - 1) / 2 + stbd * scale / p.m_per_px
    row = (SIZE - 1) - fwd * scale / p.m_per_px
    return col, row


if __name__ == "__main__":
    pose = {"x": 10.0, "y": 5.0, "yaw": 0.7}
    for wx, wy in [(15.0, 9.0), (12.0, 3.0), (11.0, 9.5)]:
        c, r = world_to_pixel(wx, wy, pose)
        bx, by = pixel_to_world(c, r, pose)
        assert abs(bx - wx) < 1e-6 and abs(by - wy) < 1e-6, (wx, wy, bx, by)
    fan = polar_to_fan(np.full((725, 260), 200, np.uint8))
    assert fan.shape == (SIZE, SIZE) and fan[SIZE - 50, SIZE // 2] == 200 and fan[0, 0] == 0
    print(f"ok: m_per_px={DEFAULT.m_per_px:.4f}, round trip + fan checks pass")
