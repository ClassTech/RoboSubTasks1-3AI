#!/usr/bin/env python3
"""
Turns camera bounding boxes into bearing and range. This file is shared with the
Raspberry Pi, so it uses only the standard library: no pygame, numpy or OpenCV.

Boxes are dicts in the find_blobs_hsv() format (center_x, center_y, width,
height, min_x, max_x, min_y, max_y, area), in pixels, origin at the top left.

Conventions:
- bearing_deg: angle from the camera's forward axis, positive = LEFT, which is
  the direction heading increases. So the object's heading is heading + bearing.
- range_m: horizontal distance from the camera to the object.
- Depths are positive down; pitch is positive nose-up (matches the simulator).
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CameraModel:
    width: int = 320
    height: int = 240
    fov_deg: float = 70.0   # Horizontal field of view

    @property
    def focal_px(self) -> float:
        return self.width / (2 * math.tan(math.radians(self.fov_deg / 2)))


@dataclass
class Estimate:
    bearing_deg: float
    range_m: Optional[float]  # None if no method applied
    method: str               # Which method gave the range: bottom, height, width, separation, poles


def bearing_deg(cam: CameraModel, pixel_x: float) -> float:
    return math.degrees(math.atan((cam.width / 2 - pixel_x) / cam.focal_px))


def _below_horizon_rad(cam: CameraModel, pixel_y: float, pitch_deg: float) -> float:
    """Angle below horizontal of a ray through image row pixel_y."""
    return math.atan((pixel_y - cam.height / 2) / cam.focal_px) - math.radians(pitch_deg)


def estimate_pole(box: dict, cam: CameraModel, diameter_m: float,
                  height_m: Optional[float] = None,
                  bottom_depth_m: Optional[float] = None,
                  camera_depth_m: Optional[float] = None,
                  pitch_deg: float = 0.0,
                  edge_margin_px: int = 1,
                  min_angle_deg: float = 3.0,
                  min_drop_m: float = 0.4) -> Estimate:
    """
    Bearing and range to a vertical pole. Tries, most accurate first:
    1. bottom: the pole's bottom end is in view and its depth is known (e.g. the
       pool floor), so range follows from the camera depth and the angle down to it.
       Only when the camera is at least min_drop_m above the bottom end: closer
       than that, the depth sensor's zero error (up to 2 cm) alone is a large share
       of the drop, and a sub flying 0.2 m off the floor got gate offsets 2-3 m wrong.
    2. height: both ends are in view and the visible length is known.
    3. width: the pole's diameter. Always available but noisy for thin poles.
    """
    b = bearing_deg(cam, box['center_x'])
    cos_b = math.cos(math.radians(b))
    f = cam.focal_px
    bottom_in_view = box['max_y'] < cam.height - edge_margin_px
    top_in_view = box['min_y'] > edge_margin_px

    if bottom_in_view and bottom_depth_m is not None and camera_depth_m is not None:
        drop = bottom_depth_m - camera_depth_m
        angle = _below_horizon_rad(cam, box['max_y'], pitch_deg)
        if drop >= min_drop_m and angle > math.radians(min_angle_deg):
            return Estimate(b, drop / math.tan(angle) / cos_b, "bottom")

    if bottom_in_view and top_in_view and height_m is not None and box['height'] > 0:
        return Estimate(b, f * height_m / box['height'] / cos_b, "height")

    if box['width'] > 0:
        return Estimate(b, f * diameter_m / box['width'] / cos_b, "width")
    return Estimate(b, None, "none")


def estimate_gate(left: dict, right: dict, cam: CameraModel, gate_width_m: float,
                  left_est: Optional[Estimate] = None,
                  right_est: Optional[Estimate] = None) -> Estimate:
    """
    Bearing and range to the middle of the gate, from its two pole boxes.
    If both poles have good ranges (bottom or height methods), the midpoint is
    found from the two pole positions. Otherwise range comes from the angle
    between the poles and the known gate width, which assumes a head-on view
    and overestimates range when viewed at an angle.
    """
    good = ("bottom", "height")
    if left_est and right_est and left_est.method in good and right_est.method in good:
        pts = []
        for est in (left_est, right_est):
            a = math.radians(est.bearing_deg)
            pts.append((est.range_m * math.cos(a), est.range_m * math.sin(a)))
        mx, my = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
        return Estimate(math.degrees(math.atan2(my, mx)), math.hypot(mx, my), "poles")

    b_left = bearing_deg(cam, left['center_x'])
    b_right = bearing_deg(cam, right['center_x'])
    half_angle = math.radians(abs(b_left - b_right)) / 2
    b = (b_left + b_right) / 2
    if half_angle <= 0:
        return Estimate(b, None, "none")
    return Estimate(b, (gate_width_m / 2) / math.tan(half_angle), "separation")


def find_gate_pair(red_boxes: List[dict], image_height: int) -> Optional[Tuple[dict, dict]]:
    """
    Picks the two red boxes most likely to be the gate's poles: tall and thin,
    side by side, overlapping vertically, with similar heights. Returns
    (left box, right box) in image order, or None.
    """
    poles = sorted((b for b in red_boxes if b['height'] > b['width'] * 1.5), key=lambda b: b['center_x'])
    # Thresholds were tuned in pixels on a 240-tall image.
    min_overlap = image_height * 10 / 240
    max_height_diff = image_height * 150 / 240
    best, best_diff = None, float('inf')
    for p1, p2 in zip(poles, poles[1:]):
        height_diff = abs(p1['height'] - p2['height'])
        y_overlap = min(p1['max_y'], p2['max_y']) - max(p1['min_y'], p2['min_y'])
        if y_overlap > min_overlap and height_diff < max_height_diff and height_diff < best_diff:
            best, best_diff = (p1, p2), height_diff
    return best


@dataclass
class GateGeometry:
    bearing_deg: float        # To the gate's center (+ = left)
    range_m: float            # To the gate's center
    approach_deg: float       # Turn needed to face straight through the gate (+ = left)
    offset_m: float           # Sideways distance from the gate's center line (+ = sub is left of it)


def gate_geometry(left_est: Estimate, right_est: Estimate) -> Optional[GateGeometry]:
    """
    Where the gate is and how it's oriented, from the two poles' bearings and
    ranges (left/right as seen in the image). None unless both have a range.
    Positions are in the sub's frame: x forward, y left.
    """
    if left_est.range_m is None or right_est.range_m is None:
        return None
    pts = []
    for est in (left_est, right_est):
        a = math.radians(est.bearing_deg)
        pts.append((est.range_m * math.cos(a), est.range_m * math.sin(a)))
    (lx, ly), (rx, ry) = pts
    cx, cy = (lx + rx) / 2, (ly + ry) / 2
    ux, uy = lx - rx, ly - ry                    # Along the gate, right pole to left pole
    length = math.hypot(ux, uy)
    if length < 1e-6:
        return None
    ux, uy = ux / length, uy / length
    nx, ny = uy, -ux                             # Normal to the gate line...
    if nx * cx + ny * cy < 0:                    # ...pointing away from the sub, through the gate
        nx, ny = -nx, -ny
    return GateGeometry(bearing_deg=math.degrees(math.atan2(cy, cx)), range_m=math.hypot(cx, cy),
                        approach_deg=math.degrees(math.atan2(ny, nx)),
                        offset_m=-(cx * ux + cy * uy))
