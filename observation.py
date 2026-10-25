#!/usr/bin/env python3
"""
Builds the policy's observation from what the real sub measures: the pressure
sensor, IMU/magnetometer and the camera detector's boxes. Shared with the
Raspberry Pi (numpy + standard library only), so training and the real sub
compute exactly the same thing.

One frame of the observation (all roughly in [-1, 1]); FRAME_FIELDS names each value:
  own state    depth, depth rate, sin/cos of heading relative to the start heading,
               pitch, pitch rate, yaw rate
  left pole,   found now, bearing, range, range confidence, time since last seen.
  right pole   Each keeps its identity when only one is in view (close to the
               gate both no longer fit), so the policy knows which way to steer.
  gate         the same five, plus the turn needed to face straight through the
               gate and the sideways distance from its center line. A pair counts
               as the gate once seen GATE_CONFIRM_FRAMES frames in a row; until then
               its boxes are treated as lone poles
  marker       (optional, for the orbit skill) the green marker pole: the same five.
               A sighting counts once seen MARKER_CONFIRM_FRAMES frames in a row,
               which filters out single false detections
  last action  whatever the policy commanded last step (3 values, or 4 with sway)
The policy sees the last `stack` frames concatenated, newest first, so it can
sense its own motion from how the detections change.

When something isn't seen, its last values are kept (bearings and the approach
angle corrected for how much the sub has turned since), with found = 0 and the
time since seen counting up. Before the first sighting everything is 0 and
the time since seen is at its maximum.

Heading is relative to the heading at reset, not absolute, so the policy can't
memorize which compass direction the gate is in.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from perception import (CameraModel, Estimate, bearing_deg, estimate_gate, estimate_pole,
                        find_gate_pair, gate_geometry)

# Scales that bring each value to roughly [-1, 1]
DEPTH_SCALE = 2.0         # m
DEPTH_RATE_SCALE = 0.5    # m/s
PITCH_SCALE = 30.0        # deg
RATE_SCALE = 1.0          # rad/s
BEARING_SCALE = 35.0      # deg (half the field of view); remembered bearings can exceed 1
RANGE_SCALE = 10.0        # m
APPROACH_SCALE = 90.0     # deg
OFFSET_SCALE = 2.0        # m
MAX_UNSEEN = 5.0          # s; time since seen is clipped here
MARKER_CONFIRM_FRAMES = 3 # A marker sighting counts only after this many frames in a row, so a single
                          # false detection can't move the remembered marker (seed 10018 hit the marker)
GATE_CONFIRM_FRAMES = 3   # Likewise a gate pair: a phantom red box paired with a real pole gave gate
                          # offsets 2-3 m wrong, and the gate policy steered into the pole

# How much each range method is trusted (perception.Estimate.method)
RANGE_CONFIDENCE = {"poles": 1.0, "bottom": 1.0, "height": 0.8, "separation": 0.6, "width": 0.3, "none": 0.0}

OWN_FIELDS = ["depth", "depth_rate", "sin_heading", "cos_heading", "pitch", "pitch_rate", "yaw_rate"]
TRACK_FIELDS = ["found", "bearing", "range", "confidence", "unseen"]
GATE_EXTRA_FIELDS = ["approach", "offset"]
ACTION_FIELDS = ["action_turn", "action_depth", "action_surge", "action_sway"]


def frame_fields(include_marker: bool = False, action_dim: int = 3) -> List[str]:
    """Names of the values in one frame, in order."""
    fields = (OWN_FIELDS + [f"left_{f}" for f in TRACK_FIELDS] + [f"right_{f}" for f in TRACK_FIELDS]
              + [f"gate_{f}" for f in TRACK_FIELDS + GATE_EXTRA_FIELDS])
    if include_marker:
        fields += [f"marker_{f}" for f in TRACK_FIELDS]
    return fields + ACTION_FIELDS[:action_dim]


FRAME_FIELDS = frame_fields()                              # The gate skill's layout
INDEX = {name: i for i, name in enumerate(FRAME_FIELDS)}   # Position of each value in its newest frame


@dataclass
class CourseGeometry:
    """What the sub knows about the course in advance."""
    gate_width_m: float = 2.0
    pole_diameter_m: float = 0.0334     # 1in PVC
    pole_height_m: float = 1.09         # Visible length, top bar to floor
    pool_depth_m: float = 2.09          # Where the poles' bottom ends are
    marker_diameter_m: float = 0.3
    marker_height_m: float = 2.69       # Floor to its top, 0.6 m above the surface


def _wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


@dataclass
class _Track:
    found: bool = False
    seen_ever: bool = False
    abs_bearing: float = 0.0     # deg, heading-compensated: heading + bearing at sighting
    range_m: float = 0.0
    confidence: float = 0.0
    unseen: float = MAX_UNSEEN
    confirm_frames: int = 1      # Sightings needed in a row before the track takes them
    streak: int = 0
    max_unseen: float = MAX_UNSEEN   # Time since seen stops counting here (its feature keeps the MAX_UNSEEN scale)

    def update(self, est: Optional[Estimate], heading: float, dt: float):
        self.streak = self.streak + 1 if est is not None else 0
        if est is None or self.streak < self.confirm_frames:
            self.found = False
            self.unseen = min(self.max_unseen, self.unseen + dt)
            return
        self.found = self.seen_ever = True
        self.abs_bearing = heading + est.bearing_deg
        if est.range_m is not None:
            self.range_m = est.range_m
        self.confidence = RANGE_CONFIDENCE.get(est.method, 0.0)
        self.unseen = 0.0

    def bearing(self, heading: float) -> float:
        return _wrap(self.abs_bearing - heading)

    def features(self, heading: float) -> List[float]:
        if not self.seen_ever:
            return [0.0, 0.0, 0.0, 0.0, 1.0]
        return [float(self.found),
                float(np.clip(self.bearing(heading) / BEARING_SCALE, -180 / BEARING_SCALE, 180 / BEARING_SCALE)),
                float(np.clip(self.range_m / RANGE_SCALE, 0.0, 3.0)),
                self.confidence,
                self.unseen / MAX_UNSEEN]


@dataclass
class _GateTrack(_Track):
    """
    The gate, plus its orientation. The orientation comes from the difference of
    the two poles' ranges over the 2 m gate width, so per-frame box noise makes
    it jumpy; it's smoothed with a time constant of SMOOTHING_S.
    """
    abs_approach: float = 0.0    # deg, heading-compensated direction straight through the gate
    offset_m: float = 0.0
    geometry_known: bool = False
    geometry_age: float = 0.0    # s since the orientation was last measured

    SMOOTHING_S = 1.0

    def update_geometry(self, approach_deg: Optional[float], offset_m: Optional[float], heading: float, dt: float):
        if approach_deg is None:
            self.geometry_age += dt
            return
        abs_approach = heading + approach_deg
        if not self.geometry_known or self.geometry_age > self.SMOOTHING_S:
            self.abs_approach, self.offset_m = abs_approach, offset_m   # Start fresh
        else:
            k = dt / (self.SMOOTHING_S + dt)
            self.abs_approach += k * _wrap(abs_approach - self.abs_approach)
            self.offset_m += k * (offset_m - self.offset_m)
        self.geometry_known, self.geometry_age = True, 0.0

    def features(self, heading: float) -> List[float]:
        extra = [0.0, 0.0]
        if self.geometry_known:
            extra = [_wrap(self.abs_approach - heading) / APPROACH_SCALE,
                     float(np.clip(self.offset_m / OFFSET_SCALE, -3.0, 3.0))]
        return super().features(heading) + extra


class ObservationBuilder:
    def __init__(self, camera: CameraModel, course: Optional[CourseGeometry] = None,
                 action_dim: int = 3, stack: int = 4, include_marker: bool = False):
        assert action_dim in (3, 4)
        self.camera = camera
        self.course = course or CourseGeometry()
        self.stack = stack
        self.include_marker = include_marker
        self.fields = frame_fields(include_marker, action_dim)
        self.index = {name: i for i, name in enumerate(self.fields)}
        self.frame_size = len(self.fields)
        self.size = self.frame_size * stack
        self.reset()

    def reset(self, start_heading: Optional[float] = None):
        """start_heading: heading (deg) that relative heading is measured from; None = first reading."""
        self.start_heading = start_heading
        self.left, self.right, self.gate = _Track(), _Track(), _GateTrack()
        self.marker = _Track(confirm_frames=MARKER_CONFIRM_FRAMES)
        self.pair_streak = 0
        self.frames: deque = deque(maxlen=self.stack)

    @property
    def tracks(self):
        tracks = {"left": self.left, "right": self.right, "gate": self.gate}
        if self.include_marker:
            tracks["marker"] = self.marker
        return tracks

    # --- Detections -> estimates ---
    def _estimate_pole(self, box: dict, depth: float, pitch: float) -> Estimate:
        c = self.course
        return estimate_pole(box, self.camera, diameter_m=c.pole_diameter_m, height_m=c.pole_height_m,
                             bottom_depth_m=c.pool_depth_m, camera_depth_m=depth, pitch_deg=pitch)

    def _single_pole_side(self, box: dict, heading: float) -> str:
        """Which gate pole a lone red box is: the one whose remembered bearing it's closest to."""
        abs_bearing = heading + bearing_deg(self.camera, box['center_x'])
        if self.left.seen_ever and self.right.seen_ever:
            d_left = abs(_wrap(abs_bearing - self.left.abs_bearing))
            d_right = abs(_wrap(abs_bearing - self.right.abs_bearing))
            return "left" if d_left < d_right else "right"
        if self.gate.seen_ever:
            return "left" if _wrap(abs_bearing - self.gate.abs_bearing) > 0 else "right"
        # No memory: the other pole is probably beyond the nearer image edge, so a
        # pole in the left half of the image is the gate's right pole, and vice versa.
        return "right" if box['center_x'] < self.camera.width / 2 else "left"

    def _update_tracks(self, red_boxes: Sequence[dict], depth: float, pitch: float, heading: float, dt: float):
        left_est = right_est = gate_est = geometry = None
        pair = find_gate_pair(list(red_boxes), self.camera.height)
        self.pair_streak = self.pair_streak + 1 if pair else 0
        if pair and self.pair_streak < GATE_CONFIRM_FRAMES:
            pair = None   # Not confirmed yet: the boxes are handled as lone poles below
        if pair:
            left_est = self._estimate_pole(pair[0], depth, pitch)
            right_est = self._estimate_pole(pair[1], depth, pitch)
            gate_est = estimate_gate(pair[0], pair[1], self.camera, self.course.gate_width_m, left_est, right_est)
            good = ("bottom", "height")
            if left_est.method in good and right_est.method in good:
                geometry = gate_geometry(left_est, right_est)
        elif red_boxes:
            box = max(red_boxes, key=lambda b: b['area'])
            est = self._estimate_pole(box, depth, pitch)
            if self._single_pole_side(box, heading) == "left":
                left_est = est
            else:
                right_est = est
        self.left.update(left_est, heading, dt)
        self.right.update(right_est, heading, dt)
        self.gate.update(gate_est, heading, dt)
        self.gate.update_geometry(geometry.approach_deg if geometry else None,
                                  geometry.offset_m if geometry else None, heading, dt)

    def _update_marker(self, green_boxes: Sequence[dict], depth: float, pitch: float, heading: float, dt: float):
        """The marker is the biggest tall green box, as in the scripted Vision."""
        tall = [b for b in green_boxes if b['height'] > b['width'] * 1.2]
        est = None
        if tall:
            c = self.course
            est = estimate_pole(max(tall, key=lambda b: b['area']), self.camera, diameter_m=c.marker_diameter_m,
                                height_m=c.marker_height_m, bottom_depth_m=c.pool_depth_m,
                                camera_depth_m=depth, pitch_deg=pitch)
        self.marker.update(est, heading, dt)

    # --- Main entry point ---
    def update(self, depth: float, depth_rate: float, heading: float, pitch: float,
               pitch_rate: float, yaw_rate: float, red_boxes: Sequence[dict],
               last_action: Sequence[float], dt: float, green_boxes: Sequence[dict] = ()) -> np.ndarray:
        """Adds this step's frame and returns the stacked observation (float32, length self.size)."""
        if self.start_heading is None:
            self.start_heading = heading
        self._update_tracks(red_boxes, depth, pitch, heading, dt)
        if self.include_marker:
            self._update_marker(green_boxes, depth, pitch, heading, dt)

        rel_heading = math.radians(heading - self.start_heading)
        frame = [depth / DEPTH_SCALE, depth_rate / DEPTH_RATE_SCALE,
                 math.sin(rel_heading), math.cos(rel_heading),
                 pitch / PITCH_SCALE, pitch_rate / RATE_SCALE, yaw_rate / RATE_SCALE]
        frame += self.left.features(heading) + self.right.features(heading) + self.gate.features(heading)
        if self.include_marker:
            frame += self.marker.features(heading)
        frame += [float(a) for a in last_action]
        assert len(frame) == self.frame_size, (len(frame), self.frame_size)

        frame = np.asarray(frame, dtype=np.float32)
        if not self.frames:
            for _ in range(self.stack - 1):
                self.frames.append(frame)   # Fill the history with the first frame
        self.frames.append(frame)
        return np.concatenate(list(reversed(self.frames)))   # Newest first
