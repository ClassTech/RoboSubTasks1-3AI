#!/usr/bin/env python3
"""
Observations for the tasks 1-3 skills, built only from what the real sub
measures: the pressure sensor, IMU/magnetometer and the detector's red, black
and white boxes (via tasks_perception). numpy and the standard library only, so
it runs on the Pi unchanged. Same style as observation.py (the qualification
skills): tracks that remember a target when it drops out of view (bearings
corrected for turning), sightings confirmed over several frames, geometry
smoothed over ~1 s, and the last `stack` frames concatenated, newest first.

Both skills take a side to go through as an input: +1 = left, -1 = right,
relative to the sub. On the way back, the same physical lane is the other side.
Neither has sway: with it, the first gate policies went through crabwise and
hit the divider they couldn't see.

Gate frame (40 values; the gate skill has no sway, so 3 actions):
  own state (7)     depth, depth rate, sin/cos of heading relative to the start,
                    pitch, pitch rate, yaw rate
  side (1)
  left pole (5)     found now, bearing, range, range confidence, time since seen.
  right pole (5)    The gate's two poles, left and right as the sub sees them now.
                    Each is tracked by its physical identity (red over black is
                    the course's right pole), so memory never mixes them up.
  divider (5)       the red divider at the gate's center
  gate (7)          the gate's center: the same five, plus the turn to face
                    straight through and the offset from its center line
  passage (7)       the commanded passage (between its pole and the divider):
                    the same seven
  last action (3)   turn, depth, surge

Slalom frame (32 values; no sway either, so 3 actions):
  own state (7), side (1)
  red (5)           the nearest set's red pole
  our white (5)     the white pole on the commanded side of it
  other white (5)
  lane (5)          our lane: found now, bearing to its center (between the red
                    and our white: precise, from image positions), its width in
                    degrees (a range cue), and the smoothed turn and offset
                    (from the two poles' ranges over the 1.52 m lane: noisy, so
                    a hint more than a guide)
  red travel (1)    meters driven since the red pole was last in view, dead-reckoned
                    from the surge commands at the sim's top speed (the red's range
                    is the range when last seen, so together: how far through)
  last action (3)
"""
import math
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

from observation import (APPROACH_SCALE, BEARING_SCALE, DEPTH_RATE_SCALE, DEPTH_SCALE, MAX_UNSEEN,
                         OFFSET_SCALE, PITCH_SCALE, RANGE_CONFIDENCE, RANGE_SCALE, RATE_SCALE,
                         _GateTrack, _Track, _wrap)
from perception import CameraModel, Estimate, bearing_deg
from tasks_perception import TasksGeometry, find_gate, find_slalom_sets

CONFIRM_FRAMES = 3   # As on the qualification course: one-frame phantoms must not move what's remembered
GATE_CONFIRM_FRAMES = 2   # The whole gate and its passages need both poles in the same frame, so every
                          # one-frame dropout cost 4 frames at 3 (confirmed only 42% of the time, against
                          # 66% seen); a phantom gate would need a false red and black box stacked end to end
WIDTH_SCALE = 30.0   # deg, lane width cue
SET_JUMP_M = 2.0
TOP_SPEED_MPS = 0.34   # Full surge, in the sim (the real sub's speed isn't measured yet): scales the dead reckoning
TRAVEL_SCALE_M = 3.0     # Half the 4 m between slalom sets: a set this much farther than the tracked one is the next

OWN_FIELDS = ["depth", "depth_rate", "sin_heading", "cos_heading", "pitch", "pitch_rate", "yaw_rate"]
TRACK = ["found", "bearing", "range", "confidence", "unseen"]
GEO = ["turn", "offset"]
ACTIONS = ["action_turn", "action_depth", "action_surge", "action_sway"]
GATE_FIELDS = (OWN_FIELDS + ["side"] + [f"left_pole_{f}" for f in TRACK] + [f"right_pole_{f}" for f in TRACK]
               + [f"divider_{f}" for f in TRACK] + [f"gate_{f}" for f in TRACK + GEO]
               + [f"passage_{f}" for f in TRACK + GEO] + ACTIONS[:3])
SLALOM_FIELDS = (OWN_FIELDS + ["side"] + [f"red_{f}" for f in TRACK] + [f"our_white_{f}" for f in TRACK]
                 + [f"other_white_{f}" for f in TRACK]
                 + ["lane_found", "lane_bearing", "lane_width", "lane_turn", "lane_offset", "red_travel"] + ACTIONS[:3])


class _LaneTrack(_GateTrack):
    """A slalom lane: bearing to its center from the two poles' image positions, plus smoothed geometry."""
    width_deg: float = 0.0

    def lane_features(self, heading: float) -> List[float]:
        if not self.seen_ever:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        geo = [0.0, 0.0]
        if self.geometry_known:
            geo = [_wrap(self.abs_approach - heading) / APPROACH_SCALE,
                   float(np.clip(self.offset_m / OFFSET_SCALE, -3.0, 3.0))]
        return [float(self.found),
                float(np.clip(self.bearing(heading) / BEARING_SCALE, -180 / BEARING_SCALE, 180 / BEARING_SCALE)),
                self.width_deg / WIDTH_SCALE] + geo


class TasksObservationBuilder:
    def __init__(self, camera: CameraModel, skill: str, geom: Optional[TasksGeometry] = None, stack: int = 4):
        assert skill in ("gate", "slalom"), skill
        self.camera, self.skill, self.geom, self.stack = camera, skill, geom or TasksGeometry(), stack
        self.fields = GATE_FIELDS if skill == "gate" else SLALOM_FIELDS
        self.index = {name: i for i, name in enumerate(self.fields)}
        self.frame_size = len(self.fields)
        self.size = self.frame_size * stack
        self.reset()

    def reset(self, start_heading: Optional[float] = None):
        self.start_heading = start_heading
        c = CONFIRM_FRAMES
        # Gate: poles by physical identity ('right' = red over black, as seen from the start)
        self.course_poles = {"left": _Track(confirm_frames=c), "right": _Track(confirm_frames=c)}
        self.divider = _Track(confirm_frames=c)
        g = GATE_CONFIRM_FRAMES
        self.gate = _GateTrack(confirm_frames=g)
        self.passages = {"left": _GateTrack(confirm_frames=g), "right": _GateTrack(confirm_frames=g)}
        # Slalom: the nearest set
        self.red = _Track(confirm_frames=c)
        self.red_travel = 0.0   # m driven since the red pole was last in view, dead-reckoned from surge commands
        self.whites = {"left": _Track(confirm_frames=c), "right": _Track(confirm_frames=c)}
        self.lanes = {"left": _LaneTrack(confirm_frames=c), "right": _LaneTrack(confirm_frames=c)}
        self.frames: deque = deque(maxlen=self.stack)

    # --- Perception -> tracks ---
    def _update_gate(self, d, depth, pitch, heading, dt):
        gate = find_gate(d.red_blobs, d.black_blobs, self.camera, depth, pitch, self.geom)
        seen = {p.course_side: p for p in gate.poles}
        for side, track in self.course_poles.items():
            track.update(seen[side].estimate if side in seen else None, heading, dt)
        div = None
        if gate.divider is not None:
            div = Estimate(bearing_deg(self.camera, gate.divider['center_x']), None, "none")
        self.divider.update(div, heading, dt)
        g = gate.geometry
        self.gate.update(Estimate(g.bearing_deg, g.range_m, "poles") if g else None, heading, dt)
        self.gate.update_geometry(g.approach_deg if g else None, g.offset_m if g else None, heading, dt)
        for side, track in self.passages.items():
            p = gate.passages.get(side)
            track.update(Estimate(p.bearing_deg, p.range_m, "poles") if p else None, heading, dt)
            track.update_geometry(p.approach_deg if p else None, p.offset_m if p else None, heading, dt)

    def _update_slalom(self, d, depth, pitch, heading, dt):
        gate = find_gate(d.red_blobs, d.black_blobs, self.camera, depth, pitch, self.geom)
        sets = find_slalom_sets(d.red_blobs, d.white_blobs, self.camera, depth, pitch, self.geom,
                                exclude=gate.used_reds)
        near = self._same_set(sets)
        self.red.update(near.red.estimate if near else None, heading, dt)
        for side in ("left", "right"):
            white = near.whites.get(side) if near else None
            self.whites[side].update(white.estimate if white else None, heading, dt)
            lane, track = (near.lanes.get(side) if near else None), self.lanes[side]
            if near and white:
                b_red, b_white = near.red.estimate.bearing_deg, white.estimate.bearing_deg
                track.width_deg = abs(b_red - b_white)
                rng = lane.range_m if lane else None
                track.update(Estimate((b_red + b_white) / 2, rng, "poles"), heading, dt)
            else:
                track.update(None, heading, dt)
            track.update_geometry(lane.approach_deg if lane else None, lane.offset_m if lane else None, heading, dt)

    def _same_set(self, sets):
        """The nearest set, unless a set is already tracked: then only one within SET_JUMP_M of its range,
        until dead reckoning says the sub has driven past the tracked set (as far as it was when last seen).
        Without this, when the near set's outer white left the view (a set needs a white), the next set,
        4 m farther, became 'the nearest set' and the policy steered for its lane, into the near set's white."""
        if not sets:
            return None
        known = self.red.seen_ever and self.red.range_m is not None
        through = known and not self.red.found and self.red_travel >= self.red.range_m
        for s in sets:   # Nearest first
            r = s.red.estimate.range_m
            if not known or through or r is None or r < self.red.range_m + SET_JUMP_M:
                return s
        return None

    def _left_right(self, heading: float) -> List[float]:
        """The two physical poles' features, as left and right the way the sub sees them now."""
        poles = [t for t in self.course_poles.values() if t.seen_ever]
        empty = _Track().features(heading)
        if len(poles) == 2:
            a, b = sorted(poles, key=lambda t: -t.bearing(heading))   # More to the left first
            return a.features(heading) + b.features(heading)
        if len(poles) == 1:
            t = poles[0]
            # One pole so far: left of the divider (or, without it, of straight ahead) is the left pole
            ref = self.divider.bearing(heading) if self.divider.seen_ever else 0.0
            if t.bearing(heading) > ref:
                return t.features(heading) + empty
            return empty + t.features(heading)
        return empty + empty

    # --- Main entry point ---
    def update(self, depth: float, depth_rate: float, heading: float, pitch: float, pitch_rate: float,
               yaw_rate: float, detections, side: int, last_action: Sequence[float], dt: float) -> np.ndarray:
        """Adds this step's frame and returns the stacked observation (float32, length self.size).
        side: +1 = go through on the left, -1 = on the right (relative to the sub)."""
        if self.start_heading is None:
            self.start_heading = heading
        if detections is not None:
            (self._update_gate if self.skill == "gate" else self._update_slalom)(detections, depth, pitch, heading, dt)
        rel = math.radians(heading - self.start_heading)
        frame = [depth / DEPTH_SCALE, depth_rate / DEPTH_RATE_SCALE, math.sin(rel), math.cos(rel),
                 pitch / PITCH_SCALE, pitch_rate / RATE_SCALE, yaw_rate / RATE_SCALE, float(side)]
        ours, other = ("left", "right") if side > 0 else ("right", "left")
        if self.skill == "gate":
            frame += self._left_right(heading) + self.divider.features(heading)
            frame += self.gate.features(heading) + self.passages[ours].features(heading)
        else:
            frame += self.red.features(heading) + self.whites[ours].features(heading)
            frame += self.whites[other].features(heading) + self.lanes[ours].lane_features(heading)
            # Dead reckoning since the set was last seen: close to a set its poles leave the view, and without
            # this nothing in the observation changed as the sub crept through blind; it dithered on the line
            # and crossed back (most of the learned mission's failures)
            self.red_travel = 0.0 if self.red.found else self.red_travel + float(last_action[2]) * TOP_SPEED_MPS * dt
            frame += [float(np.clip(self.red_travel, -3.0, 6.0)) / TRAVEL_SCALE_M]
        frame += [float(v) for v in last_action]
        assert len(frame) == self.frame_size, (len(frame), self.frame_size)
        frame = np.asarray(frame, dtype=np.float32)
        if not self.frames:
            for _ in range(self.stack - 1):
                self.frames.append(frame)
        self.frames.append(frame)
        return np.concatenate(list(reversed(self.frames)))
