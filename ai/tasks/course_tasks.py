#!/usr/bin/env python3
"""
Scripted tasks for RoboSub tasks 1-3 (the divided gate and the slalom), ported
from ClassTech/RoboSim's GateTask, SlalomTask, TimedHoverTask and SurfaceTask.

What changed in the port, and why:
- This sub's helpers and mixer: surge, sway, yaw and heave only. The vertical
  thrusters are on the sides and the sub is passively stable in pitch and roll,
  so RoboSim's pitch and roll loops are gone.
- A fixed depth (the mission's target depth) instead of RoboSim's visual depth
  servo on the gate.
- Timings stretched for the real sub, which tops out at ~0.34 m/s.
- The gate's drive-through is longer: the gate is 3 m wide, so its far pole
  leaves the camera's view ~3.3 m out, and "gate lost" comes that early.
- The gate and slalom approaches square up, slide sideways until the passage
  or lane is centered, then drive straight in on the course heading,
  correcting with sway. RoboSim turned toward the passage instead; from an
  offset start that came in at an angle, which carried the sub into a pole
  once it left the view.
- The slalom goes through the same side as the gate (the user's rule; RoboSim
  chose by where the red pole first appeared). The reversed slalom uses the
  same physical lane, so the opposite side relative to the sub, and waits until
  it has turned around before looking for poles.
- Noise: losing the gate or a slalom set only counts as having reached it if it
  was close when last seen (wide in the image); lost further out, the task
  keeps going straight and waits for it. The slalom only finishes after
  SEARCH_GIVE_UP_S with no set in view. Deciding from a single frame let a
  detector dropout end the gate before it was reached, and the slalom after
  its second set.
- As in the rest of the scripted mission, the damping that stops the sub uses
  ground-truth velocity (SIM ONLY); everything it steers by is measured.

Sides are relative to the sub: 'left' is the camera image's left.
"""
import math
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai.tasks.task_base import Task, TaskStatus
from data_structures import SensorSuite, ThrusterCommands, Vision
from controller import angle_diff

SEARCH_YAW = 0.25            # Turning effort while searching
ALIGN_YAW_P = 0.6            # Per half-image-width of horizontal error
ALIGN_SWAY_P = 1.0
ALIGN_HEADING_TOL_DEG = 3.0
ALIGN_POS_TOL = 0.08         # Fraction of image width


def _image_size(sub) -> Tuple[int, int]:
    return sub.vision.image_width, sub.vision.image_height


def _damping(sub, sensors: SensorSuite, gain: Optional[float] = None) -> Tuple[float, float]:
    """Surge and sway that brake the sub's horizontal motion. SIM ONLY: ground-truth velocity, turned
    into forward and sideways by the true heading (the measured one is off by up to 15 deg per run,
    which made forward speed look sideways and pushed the sub off its line)."""
    k = sub.DAMPING_GAIN if gain is None else gain
    h = math.radians(sensors.true_heading if sensors.true_heading is not None else sensors.heading)
    c, s = math.cos(h), math.sin(h)
    wx, wy = -sensors.velocity_x * k, -sensors.velocity_y * k
    return wx * c + wy * s, -wx * s + wy * c


def _yaw_to_pixel(sub, sensors: SensorSuite, target_x: float) -> float:
    w, _ = _image_size(sub)
    yaw = -((target_x - w / 2) / (w / 2)) * ALIGN_YAW_P - sensors.imu.gyro_z * sub.YAW_D_GAIN
    return float(np.clip(yaw, -1.0, 1.0))


def _drive_to_pixel(sub, sensors: SensorSuite, target_x: float, surge: float, depth: float) -> ThrusterCommands:
    """Point at an image column and drive toward it, damping sideways drift."""
    _, sway = _damping(sub, sensors, sub.MANEUVER_DAMPING_GAIN)
    return sub._mix_and_normalize_commands(surge, sway, _yaw_to_pixel(sub, sensors, target_x),
                                           sub.get_heave_command(sensors, depth))


def _hold_heading(sub, sensors: SensorSuite, heading: float, surge: float, depth: float) -> ThrusterCommands:
    _, sway = _damping(sub, sensors, sub.MANEUVER_DAMPING_GAIN)
    yaw = float(np.clip(angle_diff(heading, sensors.heading) * sub.HOVER_YAW_P_GAIN
                        - sensors.imu.gyro_z * sub.YAW_D_GAIN, -1.0, 1.0))
    return sub._mix_and_normalize_commands(surge, sway, yaw, sub.get_heave_command(sensors, depth))


class _CourseTask(Task):
    """A self-contained state machine (as in RoboSim), run by this repo's Submarine."""
    subtasks: List = []

    def on_enter(self, sub, sensors, vision, context):
        self._start(sub, sensors)
        self._started = True

    def _start(self, sub, sensors):
        pass

    def execute(self, sub, dt, sensors, vision, config):
        if not getattr(self, "_started", False):   # The first task in a plan is never "entered"
            self.on_enter(sub, sensors, vision, self.context)
        return self.step(sub, dt, sensors, vision)

    def step(self, sub, dt, sensors, vision) -> Tuple[TaskStatus, ThrusterCommands]:
        raise NotImplementedError


# --- Gate ---
def find_gate(vision: Vision, image_h: int) -> Optional[Dict]:
    """
    The gate from this frame's boxes, as in RoboSim: each gate pole is a red box
    and a black box stacked end to end (half red, half black), so pair them
    into poles; then the two poles of most similar height are the gate. The red
    divider has no black half, so it is never taken for a pole.
    Returns the gate's x extent and its two passages' image columns, or None.
    """
    reds, blacks = vision.red_blobs, vision.black_blobs
    if not reds or not blacks:
        return None
    gap_tol = 0.03 * image_h
    poles, used = [], set()
    for b in blacks:
        best, best_gap = None, float("inf")
        for i, r in enumerate(reds):
            if i in used:
                continue
            dx = abs(r['center_x'] - b['center_x'])
            gap = min(abs(r['max_y'] - b['min_y']), abs(b['max_y'] - r['min_y']))
            if dx < (r['width'] + b['width']) * 1.5 + 2 and gap < gap_tol and gap < best_gap:
                best, best_gap = i, gap
        if best is not None:
            used.add(best)
            r = reds[best]
            min_y, max_y = min(r['min_y'], b['min_y']), max(r['max_y'], b['max_y'])
            poles.append({'center_x': (min(r['min_x'], b['min_x']) + max(r['max_x'], b['max_x'])) / 2,
                          'min_x': min(r['min_x'], b['min_x']), 'max_x': max(r['max_x'], b['max_x']),
                          'height': max_y - min_y})
    if len(poles) < 2:
        return None
    pair = min(((p, q) for i, p in enumerate(poles) for q in poles[i + 1:]),
               key=lambda pq: abs(pq[0]['height'] - pq[1]['height']))
    if abs(pair[0]['height'] - pair[1]['height']) > 0.15 * image_h:
        return None
    left, right = sorted(pair, key=lambda p: p['center_x'])
    width = right['center_x'] - left['center_x']
    return {'min_x': left['min_x'], 'max_x': right['max_x'],
            'center_x': (left['center_x'] + right['center_x']) / 2,
            'left_passage_x': left['center_x'] + 0.25 * width,
            'right_passage_x': left['center_x'] + 0.75 * width}


class GateState(Enum):
    SEARCHING = auto()
    ALIGNING = auto()     # Turn to face the gate
    SIDESTEP = auto()     # Slide sideways until the chosen passage is centered
    APPROACHING = auto()  # Straight in, correcting with sway
    CLEARING = auto()


class CourseGateTask(_CourseTask):
    """
    Finds the divided gate (turning in place), lines up on it, picks a side and
    drives through it. The first gate pass picks the side whose passage is
    nearer the image center and records it; a later one goes back through the
    same physical opening, which is the other side relative to the sub.
    """
    CLEAR_SECONDS = 16.0       # From losing the gate (~3.3 m out) to ~2 m past it, at full surge
    LOST_TO_CLEAR_SECONDS = 0.75
    CLOSE_WIDTH = 0.45         # Gate wider than this share of the image when last seen = close (under ~5 m)
    LOST_FAR_GIVE_UP_S = 5.0   # Lost while still far: keep going, but search again after this long
    ALIGN_TIMEOUT = 8.0

    def __init__(self, target_depth: float, side: Optional[str] = None):
        """side: 'left' or 'right' (relative to the sub) to always take that passage, as the gate skill's
        training environment asks; None picks one as described above."""
        self.target_depth = target_depth
        self.fixed_side = side
        self.reset()

    def reset(self, search_direction: int = 1):
        super().reset(search_direction)
        self.state = GateState.SEARCHING
        self.timer, self.lost_for = 0.0, 0.0
        self.forced_side, self.side, self.clear_heading = None, None, None
        self.last_width = 0.0      # The gate's width in the image when last seen (px)
        self._started = False

    def _start(self, sub, sensors):
        passed = getattr(sub, "gate_passage_side", None)
        if self.fixed_side:
            self.forced_side = self.fixed_side
        elif passed is not None:
            self.forced_side = "left" if passed == "right" else "right"

    @property
    def state_name(self) -> str:
        return f"Gate: {self.state.name}" + (f" ({self.side})" if self.side else "")

    def step(self, sub, dt, sensors, vision):
        w, h = _image_size(sub)
        gate = find_gate(vision, h)
        heave = sub.get_heave_command(sensors, self.target_depth)

        if self.state == GateState.SEARCHING:
            if gate:
                self.state, self.timer = GateState.ALIGNING, 0.0
            surge, sway = _damping(sub, sensors)
            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, sway, SEARCH_YAW, heave)

        if self.state == GateState.ALIGNING:
            if not gate:
                self.lost_for += dt
                if self.lost_for > 2.0:
                    self.state = GateState.SEARCHING
                return TaskStatus.RUNNING, sub.get_spin_damping_commands(sensors, self.target_depth)
            self.lost_for = 0.0
            self.timer += dt
            surge, sway = _damping(sub, sensors)
            centered = abs(gate['center_x'] - w / 2) < 0.03 * w
            steady = abs(sensors.imu.gyro_z) < 0.05
            if (centered and steady and abs(sensors.depth - self.target_depth) < 0.15) or self.timer > self.ALIGN_TIMEOUT:
                self.state, self.timer, self.clear_heading = GateState.SIDESTEP, 0.0, sensors.heading
                if self.side is None:
                    self.side = self.forced_side or (
                        "left" if abs(gate['left_passage_x'] - w / 2) < abs(gate['right_passage_x'] - w / 2) else "right")
            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(
                surge, sway, _yaw_to_pixel(sub, sensors, gate['center_x']), heave)

        if self.state in (GateState.SIDESTEP, GateState.APPROACHING):
            self.lost_for = 0.0 if gate else self.lost_for + dt
            if gate:
                self.last_width = gate['max_x'] - gate['min_x']
            close = self.last_width > self.CLOSE_WIDTH * w
            if self.state == GateState.APPROACHING and close and self.lost_for > self.LOST_TO_CLEAR_SECONDS:
                self.state, self.timer = GateState.CLEARING, self.CLEAR_SECONDS
            elif (self.state == GateState.SIDESTEP and self.lost_for > 2.0) or self.lost_for > self.LOST_FAR_GIVE_UP_S:
                self.state, self.lost_for = GateState.SEARCHING, 0.0
            elif gate:
                err = (gate['left_passage_x'] if self.side == "left" else gate['right_passage_x']) - w / 2
                if self.state == GateState.SIDESTEP and abs(err) < 0.03 * w:
                    self.state = GateState.APPROACHING
                surge = 1.0 if self.state == GateState.APPROACHING else _damping(sub, sensors)[0]
                _, side_damp = _damping(sub, sensors, sub.MANEUVER_DAMPING_GAIN)
                sway = float(np.clip(-(err / (w / 2)) * ALIGN_SWAY_P + side_damp, -1.0, 1.0))
                yaw = float(np.clip(angle_diff(self.clear_heading, sensors.heading) * sub.HOVER_YAW_P_GAIN
                                    - sensors.imu.gyro_z * sub.YAW_D_GAIN, -1.0, 1.0))
                return TaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, sway, yaw, heave)
            surge = 1.0 if self.state == GateState.APPROACHING else 0.0
            return TaskStatus.RUNNING, _hold_heading(sub, sensors, self.clear_heading, surge, self.target_depth)

        # CLEARING: straight on through, then hand over
        self.timer -= dt
        if self.timer <= 0:
            if getattr(sub, "gate_passage_side", None) is None:
                sub.gate_passage_side = self.side
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        return TaskStatus.RUNNING, _hold_heading(sub, sensors, self.clear_heading, 1.0, self.target_depth)


# --- Slalom ---
class SlalomState(Enum):
    TURNING = auto()
    SEARCHING = auto()
    ALIGNING = auto()
    APPROACH = auto()
    CLEAR = auto()
    REALIGNING = auto()


class SlalomTask(_CourseTask):
    """
    Passes each white-red-white set between its red pole and the white pole on
    the chosen side, nearest set first, until no set is in view. Forward, the
    side is the gate's; reversed, it turns around and uses the same physical
    lane (the other side relative to the sub).
    """
    APPROACH_SURGE = 1.0       # Full thrust (RoboSim used 0.4 of its sub's speed; 0.6 here was slow)
    CLEAR_SECONDS = 7.5        # From losing the set (~1.1 m out) to ~1.5 m past it, at full thrust
    LOST_TO_CLEAR_SECONDS = 0.75
    CLOSE_WIDTH = 0.5          # Lane (red to white) wider than this share of the image when last seen = close
    LOST_FAR_GIVE_UP_S = 5.0
    SEARCH_GIVE_UP_S = 3.0     # No set in view this long: the slalom is done
    SWITCH_FROM_WIDTH = 0.8    # A set this close (lane over 80% of the image, ~1.4 m) that suddenly looks
                               # half as wide for 3 frames has been passed: the next set took its place

    def __init__(self, target_depth: float, reversed: bool = False):
        self.target_depth = target_depth
        self.reversed = reversed
        self.reset()

    def reset(self, search_direction: int = 1):
        super().reset(search_direction)
        self.state = SlalomState.TURNING
        self.align_phase = "YAW"
        self.timer, self.lost_for, self.sets_passed = 0.0, 0.0, 0
        self.axis, self.side = None, None
        self.last_width, self.search_for = 0.0, 0.0
        self.peak_width, self.narrow_frames = 0.0, 0
        self._started = False

    def _start(self, sub, sensors):
        if self.reversed:
            self.axis = (getattr(sub, "course_heading", sensors.heading) + 180.0) % 360
            forward = getattr(sub, "slalom_pass_side", None) or getattr(sub, "gate_passage_side", None) or "left"
            self.side = "left" if forward == "right" else "right"
        else:
            self.axis = sensors.heading
            sub.course_heading = self.axis
            self.side = getattr(sub, "gate_passage_side", None) or "left"
            sub.slalom_pass_side = self.side

    @property
    def state_name(self) -> str:
        phase = f" {self.align_phase}" if self.state == SlalomState.ALIGNING else ""
        return f"Slalom{' back' if self.reversed else ''}: {self.state.name}{phase} ({self.side}, {self.sets_passed} passed)"

    @staticmethod
    def _poles(vision: Vision) -> List[Dict]:
        poles = [dict(b, color='red') for b in vision.red_blobs] + [dict(b, color='white') for b in vision.white_blobs]
        return [p for p in poles if p['height'] > p['width'] * 1.2]

    @staticmethod
    def _nearest_set(poles: List[Dict]) -> List[Dict]:
        """Poles whose bottoms line up form a set; the nearest set's bottoms are lowest in the image."""
        if not poles:
            return []
        poles = sorted(poles, key=lambda p: p['max_y'], reverse=True)
        groups, group = [], [poles[0]]
        for p in poles[1:]:
            if abs(p['max_y'] - group[-1]['max_y']) < p['height'] * 0.5:
                group.append(p)
            else:
                groups.append(group)
                group = [p]
        groups.append(group)
        sets = [g for g in groups if any(p['color'] == 'red' for p in g) and any(p['color'] == 'white' for p in g)]
        return max(sets, key=lambda g: np.mean([p['max_y'] for p in g])) if sets else []

    def _lane(self, vision: Vision) -> Optional[Tuple[Dict, Dict]]:
        """The red pole and the white pole on our side of it, in the nearest set."""
        s = self._nearest_set(self._poles(vision))
        red = next((p for p in s if p['color'] == 'red'), None)
        whites = [p for p in s if p['color'] == 'white']
        if red is None:
            return None
        if self.side == "left":
            whites = sorted((p for p in whites if p['center_x'] < red['center_x']), key=lambda p: -p['center_x'])
        else:
            whites = sorted((p for p in whites if p['center_x'] > red['center_x']), key=lambda p: p['center_x'])
        return (red, whites[0]) if whites else None

    def step(self, sub, dt, sensors, vision):
        w, _ = _image_size(sub)
        hold = lambda surge=0.0: _hold_heading(sub, sensors, self.axis, surge, self.target_depth)

        if self.state == SlalomState.TURNING:
            if (abs(self.target_depth - sensors.depth) < 0.2
                    and abs(angle_diff(self.axis, sensors.heading)) < 10.0):
                self.state = SlalomState.SEARCHING
            return TaskStatus.RUNNING, hold()

        if self.state == SlalomState.SEARCHING:
            if self._nearest_set(self._poles(vision)):
                self.state, self.align_phase, self.search_for = SlalomState.ALIGNING, "YAW", 0.0
            else:
                self.search_for += dt
                if self.search_for > self.SEARCH_GIVE_UP_S:
                    self.state = SlalomState.REALIGNING
            return TaskStatus.RUNNING, hold()

        if self.state == SlalomState.ALIGNING:
            if self.align_phase == "YAW":
                if abs(angle_diff(sensors.heading, self.axis)) < ALIGN_HEADING_TOL_DEG:
                    self.align_phase = "SWAY"
                return TaskStatus.RUNNING, hold()
            lane = self._lane(vision)
            if not lane:
                self.lost_for += dt
                if self.lost_for > 2.0:
                    self.state, self.lost_for = SlalomState.SEARCHING, 0.0
                return TaskStatus.RUNNING, hold()
            self.lost_for = 0.0
            err = (lane[0]['center_x'] + lane[1]['center_x']) / 2 - w / 2
            if abs(err) < w * ALIGN_POS_TOL:
                self.state = SlalomState.APPROACH
                return TaskStatus.RUNNING, hold()
            surge, side_damp = _damping(sub, sensors)
            sway = float(np.clip(-(err / (w / 2)) * ALIGN_SWAY_P + side_damp, -1.0, 1.0))
            yaw = float(np.clip(angle_diff(self.axis, sensors.heading) * sub.HOVER_YAW_P_GAIN
                                - sensors.imu.gyro_z * sub.YAW_D_GAIN, -1.0, 1.0))
            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(
                surge, sway, yaw, sub.get_heave_command(sensors, self.target_depth))

        if self.state == SlalomState.APPROACH:
            lane = self._lane(vision)
            self.lost_for = 0.0 if lane else self.lost_for + dt
            # Passing a set: its poles leave the view, and either nothing is seen (lost) or the next set,
            # 4 m further on, becomes "the lane" (its width drops). Sliding toward the next set's lane
            # before clearing this one's red pole hit it, so both mean: clear straight ahead first
            switched = False
            if lane:
                width = abs(lane[0]['center_x'] - lane[1]['center_x'])
                if self.peak_width > self.SWITCH_FROM_WIDTH * w and width < 0.5 * self.peak_width:
                    self.narrow_frames += 1          # Must hold for a few frames: one odd box isn't a new set
                else:
                    self.narrow_frames = 0
                    self.peak_width = max(self.peak_width, width)
                switched = self.narrow_frames >= 3
                self.last_width = width
            if switched or (self.lost_for > self.LOST_TO_CLEAR_SECONDS and self.last_width > self.CLOSE_WIDTH * w):
                self.state, self.timer, self.lost_for = SlalomState.CLEAR, 0.0, 0.0
            elif self.lost_for > self.LOST_FAR_GIVE_UP_S:
                self.state, self.lost_for = SlalomState.SEARCHING, 0.0
            if self.state == SlalomState.CLEAR:
                self.last_width = self.peak_width = 0.0
                self.narrow_frames = 0
            if lane and self.state == SlalomState.APPROACH:
                err = (lane[0]['center_x'] + lane[1]['center_x']) / 2 - w / 2
                _, side_damp = _damping(sub, sensors, sub.MANEUVER_DAMPING_GAIN)
                sway = float(np.clip(-(err / (w / 2)) * ALIGN_SWAY_P + side_damp, -1.0, 1.0))
                yaw = float(np.clip(angle_diff(self.axis, sensors.heading) * sub.HOVER_YAW_P_GAIN
                                    - sensors.imu.gyro_z * sub.YAW_D_GAIN, -1.0, 1.0))
                return TaskStatus.RUNNING, sub._mix_and_normalize_commands(
                    self.APPROACH_SURGE, sway, yaw, sub.get_heave_command(sensors, self.target_depth))
            return TaskStatus.RUNNING, hold(self.APPROACH_SURGE)

        if self.state == SlalomState.CLEAR:
            self.timer += dt
            if self.timer > self.CLEAR_SECONDS:
                self.sets_passed += 1
                self.state = SlalomState.SEARCHING
            return TaskStatus.RUNNING, hold(self.APPROACH_SURGE)

        # REALIGNING: no set in view, so the slalom is done once squared up on the course again
        if abs(angle_diff(self.axis, sensors.heading)) < 5.0:
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors)
        return TaskStatus.RUNNING, hold()


# --- End of the run ---
class TimedHoverTask(_CourseTask):
    """Holds depth and brakes to a stop for a while."""
    def __init__(self, duration: float, target_depth: float):
        self.duration, self.target_depth = duration, target_depth
        self.reset()

    def reset(self, search_direction: int = 1):
        super().reset(search_direction)
        self.timer, self._started = 0.0, False

    @property
    def state_name(self) -> str:
        return f"Hovering ({self.timer:.0f} / {self.duration:.0f} s)"

    def step(self, sub, dt, sensors, vision):
        self.timer += dt
        surge, sway = _damping(sub, sensors)
        yaw = -sensors.imu.gyro_z * sub.YAW_D_GAIN
        cmd = sub._mix_and_normalize_commands(surge, sway, yaw, sub.get_heave_command(sensors, self.target_depth))
        return (TaskStatus.COMPLETED if self.timer >= self.duration else TaskStatus.RUNNING), cmd


class SurfaceTask(_CourseTask):
    """Rises straight up to the surface."""
    def __init__(self):
        self.target_depth = 0.0
        self.reset()

    def reset(self, search_direction: int = 1):
        super().reset(search_direction)
        self._started = False

    @property
    def state_name(self) -> str:
        return "Surfacing"

    def step(self, sub, dt, sensors, vision):
        if sensors.depth < 0.1:
            return TaskStatus.COMPLETED, ThrusterCommands()
        return TaskStatus.RUNNING, sub._mix_and_normalize_commands(0.0, 0.0, 0.0, -1.0)
