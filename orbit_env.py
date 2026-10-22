#!/usr/bin/env python3
"""
Gymnasium environment for the orbit skill: start just past the gate, go around
the marker pole (either side), and come back facing the gate. Actions,
observation and the episode loop are in sub_env.py; this skill adds sway as a
fourth action and a marker track to the observation.

The only camera faces forward (±35°), so the sub can't see the marker while
circling it nose-first. With sway it can face the marker and slide around it,
keeping it in view, which is how the scripted orbit does it.

Success needs both:
  - truth: the sub has swept at least SWEEP_DEG around the marker, either way, so
    it really went around (not out and back on one side)
  - handoff, measurable on the real sub: the gate is in view and the sub's
    heading has turned 180° (± HANDOFF_HEADING_DEG) since the start of the orbit
The done detector's label (done_label) is ground truth: swept SWEEP_DEG and
facing the gate within DONE_FACING_DEG.
Failures: a collision, crossing back through the gate plane early, or a timeout.

The safety margin (MIN_MARKER_DISTANCE_M from the marker's center) has three rules:
  soft  (default; the real-world rule, used for all scoring) only a collision
        fails; inside the margin costs reward every step, more the closer it gets
  hard  (for training) entering the margin ends the episode as "too_close", -5.
        Policies learn to stay clear much more reliably this way; from scratch
        under the soft rule they get stuck near the marker.
  near  (for training) in between: the margin costs reward as in soft, and only
        coming within NEAR_STOP_M ends the episode. With the real (draggy) physics,
        exploration dipped inside 1.5 m in almost every hard-rule episode, so
        policies never practiced the rest of the loop.

Practice starts (training only): with practice_starts, a share of episodes
start partway around the marker, facing it, with the sweep count and the
handoff's reference heading set to match, so the policy gets plenty of
practice at finishing the loop and turning to the gate. Scoring uses normal starts.

While going around, each step costs more the farther the marker is from the
center of the camera view (out of view counts as the edge), so the policy keeps
it centered, with room for noise, as the scripted orbit does.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sub_env import SubTaskEnv

SWEEP_DEG = 270.0
MARKER_GRACE_S = 0.5           # A dropout this short doesn't count as losing sight of the marker
HALF_FOV_DEG = 35.0
MIN_MARKER_DISTANCE_M = 1.5    # Safety margin, center to center; contact is at about 0.38 m
NEAR_STOP_M = 1.0              # The "near" training rule ends episodes this close
HANDOFF_HEADING_DEG = 45.0
DONE_FACING_DEG = 30.0         # Done label: facing the gate's center within this


def _wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


@dataclass
class OrbitStartConfig:
    """Where episodes start, relative to the gate. Each value is drawn uniformly from (min, max)."""
    past_gate_m: tuple = (1.0, 5.0)       # Beyond the gate, toward the marker
    lateral_m: tuple = (-1.0, 1.0)        # Sideways from the gate's center line
    heading_deg: tuple = (-30.0, 30.0)    # Relative to facing the marker's direction
    depth_m: tuple = (1.2, 1.7)
    # Practice starts (training only)
    practice_share: float = 0.33          # Share of episodes that start partway around
    practice_swept_deg: tuple = (150.0, 300.0)
    practice_distance_m: tuple = (1.8, 2.6)
    practice_heading_deg: tuple = (-20.0, 20.0)   # Relative to facing the marker


@dataclass
class OrbitRewardConfig:
    sweep: float = 2.0             # Per radian swept around the marker, up to SWEEP_DEG, counted from when the
                                   # sub joins the loop. Counted from the start, the policy went around while
                                   # still closing in and reached the ring ~115 deg around (3-4 m out at 90
                                   # deg). Counted only while near, it hugged the marker
    face_gate: float = 4.0         # For facing the gate, phased in over the second half of the sweep
    depth_weight: float = 1.0      # Per meter from target_depth_m
    target_depth_m: float = 1.5
    far_weight: float = 1.0        # Per meter farther than far_m from the marker (in the progress measure)
    far_m: float = 2.0             # Was 2.5: with nothing pulling it in below 2.5 m, the sub drifted out to
                                   # 3-4 m a quarter of the way around (sliding sideways while backing off)
    ring_cost: float = 0.15        # Per step, per meter off the ring at far_m from the marker, in or out, once
                                   # the sub has joined the loop (come within far_m + join_margin_m).
                                   # - Out: the progress term alone gives back on the way in what it charges
                                   #   on the way out, so it didn't stop swing-outs.
                                   # - In: the sweep reward pays per degree, and a smaller circle sweeps
                                   #   degrees faster at the same speed, so without this the policy hugged
                                   #   the margin (and with the real, slower physics, dipped inside it).
                                   # - Not on the approach: there it made ending the episode at the margin
                                   #   (-5) cheaper than flying in.
    join_margin_m: float = 0.25
    approach_sway_cost: float = 0.03  # Per step at full sway, until the sub joins the loop. Without it the
                                      # policy kept sliding sideways all the way in (a habit from when going
                                      # around early was paid), curving the approach and slowing it
    success_bonus: float = 20.0
    collision_penalty: float = 5.0
    margin_cost: float = 0.1       # Per step inside the safety margin...
    margin_cost_depth: float = 0.5 # ...plus this much more per step, scaled by how far inside it is
    wrong_way_penalty: float = 5.0
    timeout_penalty: float = 5.0
    off_center_cost: float = 0.04  # Per step while going around, scaled by how far off center the marker is
                                   # (full cost at the edge of the view or when out of view > MARKER_GRACE_S)
    time_penalty: float = 0.01     # Per step
    energy_penalty: float = 0.05   # Per second at full thrust on every thruster


class OrbitEnv(SubTaskEnv):
    OUTCOMES = ("is_success", "collision", "too_close", "wrong_way", "surfaced", "timeout")   # too_close: hard rule only
    # Fraction of the loop with the marker in view; how far off center it was while in view;
    # seconds inside the safety margin; closest pass
    EXTRA_INFO = ("marker_view", "off_center_deg", "inside_margin_s", "closest_m")
    INCLUDE_MARKER = True
    ACTION_DIM = 4
    MIN_TARGET_DEPTH_M = 1.2   # Same safety limit the mission runner applies: the orbit never needs to be shallower
    MAX_TARGET_DEPTH_M = 1.65  # At least 0.44 m above the pole feet (floor 2.09 m), so perception can range them.
                               # Allowed down to 1.9 m, a policy got stuck there (its depth output clipped at
                               # the limit gives no learning signal) and misjudged the gate by 2-3 m

    def __init__(self, detector_mode: str = "perfect", control_period: float = 0.1,
                 max_seconds: float = 120.0, start: Optional[OrbitStartConfig] = None,
                 reward: Optional[OrbitRewardConfig] = None, margin_rule: str = "soft",
                 practice_starts: bool = False, randomize: bool = False,
                 mission_starts: Optional[list] = None, mission_share: float = 0.4):
        assert margin_rule in ("soft", "hard", "near"), margin_rule
        self.margin_rule = margin_rule
        self.practice_starts = practice_starts
        self.mission_starts, self.mission_share = mission_starts or [], mission_share   # Recorded handoffs (training)
        self.start_cfg = start or OrbitStartConfig()
        self.reward_cfg = rc = reward or OrbitRewardConfig()
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=rc.energy_penalty,
                         time_penalty=rc.time_penalty, progress_weight=1.0,
                         timeout_penalty=rc.timeout_penalty, randomize=randomize)

    # --- Episode ---
    def _place_sub(self, rng: random.Random):
        s, g, m, sc = self.core.state, self.core.gate, self.core.marker, self.start_cfg
        self._preset_swept = 0.0
        if self.mission_starts and rng.random() < self.mission_share:
            # Where the mission runner actually hands over, still moving after the straight drive
            self.apply_recorded_start(rng.choice(self.mission_starts))
            return
        if self.practice_starts and rng.random() < sc.practice_share:
            # Partway around the marker, facing it. The orbit begins on the gate side (angle 180 deg).
            swept = rng.choice((-1, 1)) * rng.uniform(*sc.practice_swept_deg)
            angle = math.radians(180.0 + swept)
            d = rng.uniform(*sc.practice_distance_m)
            s.x, s.y = m.x + d * math.cos(angle), m.y + d * math.sin(angle)
            s.heading = (math.degrees(math.atan2(m.y - s.y, m.x - s.x)) + rng.uniform(*sc.practice_heading_deg)) % 360
            s.z = rng.uniform(*sc.depth_m)
            self._preset_swept = swept
            return
        s.x = g.x + rng.uniform(*sc.past_gate_m)
        s.y = g.center_y + rng.uniform(*sc.lateral_m)
        s.heading = rng.uniform(*sc.heading_deg) % 360
        s.z = rng.uniform(*sc.depth_m)

    def _marker_angle(self) -> float:
        """The sub's angular position around the marker (deg), ground truth."""
        s, m = self.core.state, self.core.marker
        return math.degrees(math.atan2(s.y - m.y, s.x - m.x))

    def _on_reset(self):
        self.swept = self._preset_swept          # Signed degrees swept around the marker
        self._last_angle = self._marker_angle()
        if self._preset_swept:
            # Reference heading for the handoff: as if the orbit had begun facing the marker
            # from the gate side (true heading 0), expressed in the magnetometer's frame
            self.builder.start_heading = (self.readings.heading - self.core.state.heading) % 360
        self._loop_steps = self._loop_steps_seen = 0
        self._off_center_sum = 0.0
        self._inside_margin_s, self._closest = 0.0, self._marker_distance()
        # Has the sub come in to the loop yet? The ring cost and the sweep reward start then.
        # Practice starts begin on the loop, partway around
        self._joined = bool(self._preset_swept)
        self.swept_joined = self._preset_swept   # Signed degrees swept since joining (the sweep reward)

    def _marker_distance(self) -> float:
        s, m = self.core.state, self.core.marker
        return math.hypot(s.x - m.x, s.y - m.y)

    def _going_around(self) -> bool:
        return abs(self.swept) < SWEEP_DEG

    def _step_cost(self) -> float:
        """Keeping the marker in view while going around is how the sub stays safe with one forward camera."""
        rc, cost = self.reward_cfg, 0.0
        distance = self._marker_distance()
        self._joined = self._joined or distance <= rc.far_m + rc.join_margin_m
        if self._joined:
            cost += rc.ring_cost * abs(distance - rc.far_m)
        else:
            cost += rc.approach_sway_cost * abs(float(self.last_action[3]))
        inside = MIN_MARKER_DISTANCE_M - distance
        if inside > 0:
            cost += rc.margin_cost + rc.margin_cost_depth * inside / MIN_MARKER_DISTANCE_M
        if self._going_around():
            self._loop_steps += 1
            marker = self.builder.marker
            off_center = 1.0
            if marker.seen_ever and marker.unseen <= MARKER_GRACE_S:
                off_center = min(abs(marker.bearing(self.readings.heading)) / HALF_FOV_DEG, 1.0)
            if marker.found:
                self._loop_steps_seen += 1
                self._off_center_sum += abs(marker.bearing(self.readings.heading))
            cost += rc.off_center_cost * off_center
        return cost

    def _episode_info(self) -> dict:
        return {"marker_view": self._loop_steps_seen / max(1, self._loop_steps),
                "off_center_deg": self._off_center_sum / max(1, self._loop_steps_seen),
                "inside_margin_s": self._inside_margin_s, "closest_m": self._closest}

    def _physics_outcome(self) -> Optional[str]:
        angle = self._marker_angle()
        step = _wrap(angle - self._last_angle)
        self.swept += step
        rc = self.reward_cfg
        self._joined = self._joined or self._marker_distance() <= rc.far_m + rc.join_margin_m
        if self._joined:
            self.swept_joined += step
        self._last_angle = angle
        d = self._marker_distance()
        self._closest = min(self._closest, d)
        if d < MIN_MARKER_DISTANCE_M:
            self._inside_margin_s += self.core.config.physicsDt
            if self.margin_rule == "hard" or (self.margin_rule == "near" and d < NEAR_STOP_M):
                return "too_close"
        if self.new_crossings():
            return "wrong_way"
        return None

    def handoff_ready(self) -> bool:
        """Measurable on the real sub: gate in view, heading turned about 180° since the orbit began."""
        b = self.builder
        if b.start_heading is None or not b.gate.found:
            return False
        return abs(_wrap(self.readings.heading - b.start_heading - 180.0)) <= HANDOFF_HEADING_DEG

    def _facing_gate_deg(self) -> float:
        s, g = self.core.state, self.core.gate
        return abs(_wrap(math.degrees(math.atan2(g.center_y - s.y, g.x - s.x)) - s.heading))

    def _complete(self) -> bool:
        return abs(self.swept) >= SWEEP_DEG and self.handoff_ready()

    def done_label(self) -> bool:
        return abs(self.swept) >= SWEEP_DEG and self._facing_gate_deg() <= DONE_FACING_DEG

    def _terminal_reward(self, outcome: str) -> float:
        rc = self.reward_cfg
        return {"success": rc.success_bonus, "collision": -rc.collision_penalty,
                "too_close": -rc.collision_penalty, "wrong_way": -rc.wrong_way_penalty}[outcome]

    def _potential(self) -> float:
        """Sweep around the marker, then face the gate, at mission depth, without drifting far. Ground truth."""
        s, m, g, rc = self.core.state, self.core.marker, self.core.gate, self.reward_cfg
        sweep = min(abs(self.swept_joined), SWEEP_DEG)
        to_gate = math.degrees(math.atan2(g.center_y - s.y, g.x - s.x))
        facing = 1.0 - abs(_wrap(to_gate - s.heading)) / 180.0          # 1 facing the gate, 0 facing away
        phase_in = float(np.clip((abs(self.swept) - 180.0) / (SWEEP_DEG - 180.0), 0.0, 1.0))
        distance = math.hypot(s.x - m.x, s.y - m.y)
        return (rc.sweep * math.radians(sweep) + rc.face_gate * phase_in * facing
                - rc.depth_weight * abs(s.z - rc.target_depth_m)
                - rc.far_weight * max(0.0, distance - rc.far_m))

    def scripted_mission(self) -> List:
        # The scripted mission's orbit, the pause after it, and the return toward the gate
        from main import build_mission
        return build_mission()[2:5]
