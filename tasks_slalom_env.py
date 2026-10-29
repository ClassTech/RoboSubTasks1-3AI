#!/usr/bin/env python3
"""
Gymnasium environment for the tasks 1-3 slalom skill: the whole slalom, every
set from the first one ahead to the last in this direction, in the commanded
lane (left or right of the red poles, relative to the sub), then carry on 1.5 m
beyond the last. The mission runs it once each way (user: "Run the slalom as
one task"). It began as one set per skill step, run three times; then each
handoff between sets came from the previous step's done detector, moving, at
whatever moment it fired, which training never saw: skills that scored ~90% on
one set lost about a quarter of the missions between them.

Reward (the gate skill's structure, agreed with the user):
  progress   change in a score: minus the distance to the commanded lane's
             center (a point on the set's line, 0.76 m from its red pole), plus
             1.0 x depth error from 1.25 m, plus 1.0 x sideways distance from
             the lane's center line, plus 1.0 per radian off straight through,
             phased in over the last 2 m. After the pass: its value then, +20,
             plus distance beyond the set up to 1.5 m.
  success    +5, 1.5 m beyond the set after passing through the right lane
  failures   -5 each: collision, wrong lane, wrong course (crossing another set's
             line or the gate's, as backing up through the set just passed), missed (crossing the set's line
             outside its lanes, including over or under the floating poles, or
             back again after passing), backwards (crossing it facing more than
             90 deg away from the way through), surfaced, timeout
The backwards rule closed a loophole: on the turnaround starts a policy learned
to reverse blind through set 3 instead of turning to see it, picking a lane by
luck. (Backing through would count in competition, but with one forward camera
it's always blind.)
  time       -0.01 per step;  energy  -0.05 per second at full thrust

Actions: turn, depth, surge (3), no sway: with sway the gate skill went
through crabwise. Depth targets are limited to 1.0-1.45 m: the poles float
with their tops 0.6-0.9 m deep and bottoms 1.5-1.8 m, and a lane only counts
inside their span.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sub_env import SubTaskEnv
from tasks_observation import TasksObservationBuilder

CLEAR_M = 1.5
HOLD_S = 3.0          # The episode carries on this long after the finish (CLEAR_M beyond the set); crossing
                      # back meanwhile fails. Ending at the finish left the policy untrained just past it, and
                      # in the mission, where the done detector fires up to a second or two later, half the
                      # failures were the sub turning back across the set it had just passed
LANE_HALF_M = 1.524 / 2
TARGET_DEPTH_M = 1.25
LEAD_IN_POLICY = "runs/tslalom18_785_s23/best_model.zip"   # With its done detector beside it
LEAD_IN_MAX_S = 40.0
GATE_LEAD_IN_POLICY = "runs/tgate5_785_s25/best_model.zip"   # With its done detector beside it
GATE_DEPTH_LIMITS_M = (1.25, 1.65)   # The mission's limits on the gate skill's depth (tasks_mission.DEPTH_LIMITS)
TURNAROUND_LEAD_IN_POLICY = "runs/tturn8_785_s22/best_model.zip"   # With its done detector beside it


@dataclass
class TasksSlalomStarts:
    distance_m: tuple = (1.5, 3.5)        # Before the set, along the direction of travel. Less than the 4 m
                                          # between sets, so no other set is ever in the way
    lateral_m: tuple = (-1.5, 1.5)        # From the set's red pole
    heading_spread_deg: float = 30.0      # Facing the set within this...
    turnaround_share: float = 0.0         # ...except these: past set 3, facing on (away from it), to come back
                                          # through it, with the set remembered (_remember_set). Off: at 0.3-0.5
                                          # the skill turned around only 8-14% of the time and got worse at the
                                          # rest (88% -> 73%), so turning around is its own skill now
                                          # (tasks_turnaround_env.py)
    depth_m: tuple = (1.0, 1.4)
    gate_handoff_share: float = 0.3       # Of the rest: as the gate hands over, 5-7.5 m before set 1 going
    gate_handoff_distance_m: tuple = (5.0, 7.5)   # out (1.5 m past the gate, 8 m before it), near the line
    gate_handoff_lateral_m: tuple = (-0.6, 0.6)   # of the lane (set 1 lines up with the gate's passages).
    gate_lead_in_share: float = 0.0       # Training: of the gate-handoff starts, first fly the gate skill
                                          # (GATE_LEAD_IN_POLICY) from RoboSim's start through this lane's
                                          # passage, handing over where its detector fires, moving, as in the
                                          # mission. From the placed starts, the mission's slalom recrossed the
                                          # gate or backed into its top bar after early handoffs
                                          # Without them the skill had trained only from 1.5-3.5 m: 39/50
                                          # from these starts, against 49/50 between sets
    back_start_share: float = 0.3         # Of the rest: as the turnaround hands over, 2.5-3.5 m before set 3
                                          # coming back, facing it, near the lane's line
    back_start_distance_m: tuple = (2.5, 3.5)
    back_start_lateral_m: tuple = (-0.5, 0.5)
    back_lead_in_share: float = 0.0       # Training: of the back starts, first fly the whole slalom out and the
                                          # turnaround skill (TURNAROUND_LEAD_IN_POLICY), handing over where its
                                          # detector fires, moving, as in the mission. From the placed starts the
                                          # mission's slalom went back through set 3 and straight across it again
                                          # (6 of 16 slalom failures in 180 missions)
    blind_share: float = 0.0              # Of the rest (per-set skill only, unused now): the slalom skill flies in from 2-3 m until the set leaves
                                          # the view (within 2.5 m), and the episode starts there, moving, with the
                                          # memory and dead reckoning it had: the blind crossing, where the
                                          # mission's slalom hovered on the line and crossed back
    lead_in_share: float = 0.0            # (Per-set skill only, unused now.) Of the starts with a set before this one on the way: first fly the
                                          # slalom skill through that set, handing over when its detector
                                          # fires, as in the mission (moving, wherever it fired). In the first
                                          # learned missions ~4% of slalom steps failed, against 0-2% from
                                          # these starts at rest


@dataclass
class TasksSlalomReward:
    progress: float = 1.0
    depth_weight: float = 1.0
    center_line_weight: float = 2.0   # Was 1.0: policies kept shaving the red pole, the lane's inner edge
    facing_weight: float = 1.0
    facing_range_m: float = 2.0
    lane_bonus: float = 20.0
    clear_bonus: float = 5.0
    failure_penalty: float = 5.0
    time_penalty: float = 0.01
    energy_penalty: float = 0.05


class TasksSlalomEnv(SubTaskEnv):
    COURSE = "tasks"
    OUTCOMES = ("is_success", "collision", "wrong_lane", "missed", "backwards", "wrong_course", "surfaced", "timeout")
    ACTION_DIM = 3
    MIN_TARGET_DEPTH_M = 1.0
    MAX_TARGET_DEPTH_M = 1.45

    def __init__(self, detector_mode: str = "noisy", control_period: float = 0.1, max_seconds: float = 120.0,
                 starts: Optional[TasksSlalomStarts] = None, reward: Optional[TasksSlalomReward] = None,
                 randomize: bool = False, side: Optional[int] = None):
        self.starts = starts or TasksSlalomStarts()
        self.reward_cfg = rc = reward or TasksSlalomReward()
        self.fixed_side = side
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=rc.energy_penalty,
                         time_penalty=rc.time_penalty, progress_weight=rc.progress,
                         timeout_penalty=rc.failure_penalty, randomize=randomize)
        self.builder = TasksObservationBuilder(self.core.camera, "slalom")
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.builder.size,), dtype=np.float32)

    # --- Episode ---
    def _place_sub(self, rng: random.Random):
        s, st, info = self.core.state, self.starts, self.core.course.info
        turnaround = self.turnaround = rng.random() < st.turnaround_share
        handoff = self.handoff = not turnaround and rng.random() < st.gate_handoff_share
        back = self.back = not turnaround and not handoff and rng.random() < st.back_start_share
        self.set_index = 2 if turnaround or back else 0 if handoff else rng.randrange(3)
        self.direction = -1 if turnaround or back else 1 if handoff else rng.choice((1, -1))
        self.side = self.fixed_side or rng.choice((1, -1))
        self.lane = "left" if (self.side > 0) == (self.direction > 0) else "right"
        # The whole slalom: this set and every one after it in this direction
        self.remaining = list(range(self.set_index, 3) if self.direction > 0 else range(self.set_index, -1, -1))
        self._aim(self.set_index)
        if handoff:
            s.x = self.set_x - rng.uniform(*st.gate_handoff_distance_m)
            s.y = self.lane_y + rng.uniform(*st.gate_handoff_lateral_m)
        elif back:
            s.x = self.set_x + rng.uniform(*st.back_start_distance_m)
            s.y = self.lane_y + rng.uniform(*st.back_start_lateral_m)
        else:
            s.x = self.set_x - self.direction * rng.uniform(*st.distance_m)
            s.y = self.set_y + rng.uniform(*st.lateral_m)
        facing = 0.0 if self.direction > 0 else 180.0
        if turnaround:
            facing += 180.0   # Still facing the way the forward slalom went
        s.heading = (facing + rng.uniform(-st.heading_spread_deg, st.heading_spread_deg)) % 360
        s.z = rng.uniform(*st.depth_m)

    def _aim(self, set_index: int):
        """The set the episode is on now, in this episode's lane."""
        self.set_index = set_index
        self.set_x, self.set_y = self.core.course.info["slalom_centers"][set_index]
        self.lane_y = self.set_y + (LANE_HALF_M if self.lane == "left" else -LANE_HALF_M)
        self.line = f"slalom {set_index + 1}"

    def _on_reset(self):
        self.passed, self._at_pass, self._passes_seen, self._finished_at = False, None, 0, None
        self.sets_done = []   # Lines of the sets already through (not the last)
        self.builder.reset()
        if self.turnaround:
            self._remember_set()

    # --- Lead-in: fly the slalom skill through the set before, as in the mission ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        if type(self) is not TasksSlalomEnv or self.turnaround:
            return obs, info
        before = self.set_index - self.direction
        if self.handoff and self.np_random.random() < self.starts.gate_lead_in_share:
            if self._fly_gate():
                self.builder.reset()   # The slalom step starts with a fresh observation, as in the mission
                obs = self._start_after_lead_in()
        elif self.back and self.np_random.random() < self.starts.back_lead_in_share:
            if self._fly_slalom(2, 1, self.lane, whole=True) and self._fly_turnaround():
                self.builder.reset()   # The slalom step back starts with a fresh observation, as in the mission
                obs = self._start_after_lead_in()
        elif self.np_random.random() < self.starts.blind_share:
            if self._fly_slalom(self.set_index, self.direction, self.lane, until_blind=True):
                obs = self._start_after_lead_in(keep_observation=True)
        elif (0 <= before <= 2 and self.np_random.random() < self.starts.lead_in_share
                and self._fly_slalom(before, self.direction, self.lane)):
            self.builder.reset()   # Each slalom step starts with a fresh observation
            obs = self._start_after_lead_in()
        return obs, info

    def _lead_in_skill(self):
        if not hasattr(self, "_lead_in_parts"):
            from done_detector import DoneDetector, detector_path
            from gate_controllers import load_policy
            model, norm = load_policy(LEAD_IN_POLICY)
            self._lead_in_parts = (model, norm, DoneDetector(detector_path(LEAD_IN_POLICY)))
        return self._lead_in_parts

    def _fly_gate(self) -> bool:
        """The gate skill from RoboSim's start (9.144 m before the gate, +-0.914 m sideways, heading 0, 90 or
        270 deg, 0.5 m deep) through this episode's lane's passage, until its done detector fires. True if it
        went through just that passage."""
        from sub_env import DEPTH_MAX, DEPTH_MIN
        from tasks_observation import TasksObservationBuilder
        if not hasattr(self, "_gate_parts"):
            from done_detector import DoneDetector, detector_path
            from gate_controllers import load_policy
            model, norm = load_policy(GATE_LEAD_IN_POLICY)
            self._gate_parts = (model, norm, DoneDetector(detector_path(GATE_LEAD_IN_POLICY)),
                                TasksObservationBuilder(self.core.camera, "gate"))
        model, norm, detector, b = self._gate_parts
        rng, s, info = self.np_random, self.core.state, self.core.course.info
        s.x = info["gate_x"] - 9.144
        s.y = info["gate_center_y"] + rng.uniform(-0.914, 0.914)
        s.heading = float(rng.choice([0.0, 90.0, 270.0]))
        s.z = 0.5
        s.velocity_x = s.velocity_y = s.velocity_z = s.angular_velocity_z = 0.0
        self.readings, self.detections = self.core.read_sensors(0.0), self.core.detect(0.0)
        passes = len(self.core.passes)
        b.reset()
        detector.reset()
        side = 1 if self.lane == "left" else -1   # Going out, relative is the course's side
        lo, hi = ((d - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 2 - 1 for d in GATE_DEPTH_LIMITS_M)
        action = np.zeros(3, dtype=np.float32)
        min_depth, max_depth = self.MIN_TARGET_DEPTH_M, self.MAX_TARGET_DEPTH_M
        self.MIN_TARGET_DEPTH_M, self.MAX_TARGET_DEPTH_M = GATE_DEPTH_LIMITS_M
        try:
            for _ in range(int(2 * LEAD_IN_MAX_S / self.control_period)):
                r = self.readings
                obs = b.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                               self.detections, side, action, self.control_period)
                if detector.update(obs, r.heading):
                    new = [(p.line, p.direction, p.lane) for p in self.core.passes[passes:]]
                    return new == [("gate", 1, self.lane)]
                action = np.clip(model.predict(norm.normalize_obs(obs), deterministic=True)[0], -1.0, 1.0)
                action[1] = np.clip(action[1], lo, hi)
                if not self._drive(action):
                    return False
            return False
        finally:
            self.MIN_TARGET_DEPTH_M, self.MAX_TARGET_DEPTH_M = min_depth, max_depth

    def _fly_turnaround(self) -> bool:
        """After a whole-slalom lead-in: the turnaround skill, keeping the slalom's tracks (as the runner does),
        until its done detector fires. True if it crossed nothing on the way."""
        from collections import deque
        if not hasattr(self, "_turnaround_parts"):
            from done_detector import DoneDetector, detector_path
            from gate_controllers import load_policy
            model, norm = load_policy(TURNAROUND_LEAD_IN_POLICY)
            self._turnaround_parts = (model, norm, DoneDetector(detector_path(TURNAROUND_LEAD_IN_POLICY)))
        model, norm, detector = self._turnaround_parts
        b, passes = self.builder, len(self.core.passes)
        b.start_heading, b.frames = None, deque(maxlen=b.stack)
        detector.reset()
        action = np.zeros(3, dtype=np.float32)
        for _ in range(int(LEAD_IN_MAX_S / self.control_period)):
            r = self.readings
            obs = b.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                           self.detections, self.side, action, self.control_period)
            if detector.update(obs, r.heading):
                return len(self.core.passes) == passes
            action = np.clip(model.predict(norm.normalize_obs(obs), deterministic=True)[0], -1.0, 1.0)
            if not self._drive(action):
                return False
        return False

    def _drive(self, action) -> bool:
        """One control period toward the action's setpoints, without this skill's scoring. False on a collision."""
        from sub_env import DEPTH_MAX, DEPTH_MIN, HEADING_CHANGE_DEG
        a = np.clip(action, -1.0, 1.0)
        heading = (self.readings.heading + HEADING_CHANGE_DEG * float(a[0])) % 360
        depth = min(max(DEPTH_MIN + (float(a[1]) + 1) / 2 * (DEPTH_MAX - DEPTH_MIN), self.MIN_TARGET_DEPTH_M),
                    self.MAX_TARGET_DEPTH_M)
        dt, hits = self.core.config.physicsDt, self.core.collision_count
        for _ in range(max(1, round(self.control_period / dt))):
            r = self.readings
            cmd = self.controller.update(dt, r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                         heading, depth, float(a[2]), 0.0)
            self.core.advance(dt, cmd)
            self.readings, self.detections = self.core.read_sensors(dt), self.core.detect(dt)
        return self.core.collision_count == hits

    def _fly_slalom(self, set_index: int, direction: int, lane: str, until_blind: bool = False,
                    whole: bool = False) -> bool:
        """Flies the slalom skill (LEAD_IN_POLICY) through one set in a course lane, from 1.5-3 m before it at
        rest, until its done detector fires. True if it went through just that set, in that lane.
        whole: from where the gate hands over (5-7.5 m before set 1, going out) through every set to set 3.
        until_blind: instead, from 2-3 m, stop when the set first leaves the view within 2.5 m, before crossing."""
        model, norm, detector = self._lead_in_skill()
        if model.observation_space.shape[0] != self.builder.size:
            return False   # A policy for an older observation: no lead-ins until one for this one is trained
        rng, s = self.np_random, self.core.state
        x, y = self.core.course.info["slalom_centers"][set_index]
        if whole:
            set_index, direction = 2, 1
            x, y = self.core.course.info["slalom_centers"][0]
            s.x = x - rng.uniform(5.0, 7.5)
        else:
            s.x = x - direction * rng.uniform(2.0 if until_blind else 1.5, 3.0)
        s.y = y + (LANE_HALF_M if lane == "left" else -LANE_HALF_M) + rng.uniform(-0.5, 0.5)
        s.heading = ((0.0 if direction > 0 else 180.0) + rng.uniform(-10.0, 10.0)) % 360
        s.z = rng.uniform(1.0, 1.4)
        s.velocity_x = s.velocity_y = s.velocity_z = s.angular_velocity_z = 0.0
        self.readings, self.detections = self.core.read_sensors(0.0), self.core.detect(0.0)
        passes = len(self.core.passes)
        self.builder.reset()
        detector.reset()
        side = 1 if (lane == "left") == (direction > 0) else -1   # Relative to the sub
        action = np.zeros(3, dtype=np.float32)
        for _ in range(int((3 if whole else 1) * LEAD_IN_MAX_S / self.control_period)):
            r = self.readings
            obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                      self.detections, side, action, self.control_period)
            if until_blind:
                b = self.builder
                if len(self.core.passes) > passes:
                    return False   # Crossed something before going blind
                if b.red.seen_ever and not b.red.found and b.red.range_m < 2.5:
                    self.last_action = np.append(action, np.zeros(self.ACTION_DIM - 3)).astype(np.float32)
                    return True
            elif detector.update(obs, r.heading):
                new = [(p.line, p.direction, p.lane) for p in self.core.passes[passes:]]
                if whole:
                    return new == [(f"slalom {i}", 1, lane) for i in (1, 2, 3)]
                return new == [(f"slalom {set_index + 1}", direction, lane)]
            action = np.clip(model.predict(norm.normalize_obs(obs), deterministic=True)[0], -1.0, 1.0)
            if not self._drive(action):
                return False
        return False

    def _start_after_lead_in(self, keep_observation: bool = False) -> np.ndarray:
        """The episode takes over where the lead-in handed over: clocks, passes and the reward's baseline.
        keep_observation: carry on mid-skill (blind starts), frames and start heading included."""
        from collections import deque
        b = self.builder
        if not keep_observation:
            b.start_heading, b.frames = None, deque(maxlen=b.stack)
        self.target_heading, self.target_depth = self.readings.heading, self.readings.depth
        self.time, self.energy = 0.0, 0.0
        self._passes_seen, self._collisions_seen = len(self.core.passes), self.core.collision_count
        self._submerged = True
        self.potential = self._potential()
        if not keep_observation:
            self.last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        return self._observe()

    def _remember_set(self, seen_ago_s: float = 2.0):
        """
        Turnaround starts: the sub has just come through this set going forward, so it saw it seconds ago.
        Fill the observation's memory as the forward pass would have left it (labels as seen facing forward:
        'left' is the course's left). In the mission the slalom keeps its memory from forward to back.
        Without this, the sub started blind with the set behind it, and learned to reverse through blind.
        """
        s, r, b = self.core.state, self.readings, self.builder
        offset = r.heading - s.heading            # Measured minus true heading (the magnetometer's error)

        def remember(track, x, y):
            bearing = (math.degrees(math.atan2(y - s.y, x - s.x)) - s.heading + 180.0) % 360.0 - 180.0
            track.seen_ever, track.found = True, False
            track.abs_bearing = s.heading + offset + bearing
            track.range_m, track.confidence, track.unseen = math.hypot(x - s.x, y - s.y), 0.8, seen_ago_s
            return bearing

        red = remember(b.red, self.set_x, self.set_y)
        b.red_travel = max(abs(s.x - self.set_x) + 1.0, 0.0)   # Driven since it left the view, about 1 m before it
        for side, sign in (("left", 1), ("right", -1)):
            white = remember(b.whites[side], self.set_x, self.set_y + sign * 2 * LANE_HALF_M)
            remember(b.lanes[side], self.set_x, self.set_y + sign * LANE_HALF_M)
            b.lanes[side].width_deg = abs(red - white)

    def _past_set_m(self) -> float:
        return (self.core.state.x - self.set_x) * self.direction

    def _physics_outcome(self) -> Optional[str]:
        new = self.core.passes[self._passes_seen:]
        self._passes_seen = len(self.core.passes)
        for p in new:
            if p.line != self.line:
                # Another set (one already passed, or one ahead out of order) or the gate: as in the mission, a
                # failure. It used to be ignored, and once starts put the set just passed 1.5 m behind, the
                # policy's habit of backing up first took it back through that set
                return "wrong_course"
            through = 0.0 if p.direction > 0 else 180.0
            if abs((self.core.state.heading - through + 180.0) % 360.0 - 180.0) > 90.0:
                return "backwards"
            if self.passed or p.direction != self.direction or p.lane is None:
                return "missed"
            if p.lane != self.lane:
                return "wrong_lane"
            if self.set_index != self.remaining[-1]:
                self.sets_done.append(self.line)
                self._aim(self.remaining[len(self.sets_done)])   # On to the next set
            else:
                self.passed = True   # The last set
        return None

    def done_label(self) -> bool:
        """The finish the done detector learns: through, and CLEAR_M beyond."""
        return self.passed and self._past_set_m() >= CLEAR_M

    def _complete(self) -> bool:
        if type(self) is not TasksSlalomEnv:
            return self.done_label()
        if self.done_label() and self._finished_at is None:
            self._finished_at = self.time
        return self._finished_at is not None and self.time >= self._finished_at + HOLD_S

    def _terminal_reward(self, outcome: str) -> float:
        rc = self.reward_cfg
        return rc.clear_bonus if outcome == "success" else -rc.failure_penalty

    def _approach_potential(self) -> float:
        s, rc = self.core.state, self.reward_cfg
        dist = math.hypot(self.set_x - s.x, self.lane_y - s.y)
        through = 0.0 if self.direction > 0 else 180.0
        off = abs(math.radians((s.heading - through + 180.0) % 360.0 - 180.0))
        phase = min(max((rc.facing_range_m - dist) / rc.facing_range_m, 0.0), 1.0)
        return -(dist + rc.depth_weight * abs(s.z - TARGET_DEPTH_M)
                 + rc.center_line_weight * abs(s.y - self.lane_y) + rc.facing_weight * phase * off)

    def _potential(self) -> float:
        done = self.reward_cfg.lane_bonus * len(getattr(self, "sets_done", []))   # Each set through so far
        if self.passed:
            if self._at_pass is None:
                self._at_pass = self._approach_potential()
            return done + self._at_pass + self.reward_cfg.lane_bonus + min(max(self._past_set_m(), 0.0), CLEAR_M)
        return done + self._approach_potential()

    def _observe(self) -> np.ndarray:
        r = self.readings
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                  self.detections, self.side, self.last_action, self.control_period)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def scripted_mission(self) -> List:
        from ai.tasks.course_tasks import SlalomTask
        # The course heading in the measured heading's frame (the magnetometer is off by up to 15 deg per run)
        offset = self.readings.heading - self.core.state.heading
        axis = ((0.0 if self.direction > 0 else 180.0) + offset) % 360
        return [SlalomTask(target_depth=TARGET_DEPTH_M, side="left" if self.side > 0 else "right", axis=axis)]
