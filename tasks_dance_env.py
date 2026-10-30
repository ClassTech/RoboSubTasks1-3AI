#!/usr/bin/env python3
"""
Gymnasium environment for the victory dance: points are scored for each 90 deg
turn in any axis and direction. The user asked for it learned, not programmed.
The runner says "dance" right after the gate.

Yaw, then roll (RoboSim's order):
  yaw    a full turn clockwise with a stop at each 90 deg, then a full turn back
         counterclockwise: eight quarter turns. A stop counts once the sub is within
         STOP_DEG of the stop's heading and nearly still (under STILL_DPS) for HOLD_S,
         measured from the heading the dance started on (so small errors don't add up
         stop to stop, and the magnetometer's offset cancels).
  roll   a full roll right (right side down), then a full roll back left: eight more
         quarters, each counted on reaching it (within ROLL_PASS_DEG) with no hold: rolled
         past ~60 deg the side thrusters can't hold the sub there, and on its side or
         upside down they push it sideways and the buoyant sub rises. The last quarter,
         level again, is held still for LEVEL_HOLD_S.
Judged on the true heading and roll, as the judges see them. Roll's numbers are
guesses (config.py: nothing measured), so training varies the righting moment widely.

Actions (3), at 10 Hz:
  turn   slews the turn target: past +-TURN_DEAD it moves at up to TURN_DPS (+ clockwise);
         inside, the target snaps to a quarter: on to the next one if it has moved more than
         TURN_COMMIT_DEG that way since the last, else back. Jitter averages out and a short
         push snaps back, so holding a stop is easy; turning on is a push past TURN_COMMIT_DEG.
         One push moves it at most a quarter: going on needs a release first. (Without
         that, an imitating policy's pushes often ran through a stop: tdance_bc6, lost.)
         (Snapping to the nearest, an imitating policy pushing at 0.5-0.9 kept dropping into
         the dead band and snapping back to the stop it had left.)
         The target is measured from the dance's starting heading, -360..0 deg, and the
         controller steps toward it at most 45 deg per action. Tried before: turning
         relative to the current heading each step (holding a stop needed a near-zero action
         30 steps running); an absolute target over 440 deg (exploring moved it +-80 deg a
         step); a quarter fired on crossing +-0.5 (exploring fired it early in a stop's
         hold and lost the stop, so the policy learned never to fire), and on 3 steps
         running past it (the policy's output jittered and never held 3). None got past
         the first stop.
  depth  target depth, 1.0-1.8 m
  roll   inside +-ROLL_DEAD: hold level (a PD on the IMU's roll); past it, open-loop roll
         effort, the side vertical thrusters pushed apart (+ = right side down). (All open
         loop, a hand controller with 0.1 of noise on it never held the last level stop:
         the sub rocked past its 10 deg/s.)

Observation (DanceObservation; measured only, numpy, runs on the Pi), per frame:
  depth, depth rate, pitch, yaw rate
  turned       heading turned since the dance began (unwrapped magnetometer), /360
  deepest      the farthest clockwise turned so far, /360: -180 on the way out and
               on the way back look the same without it
  still        seconds within OWN_STOP_DEG of the turn target and under OWN_STILL_DPS, /HOLD_S: the
               stop's hold as the judges count it, measured. (Counting the yaw rate alone, a
               policy couldn't tell when a stop had counted, and turning on early loses it, so
               it learned never to turn on: tdance4.)
  target       the turn target from the start, /360
  stop_done    1 once the turn target has been held for STOP_DONE_S (after the judges' HOLD_S, so
               it lags their count rather than leading it), until it settles on another
               quarter (a push that snaps back keeps it): without it,
               "this stop counted" and "this stop still to hold" looked the same once the sub
               wobbled, and an imitating policy pushed on half-heartedly (tdance_bc2)
  roll         sin and cos of the IMU's roll, and the roll rate
  rolled       roll since the dance began (unwrapped), /360, and the farthest so far
  level        seconds level and still in roll, /LEVEL_HOLD_S
  (No last action, unlike the other skills: imitating the hand controller, whose actions are
  crisp -1/0/1, a policy that saw its own fuzzy ones flew into states it never trained on.)
4 frames stacked.

Reward:
  progress   change in a score: 3 per stop or quarter done, minus 1 per 90 deg from the
             current one, plus 1 x the fraction of its hold done; minus 1 per 90 deg of
             roll during the yaw part and of heading off the start during the roll part;
             minus 1.0 x depth error from 1.3 m, minus 0.5 per metre drifted from the start
             (clear of the gate behind and set 1 ahead)
  success    +5 after the last quarter
  failures   -5 each: collision, surfaced, timeout, lost (more than 180 deg off)
  time       -0.01 per step;  energy  -0.05 per second at full thrust
Starts: as the gate hands over, 1-2.5 m past it near a passage's line, facing on
within 15 deg, 1.2-1.6 m deep, up to 0.3 m/s forward. In training, stage_share of
episodes start partway through, at stop 1-15 with the previous one just done (turned or
rolled there, and the counters to match), so every change of direction gets practised.
With segment set, training episodes are pieces of the dance instead: each starts at a
random stop (0-15) and succeeds after segment more, in SEGMENT_SECONDS each (the whole
dance at once never learned more than one stop in 4M steps: tdance3-7). Before stage starts,
started from the beginning only, a policy held the first stop and never tried turning on
(tdance3), and one trained half from the roll part rolled right and never back.
"""
import math
import random
from collections import deque
from typing import List, Optional

import numpy as np

from controller import angle_diff
from sub_env import HEADING_CHANGE_DEG, SubTaskEnv

YAW_STOPS = [-90.0, -180.0, -270.0, -360.0, -270.0, -180.0, -90.0, 0.0]   # Clockwise is negative (heading grows left)
ROLL_STOPS = [90.0, 180.0, 270.0, 360.0, 270.0, 180.0, 90.0, 0.0]          # Right side down is positive
STAGES = [("yaw", a) for a in YAW_STOPS] + [("roll", a) for a in ROLL_STOPS]
STOP_DEG = 5.0
STILL_DPS = 5.0
HOLD_S = 3.0
STOP_DONE_S = HOLD_S + 0.5
OWN_STOP_DEG, OWN_STILL_DPS = 3.0, 3.0   # The policy's own count, stricter than the judges': they see rate
                                         # spikes between its 10 Hz gyro readings (a quarter of its holds
                                         # at 5 deg and 5 deg/s were ones they hadn't counted)
ROLL_PASS_DEG = 15.0
LEVEL_HOLD_S = 2.0
TURN_LIMITS_DEG = (-360.0, 0.0)   # The turn target from the start, clockwise negative
TURN_DEAD = 0.3
TURN_DPS = 60.0
TURN_COMMIT_DEG = 15.0
SEGMENT_SECONDS = 15.0
ROLL_DEAD = 0.4
LEVEL_P, LEVEL_D = 1 / 40.0, 1 / 200.0   # Roll effort per deg, per deg/s
TARGET_DEPTH_M = 1.3
DEPTH_RANGE_M = (1.0, 1.8)   # Deep before rolling: the rolls use all the vertical thrust, and the sub rises ~0.1 m/s
HEADING_GAIN_SCALE = 0.5     # The heading gains halved: at the real thrust the defaults wobble up to 14 deg/s
                             # holding a heading, past the stops' STILL_DPS (controller.ControllerGains)
FIELDS = ["depth", "depth_rate", "pitch", "yaw_rate", "turned", "deepest", "still", "target", "stop_done",
          "roll_sin", "roll_cos", "roll_rate", "rolled", "rolled_most", "level"]


class DanceObservation:
    """The dance's observation, from the depth sensor, IMU and magnetometer."""
    def __init__(self, stack: int = 4):
        self.stack, self.fields = stack, FIELDS
        self.index = {f: i for i, f in enumerate(FIELDS)}
        self.frame_size = len(FIELDS)
        self.size = self.frame_size * stack
        self.reset()

    def reset(self, turned: float = 0.0, deepest: float = 0.0, rolled: float = 0.0, rolled_most: float = 0.0,
              stop_done: bool = False):
        """Starting partway through the dance (training): the counters as the dance so far left them."""
        self.turned, self.deepest, self.still = turned, deepest, 0.0
        self.stop_done, self._last_target = stop_done, turned
        self.rolled, self.rolled_most, self.level = rolled, rolled_most, 0.0
        self._last_heading = self._last_roll = None
        self.frames: deque = deque(maxlen=self.stack)

    def update(self, depth, depth_rate, heading, pitch, yaw_rate, roll, roll_rate, target, dt) -> np.ndarray:
        if self._last_heading is not None:
            self.turned += angle_diff(heading, self._last_heading)
            self.rolled += angle_diff(roll, self._last_roll)
        self._last_heading, self._last_roll = heading, roll
        self.deepest = min(self.deepest, self.turned)
        self.rolled_most = max(self.rolled_most, self.rolled)
        on_stop = abs(self.turned - target) < OWN_STOP_DEG and abs(math.degrees(yaw_rate)) < OWN_STILL_DPS
        self.still = self.still + dt if on_stop else 0.0
        if target % 90.0 == 0.0 and target != self._last_target:   # Settled on another quarter (not mid-push)
            self.stop_done, self._last_target = False, target
        self.stop_done = self.stop_done or self.still >= STOP_DONE_S
        level = abs(roll) < STOP_DEG and abs(math.degrees(roll_rate)) < 2 * STILL_DPS
        self.level = self.level + dt if level else 0.0
        r = math.radians(roll)
        frame = np.asarray([depth / 2.0, depth_rate / 0.5, pitch / 20.0, yaw_rate / 1.0,
                            self.turned / 360.0, self.deepest / 360.0, min(self.still, 2 * HOLD_S) / HOLD_S, target / 360.0, float(self.stop_done),
                            math.sin(r), math.cos(r), roll_rate / 3.0, self.rolled / 360.0,
                            self.rolled_most / 360.0, min(self.level, 2 * LEVEL_HOLD_S) / LEVEL_HOLD_S],
                           dtype=np.float32)
        if not self.frames:
            for _ in range(self.stack - 1):
                self.frames.append(frame)
        self.frames.append(frame)
        return np.concatenate(list(reversed(self.frames)))


class DanceDriver:
    """The dance's actions as setpoints and thruster commands, for its env and for the mission."""
    def __init__(self, controller, control_period: float):
        self.controller, self.control_period = controller, control_period
        self.reset()

    def reset(self, turned: float = 0.0):
        self.turn_target, self.turn_dir = turned, 0.0   # turn_dir: the way it last slewed, + counterclockwise
        self.turn_from = turned   # The quarter this push started from

    def act(self, action, heading: float, turned: float):
        """Returns (target heading, target depth, driver) for one control period; turned is the
        observation's own count of the heading turned since the dance began."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._slew_turn(float(a[0]))
        turn = float(np.clip(self.turn_target - turned, -HEADING_CHANGE_DEG, HEADING_CHANGE_DEG))
        target_heading = (heading + turn) % 360
        target_depth = DEPTH_RANGE_M[0] + (float(a[1]) + 1) / 2 * (DEPTH_RANGE_M[1] - DEPTH_RANGE_M[0])
        push = float(a[2])

        def driver(r, detections, dt: float):
            if abs(push) > ROLL_DEAD:
                roll = math.copysign((abs(push) - ROLL_DEAD) / (1 - ROLL_DEAD), push)
            else:
                roll = -LEVEL_P * r.roll - LEVEL_D * math.degrees(r.imu.gyro_x)
            return self.controller.update(dt, r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y,
                                          r.imu.gyro_z, target_heading, target_depth, 0.0, 0.0, roll, r.roll)
        return target_heading, target_depth, driver

    def _slew_turn(self, a: float):
        """The turn action: slews the target past TURN_DEAD, up to a quarter from where the push began;
        inside, snaps it to a quarter (on to the next past TURN_COMMIT_DEG, else back)."""
        if abs(a) > TURN_DEAD:
            rate = (abs(a) - TURN_DEAD) / (1 - TURN_DEAD) * TURN_DPS
            self.turn_dir = -math.copysign(1.0, a)
            self.turn_target += self.turn_dir * rate * self.control_period
            lo, hi = max(TURN_LIMITS_DEG[0], self.turn_from - 90.0), min(TURN_LIMITS_DEG[1], self.turn_from + 90.0)
            self.turn_target = min(max(self.turn_target, lo), hi)
            return
        if self.turn_dir < 0:
            self.turn_target = 90.0 * math.floor((self.turn_target + TURN_COMMIT_DEG) / 90.0)
        elif self.turn_dir > 0:
            self.turn_target = 90.0 * math.ceil((self.turn_target - TURN_COMMIT_DEG) / 90.0)
        else:
            self.turn_target = 90.0 * round(self.turn_target / 90.0)
        self.turn_from = self.turn_target


class DanceJudge:
    """The judges' count of the dance's stops, on the true heading and roll, every physics step."""
    def __init__(self, state, stage: int = 0, turned: float = 0.0, rolled: float = 0.0):
        self.true_turned, self._last_true = turned, state.heading
        self.true_rolled, self._last_roll = rolled, state.roll
        self.stage, self.held = stage, 0.0

    @property
    def done(self) -> bool:
        return self.stage >= len(STAGES)

    def err_deg(self) -> float:
        axis, angle = STAGES[min(self.stage, len(STAGES) - 1)]
        return (self.true_turned if axis == "yaw" else self.true_rolled) - angle

    def update(self, s, dt: float) -> Optional[str]:
        """Counts this physics step; returns "lost" once more than 180 deg off the stop."""
        self.true_turned += angle_diff(s.heading, self._last_true)
        self.true_rolled += angle_diff(s.roll, self._last_roll)
        self._last_true, self._last_roll = s.heading, s.roll
        if self.done:
            return None
        err = self.err_deg()
        if abs(err) > 180.0:
            return "lost"
        axis, _ = STAGES[self.stage]
        if axis == "roll" and self.stage < len(STAGES) - 1:
            if abs(err) < ROLL_PASS_DEG:
                self.stage, self.held = self.stage + 1, 0.0
            return None
        rate = s.angular_velocity_z if axis == "yaw" else s.angular_velocity_x
        still = STILL_DPS if axis == "yaw" else 2 * STILL_DPS
        hold = HOLD_S if axis == "yaw" else LEVEL_HOLD_S
        if abs(err) < STOP_DEG and abs(math.degrees(rate)) < still:
            self.held += dt
            if self.held >= hold:
                self.stage, self.held = self.stage + 1, 0.0
        else:
            self.held = 0.0
        return None


class TasksDanceEnv(SubTaskEnv):
    COURSE = "tasks"
    OUTCOMES = ("is_success", "collision", "lost", "surfaced", "timeout")
    EXTRA_INFO = ("stops",)
    ACTION_DIM = 3   # No surge: exploring it shook the sub off its stops (hand controller 95% -> 50%)
    RIGHTING_SCALE = (0.5, 1.5)   # Training: the roll righting moment is a guess, so vary it widely

    def __init__(self, detector_mode: str = "noisy", control_period: float = 0.1, max_seconds: float = 120.0,
                 randomize: bool = False, side: Optional[int] = None, stage_share: float = 0.0, segment: int = 0):
        """stage_share: share of episodes that start partway through the dance (training).
        segment: train on pieces of the dance, this many stops each, from a random stop (training)."""
        self.stage_share, self.segment = stage_share, segment
        if segment:
            max_seconds = SEGMENT_SECONDS * segment
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=0.05, time_penalty=0.01,
                         progress_weight=1.0, timeout_penalty=5.0, randomize=randomize)
        self.builder = DanceObservation()
        self.dancer = DanceDriver(self.controller, control_period)
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.builder.size,), dtype=np.float32)

    def _place_sub(self, rng: random.Random):
        s, info = self.core.state, self.core.course.info
        s.x = info["gate_x"] + rng.uniform(1.0, 2.5)
        s.y = info["gate_center_y"] + rng.choice((1, -1)) * 0.762 + rng.uniform(-0.4, 0.4)
        s.heading = rng.uniform(-15.0, 15.0) % 360
        s.z = rng.uniform(1.2, 1.6)
        speed = rng.uniform(0.0, 0.3)
        s.velocity_x, s.velocity_y = speed * math.cos(math.radians(s.heading)), speed * math.sin(math.radians(s.heading))
        if self.segment:
            self._start_stage = rng.randint(0, len(STAGES) - 1)
        else:
            self._start_stage = rng.randint(1, len(STAGES) - 1) if rng.random() < self.stage_share else 0
        self._start_heading = s.heading
        k = self._start_stage
        if 0 < k <= len(YAW_STOPS):
            s.heading = (s.heading + YAW_STOPS[k - 1]) % 360
        elif k > len(YAW_STOPS):
            s.roll = (ROLL_STOPS[k - 1 - len(YAW_STOPS)] + 180.0) % 360 - 180.0
        if self.randomize:
            self.core.config.rollRightingMoment *= rng.uniform(*self.RIGHTING_SCALE)

    def _vary_dynamics(self, rng: random.Random):
        super()._vary_dynamics(rng)
        g = self.controller.gains
        g.heading_p *= HEADING_GAIN_SCALE
        g.yaw_d *= HEADING_GAIN_SCALE

    def _on_reset(self):
        s = self.core.state
        k, n = self._start_stage, len(YAW_STOPS)
        yaw_done, roll_done = YAW_STOPS[:min(k, n)], ROLL_STOPS[:max(0, k - n)]
        turned = yaw_done[-1] if 0 < k <= n else 0.0
        rolled = roll_done[-1] if roll_done else 0.0
        deepest, rolled_most = min([0.0] + yaw_done), max([0.0] + roll_done)
        self.start_xy, self.start_heading = (s.x, s.y), self._start_heading
        self.judge = DanceJudge(s, k, turned, rolled)
        self.dancer.reset(turned)
        self.builder.reset(turned, deepest, rolled, rolled_most, stop_done=True)   # The start (or last stop) needs no hold

    @property
    def stage(self) -> int:
        return self.judge.stage

    @property
    def turn_target(self) -> float:
        return self.dancer.turn_target

    # --- Acting ---
    def step(self, action):
        self.target_heading, self.target_depth, driver = self.dancer.act(action, self.readings.heading,
                                                                         self.builder.turned)
        self.last_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self.step_with_driver(driver)

    # --- Judging ---
    def _physics_outcome(self) -> Optional[str]:
        return self.judge.update(self.core.state, self.core.config.physicsDt)

    def done_label(self) -> bool:
        return self.judge.done

    def _complete(self) -> bool:
        if self.segment:
            return self.stage >= min(len(STAGES), self._start_stage + self.segment)
        return self.done_label()

    def _terminal_reward(self, outcome: str) -> float:
        return 5.0 if outcome == "success" else -5.0

    def _potential(self) -> float:
        s = self.core.state
        drift = math.hypot(s.x - self.start_xy[0], s.y - self.start_xy[1])
        if self.stage >= len(STAGES):
            stages_part, off_axis = 3.0 * len(STAGES), 0.0
        else:
            axis, _ = STAGES[self.stage]
            hold = HOLD_S if axis == "yaw" else LEVEL_HOLD_S
            stages_part = 3.0 * self.stage - abs(self.judge.err_deg()) / 90.0 + self.judge.held / hold
            off_axis = abs(s.roll) / 90.0 if axis == "yaw" else abs(self.judge.true_turned) / 90.0
        return stages_part - off_axis - 1.0 * abs(s.z - TARGET_DEPTH_M) - 0.5 * drift

    def _observe(self) -> np.ndarray:
        r = self.readings
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_z, r.roll, r.imu.gyro_x,
                                  self.turn_target, self.control_period)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _episode_info(self) -> dict:
        return {"stops": self.stage}

    def scripted_mission(self) -> List:
        """No scripted dance (it's learned only); the baseline just holds station, for train.py's comparison."""
        from ai.tasks.stabilize_task import StabilizeTask
        return [StabilizeTask(duration=self.max_seconds, target_depth=TARGET_DEPTH_M)]
