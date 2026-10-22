#!/usr/bin/env python3
"""
Base class for the RL environments (one per skill: gate, orbit, ...).

The policy acts at 10 Hz. Each action sets targets that the sub's own control
loops (controller.py) then hold at the physics rate until the next action:
  action[0]  heading change from the current heading, -1..1 -> -45..+45 deg (+ = left)
  action[1]  target depth, -1..1 -> 0.3..1.9 m
  action[2]  surge effort (open-loop forward/back thrust), -1..1
  action[3]  sway effort (open-loop sideways thrust, + = left), -1..1; only skills with ACTION_DIM = 4
The policy sees observation.py's observation, built only from what the real
sub measures.

Each skill subclass decides where episodes start, how they end, and what
counts as progress:
  _place_sub(rng)          put the sub at a random start
  _physics_outcome()       after each physics step: an outcome name, or None
  _step_outcome()          after each 10 Hz step (once the observation is updated): outcome or None
  _potential()             progress measure (ground truth); reward is its change each step
  _terminal_reward(name)   bonus or penalty for how the episode ended
  _step_cost()             optional extra cost each 10 Hz step (default 0)
  _complete()              success check after each 10 Hz step: True ends the episode as a success
                           (default: the skill decides in _step_outcome instead)
  done_label()             ground truth for the skill's done detector (done_detector.py): is the job
                           done now? Defaults to _complete()
  _episode_info()          optional extra numbers for info at the end (named in EXTRA_INFO)

With randomize=True (training), each episode scales the controller gains,
inertia, drag and thrust by factors from config.RandomizationConfig, after the
start is placed (so a seed gives the same start either way).
  scripted_mission()       the scripted tasks that do this skill, for comparison
OUTCOMES lists the possible endings ("is_success", "timeout" and the
skill's own failures); info carries them as booleans when an episode ends,
plus time_s and energy. Collisions always end an episode, and so does
surfacing (as in a RoboSub run): coming back up to SURFACE_DEPTH_M after
having been below SUBMERGED_DEPTH_M.
"""
import random
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np

from config import RandomizationConfig
from controller import ControllerGains, SetpointController
from core_types import ThrusterCommands
from observation import ObservationBuilder
from sensor_model import SensorReadings
from sim_core import SubSim

HEADING_CHANGE_DEG = 45.0
SUBMERGED_DEPTH_M = 0.3     # Below this, the sub counts as having dived (episodes can start near the surface)
SURFACE_DEPTH_M = 0.05      # Back up to here after diving = surfaced, which ends the run
DEPTH_MIN, DEPTH_MAX = 0.3, 1.9

Driver = Callable[[SensorReadings, object, float], ThrusterCommands]


class SubTaskEnv(gym.Env):
    metadata = {"render_modes": []}
    OUTCOMES = ("is_success", "collision", "surfaced", "timeout")
    SURFACED_PENALTY = 5.0
    EXTRA_INFO = ()          # Names of extra per-episode numbers from _episode_info()
    INCLUDE_MARKER = False   # Whether the observation tracks the green marker pole
    ACTION_DIM = 3           # 4 adds sway
    MIN_TARGET_DEPTH_M = None   # If set, the policy can't command a shallower target depth (a safety limit)
    MAX_TARGET_DEPTH_M = None   # ...or a deeper one

    def __init__(self, detector_mode: str = "perfect", control_period: float = 0.1,
                 max_seconds: float = 60.0, energy_penalty: float = 0.05, time_penalty: float = 0.01,
                 progress_weight: float = 1.0, timeout_penalty: float = 5.0, randomize: bool = False):
        assert detector_mode in ("perfect", "noisy"), detector_mode
        self.detector_mode = detector_mode
        self.control_period = control_period
        self.max_seconds = max_seconds
        self.energy_penalty, self.time_penalty = energy_penalty, time_penalty
        self.progress_weight, self.timeout_penalty = progress_weight, timeout_penalty
        self.randomize = randomize
        self.dynamics = {}   # The factors drawn for this episode (empty when not randomizing)
        self.controller = SetpointController()
        self.on_physics_step = None   # Optional callback(env, thruster_commands) after each physics step, for viewers
        self.end_on_complete = True   # False keeps flying past success (done-detector data collection)
        self.core = SubSim(detector_mode=detector_mode, log=None)
        self.builder = ObservationBuilder(self.core.camera, action_dim=self.ACTION_DIM,
                                          include_marker=self.INCLUDE_MARKER)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.ACTION_DIM,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.builder.size,), dtype=np.float32)

    # --- For subclasses ---
    def _place_sub(self, rng: random.Random):
        raise NotImplementedError

    def _physics_outcome(self) -> Optional[str]:
        return None

    def _step_outcome(self) -> Optional[str]:
        return None

    def _potential(self) -> float:
        raise NotImplementedError

    def _terminal_reward(self, outcome: str) -> float:
        raise NotImplementedError

    def _on_reset(self):
        pass

    def _step_cost(self) -> float:
        return 0.0

    def _episode_info(self) -> dict:
        return {}

    def _complete(self) -> bool:
        return False

    def done_label(self) -> bool:
        return self._complete()

    def scripted_mission(self) -> List:
        raise NotImplementedError

    # --- Episode start ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(2 ** 31)))
        self.core = SubSim(detector_mode=self.detector_mode, rng=rng, log=None)
        self._place_sub(rng)
        self._vary_dynamics(rng)

        self.controller.reset()
        self.builder.reset()
        self.readings = self.core.read_sensors(0.0)
        self.detections = self.core.detect(0.0)
        self.target_heading, self.target_depth = self.readings.heading, self.readings.depth
        self.time, self.energy = 0.0, 0.0
        self._crossings_seen, self._collisions_seen = 0, 0
        self._submerged = self.core.state.z > SUBMERGED_DEPTH_M
        self._on_reset()
        self.potential = self._potential()
        self.last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        return self._observe(), {}

    def apply_recorded_start(self, rec: dict):
        """Put the sub in a recorded state (tools/record_handoffs.py): pose, velocities, thrusters."""
        s, t = self.core.state, self.core.thrust
        for k, v in rec.items():
            if k.startswith("thrust_"):
                name = {"vf": "vl"}.get(k[len("thrust_"):], k[len("thrust_"):])   # Older recordings: front/rear
                setattr(t, name, v)
            else:
                setattr(s, k, v)

    def _vary_dynamics(self, rng: random.Random):
        nominal = ControllerGains()
        if not self.randomize:
            self.controller.gains, self.dynamics = nominal, {}
            return
        rc, c = RandomizationConfig(), self.core.config
        gains = {k: (v if k == "depth_i_clamp" else v * rng.uniform(*rc.pid_scale))
                 for k, v in vars(nominal).items()}
        self.controller.gains = ControllerGains(**gains)
        d = {"inertia_yaw": rng.uniform(*rc.inertia_scale), "inertia_pitch": rng.uniform(*rc.inertia_scale),
             "drag_surge": rng.uniform(*rc.drag_scale), "drag_sway": rng.uniform(*rc.drag_scale),
             "drag_heave": rng.uniform(*rc.drag_scale), "drag_yaw": rng.uniform(*rc.drag_scale),
             "drag_pitch": rng.uniform(*rc.drag_scale), "thrust": rng.uniform(*rc.thrust_scale)}
        c.subInertia_Z *= d["inertia_yaw"]; c.subInertia_Y *= d["inertia_pitch"]
        c.surgeDragCoeff *= d["drag_surge"]; c.swayDragCoeff *= d["drag_sway"]
        c.heaveDragCoeff *= d["drag_heave"]; c.angularDragCoeff_Z *= d["drag_yaw"]
        c.angularDragCoeff_Y *= d["drag_pitch"]; c.thrusterMaxForce *= d["thrust"]
        self.dynamics = {**d, **{f"gain_{k}": gains[k] / getattr(nominal, k) for k in gains if k != "depth_i_clamp"}}

    # --- Acting ---
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.target_heading = (self.readings.heading + HEADING_CHANGE_DEG * float(a[0])) % 360
        self.target_depth = DEPTH_MIN + (float(a[1]) + 1) / 2 * (DEPTH_MAX - DEPTH_MIN)
        if self.MIN_TARGET_DEPTH_M is not None:
            self.target_depth = max(self.target_depth, self.MIN_TARGET_DEPTH_M)
        if self.MAX_TARGET_DEPTH_M is not None:
            self.target_depth = min(self.target_depth, self.MAX_TARGET_DEPTH_M)
        surge = float(a[2])
        sway = float(a[3]) if self.ACTION_DIM > 3 else 0.0

        def driver(r: SensorReadings, detections, dt: float) -> ThrusterCommands:
            return self.controller.update(dt, r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y,
                                          r.imu.gyro_z, self.target_heading, self.target_depth, surge, sway)
        self.last_action = a
        return self.step_with_driver(driver)

    def new_crossings(self) -> list:
        """Gate-plane crossings since the last call (for _physics_outcome)."""
        new = self.core.crossings[self._crossings_seen:]
        self._crossings_seen = len(self.core.crossings)
        return new

    def step_with_driver(self, driver: Driver):
        """
        Runs one control period with `driver` choosing thruster commands at every
        physics step from the latest readings and detections. step() uses the
        policy's setpoints; evaluation tools can plug in the scripted tasks instead.
        """
        dt = self.core.config.physicsDt
        n = max(1, round(self.control_period / dt))
        outcome, step_energy = None, 0.0
        for _ in range(n):
            cmd = driver(self.readings, self.detections, dt)
            self.core.advance(dt, cmd)
            step_energy += dt * (abs(cmd.hfl) + abs(cmd.hfr) + abs(cmd.hal) + abs(cmd.har)
                                 + abs(cmd.vl) + abs(cmd.vr)) / 6
            self.readings = self.core.read_sensors(dt)
            self.detections = self.core.detect(dt)
            self.time += dt
            if self.on_physics_step:
                self.on_physics_step(self, cmd)
            z = self.core.state.z
            self._submerged = self._submerged or z > SUBMERGED_DEPTH_M
            if self.core.collision_count > self._collisions_seen:
                outcome = "collision"
            elif self._submerged and z <= SURFACE_DEPTH_M:
                outcome = "surfaced"
            else:
                outcome = self._physics_outcome()
            if outcome:
                break

        obs = self._observe()
        if not outcome:
            outcome = self._step_outcome() or ("success" if self.end_on_complete and self._complete() else None)
        self.energy += step_energy
        new_potential = self._potential()
        reward = self.progress_weight * (new_potential - self.potential) - self.time_penalty
        reward -= self.energy_penalty * step_energy + self._step_cost()
        self.potential = new_potential

        terminated = outcome is not None
        truncated = not terminated and self.time >= self.max_seconds - 1e-9
        if terminated:
            reward += -self.SURFACED_PENALTY if outcome == "surfaced" else self._terminal_reward(outcome)
        elif truncated:
            reward -= self.timeout_penalty
            outcome = "timeout"
        info = {}
        if terminated or truncated:
            info = {k: (k == "is_success" and outcome == "success") or k == outcome for k in self.OUTCOMES}
            info.update(time_s=self.time, energy=self.energy, **self._episode_info())
        return obs, float(reward), terminated, truncated, info

    # --- Helpers ---
    def _observe(self) -> np.ndarray:
        r, d = self.readings, self.detections
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                  d.red_blobs if d else [], self.last_action, self.control_period,
                                  green_boxes=d.green_blobs if d else [])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
