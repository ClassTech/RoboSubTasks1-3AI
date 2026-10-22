#!/usr/bin/env python3
"""
The full qualification run with the learned skills: through the gate, around
the marker, and back through the gate.

MissionEnv is the whole course from the scripted mission's start (3 m before
the gate, at the surface). It judges the run on ground truth: success needs a
forward pass through the gate opening, at least 270° swept around the marker,
then a pass back through the opening. Failures: a collision, crossing the gate
line outside the opening, coming back through the gate without having gone
around the marker (skipped_orbit), surfacing, or running out of time.

MissionRunner runs the learned skills in turn. Each skill decides for itself
when it's finished, with its done detector (done_detector.py, a small network
trained on the skill's own flights), so the runner only sequences them:
  DIVE   (scripted)      -> GATE   once the measured depth is past DIVE_DONE_M and
                                    steady (the run starts at the surface 3 m from
                                    the gate, too close for the gate policy to dive)
  GATE   (gate policy)   -> ORBIT  when the gate's detector says done (trained on: through
                                    the gate and 1.5 m clear of it)
  ORBIT  (orbit policy)  -> RETURN when the orbit's detector says done (trained on: 270°
                                    around the marker and facing the gate)
  RETURN (gate policy)   -> DONE   when the gate's detector says done
Each skill gets its own observation builder (the two policies see different
layouts), reset when the skill takes over, as in training.

Depth limits: the gate and orbit skills command depths between MIN_DEPTH_M
and MAX_DEPTH_M, matching the limits their environments train with. The
opening is below 1.15 m, and both policies tended to rise and surface
otherwise; below 1.65 m the camera is too close to the pole feet to range them.
"""
import math
import random
from typing import List, Optional

import numpy as np

from observation import ObservationBuilder
from sub_env import SubTaskEnv

GATE_POLICY = "runs/gate_phys6_ft/final_model.zip"
DIVE_DEPTH_M = 1.5
DIVE_DONE_M = 1.3           # Dive is done below this...
DIVE_STEADY_MPS = 0.1       # ...once the depth rate has settled under this
MIN_DEPTH_M = 1.2
MAX_DEPTH_M = 1.65
ORBIT_POLICY = "runs/orbit_straight_in2/final_model.zip"

SWEEP_DEG = 270.0           # Ground truth: what counts as having gone around the marker
MIN_MARKER_DISTANCE_M = 1.5


def _wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


class MissionEnv(SubTaskEnv):
    OUTCOMES = ("is_success", "collision", "missed_gate", "skipped_orbit", "surfaced", "timeout")
    EXTRA_INFO = ("inside_margin_s", "closest_m")
    INCLUDE_MARKER = True
    ACTION_DIM = 4

    def __init__(self, detector_mode: str = "noisy", max_seconds: float = 300.0, randomize: bool = False):
        super().__init__(detector_mode, max_seconds=max_seconds, timeout_penalty=0.0, randomize=randomize)

    def _place_sub(self, rng: random.Random):
        s, g = self.core.state, self.core.gate
        s.x = self.core.prequal.START_X_POS
        s.y = g.center_y + rng.uniform(-0.5, 0.5)
        s.z = self.core.prequal.START_Z_POS
        s.heading = rng.uniform(-10.0, 10.0) % 360

    def _marker_angle(self) -> float:
        s, m = self.core.state, self.core.marker
        return math.degrees(math.atan2(s.y - m.y, s.x - m.x))

    def _on_reset(self):
        self.through_gate = False
        self.swept = 0.0
        self._last_angle = self._marker_angle()
        self._inside_margin_s, self._closest = 0.0, float("inf")

    def _physics_outcome(self) -> Optional[str]:
        s, m = self.core.state, self.core.marker
        if self.through_gate:
            angle = self._marker_angle()
            self.swept += _wrap(angle - self._last_angle)
            self._last_angle = angle
        d = math.hypot(s.x - m.x, s.y - m.y)
        self._closest = min(self._closest, d)
        if d < MIN_MARKER_DISTANCE_M:
            self._inside_margin_s += self.core.config.physicsDt
        for c in self.new_crossings():
            if not c.inside_opening:
                return "missed_gate"
            if not self.through_gate and c.direction > 0:
                self.through_gate = True
                self._last_angle = self._marker_angle()
            elif self.through_gate and c.direction < 0:
                return "success" if abs(self.swept) >= SWEEP_DEG else "skipped_orbit"
            else:
                return "missed_gate"
        return None

    def _potential(self) -> float:
        return 0.0   # Only for scoring, not training

    def _terminal_reward(self, outcome: str) -> float:
        return 0.0

    def _episode_info(self) -> dict:
        return {"inside_margin_s": self._inside_margin_s, "closest_m": self._closest}

    def scripted_mission(self) -> List:
        from main import build_mission
        return build_mission()


class MissionRunner:
    """Runs the learned gate and orbit skills in turn; each one's done detector says when to move on."""
    name = "mission"
    NEXT = {"GATE": "ORBIT", "ORBIT": "RETURN", "RETURN": "DONE"}

    def __init__(self, gate_path: str = GATE_POLICY, orbit_path: str = ORBIT_POLICY):
        from done_detector import DoneDetector, detector_path
        from gate_controllers import load_policy
        self.gate_model, self.gate_norm = load_policy(gate_path)
        self.orbit_model, self.orbit_norm = load_policy(orbit_path)
        self.gate_done = DoneDetector(detector_path(gate_path))
        self.orbit_done = DoneDetector(detector_path(orbit_path))
        self.last_action = None

    def reset(self, env):
        cam = env.core.camera
        self.gate_builder = ObservationBuilder(cam, action_dim=3)
        self.orbit_builder = ObservationBuilder(cam, action_dim=4, include_marker=True)
        self.phase = "DIVE"
        self.dive_heading = env.readings.heading
        self.skill_action = np.zeros(3, dtype=np.float32)
        self.last_action = None
        self.switch_times = {}

    def state_name(self) -> str:
        return self.phase

    def _update(self, env, builder) -> np.ndarray:
        r, d = env.readings, env.detections
        return builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                              d.red_blobs if d else [], self.skill_action, env.control_period,
                              green_boxes=d.green_blobs if d else [])

    def _switch(self, env, phase: str):
        self.phase = phase
        self.switch_times[phase] = env.time
        self.skill_action = np.zeros(4 if phase == "ORBIT" else 3, dtype=np.float32)
        if phase == "ORBIT":
            self.orbit_builder.reset()
            self.orbit_done.reset()
        elif phase in ("GATE", "RETURN"):
            self.gate_builder.reset()
            self.gate_done.reset()

    @staticmethod
    def _depth_action(depth_m: float) -> float:
        return (depth_m - 0.3) / (1.9 - 0.3) * 2 - 1

    def _skill(self, env):
        """The current skill's action, and whether its detector says the skill is done."""
        orbit = self.phase == "ORBIT"
        builder, model, norm, detector = ((self.orbit_builder, self.orbit_model, self.orbit_norm, self.orbit_done)
                                          if orbit else
                                          (self.gate_builder, self.gate_model, self.gate_norm, self.gate_done))
        obs = self._update(env, builder)
        a = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        return np.asarray(a, dtype=np.float32), detector.update(obs, env.readings.heading)

    def step(self, env, obs):
        r = env.readings
        if self.phase == "DIVE" and r.depth > DIVE_DONE_M and abs(r.depth_rate) < DIVE_STEADY_MPS:
            self._switch(env, "GATE")
        if self.phase in self.NEXT:
            self.skill_action, done = self._skill(env)
            if done:
                self._switch(env, self.NEXT[self.phase])
                if self.phase in self.NEXT:
                    self.skill_action, _ = self._skill(env)   # The next skill acts on this step already
        if self.phase in self.NEXT:
            action = np.append(self.skill_action, 0.0) if self.phase != "ORBIT" else self.skill_action.copy()
            action[1] = float(np.clip(action[1], self._depth_action(MIN_DEPTH_M),
                                      self._depth_action(MAX_DEPTH_M)))   # Depth limits, as in training
        else:
            # DIVE: hold the start heading and dive to DIVE_DEPTH_M. DONE: hold position at depth.
            turn = float(np.clip(_wrap(self.dive_heading - r.heading) / 45.0, -1, 1)) if self.phase == "DIVE" else 0.0
            action = np.array([turn, self._depth_action(DIVE_DEPTH_M), 0.0, 0.0], dtype=np.float32)
        self.last_action = action
        return env.step(action)
