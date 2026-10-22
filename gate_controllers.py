#!/usr/bin/env python3
"""
Controllers that can fly the skill environments (gate, orbit), shared by
tools/eval_gate.py (scoring), train.py and watch.py (viewing):
  scripted   the environment's scripted tasks (the baseline)
  random     random actions
  straight   hold the start heading at 1.5 m and drive forward
  mission    the learned gate and orbit skills run in turn (mission.py), for --task mission
  <path>     a saved Stable-Baselines3 PPO model (.zip)

Each controller's step(env, obs) advances the environment one control period
and returns what env.step() returns.

A PPO model "<name>.zip" is used with the observation normalization saved next
to it as "<name>_vecnormalize.pkl", if present.
"""
import io
import os
import pickle
from contextlib import redirect_stdout

import numpy as np

from sub_env import DEPTH_MAX, DEPTH_MIN, SubTaskEnv


class ActionController:
    """Chooses an action from the observation each step."""
    def __init__(self, name, choose):
        self.name, self.choose, self.last_action = name, choose, None

    def reset(self, env: SubTaskEnv):
        self.last_action = None

    def step(self, env: SubTaskEnv, obs):
        self.last_action = np.asarray(self.choose(obs), dtype=np.float32)
        return env.step(self.last_action)


class _ImageSize:
    """Stands in for the camera image: the scripted tasks only ask for its size."""
    def __init__(self, w, h): self.w, self.h = w, h
    def get_size(self): return self.w, self.h


class ScriptedController:
    """The environment's scripted tasks (env.scripted_mission()), driving the thrusters at the physics rate."""
    name, last_action = "scripted", None

    def reset(self, env: SubTaskEnv):
        from ai.submarine import Submarine
        with redirect_stdout(io.StringIO()):
            self.sub = Submarine(mission_plan=env.scripted_mission())
        self.image = _ImageSize(env.core.camera.width, env.core.camera.height)

    def state_name(self) -> str:
        return self.sub.get_current_state_name()

    def step(self, env: SubTaskEnv, obs):
        from data_structures import SensorSuite

        def driver(r, detections, dt):
            s = env.core.state   # Ground truth: only the scripted hover/damping helpers use it (SIM ONLY)
            sensors = SensorSuite(camera_image=self.image, depth=r.depth, heading=r.heading, pitch=r.pitch,
                                  imu=r.imu, depth_rate=r.depth_rate, x=s.x, y=s.y,
                                  velocity_x=s.velocity_x, velocity_y=s.velocity_y,
                                  angular_velocity_y=s.angular_velocity_y, velocity_z=s.velocity_z,
                                  detections=detections)
            with redirect_stdout(io.StringIO()):
                commands, _ = self.sub.update(dt, sensors)
            return commands
        return env.step_with_driver(driver)


def run_episode(env: SubTaskEnv, controller, seed: int):
    """Flies one episode. Returns the final info and the total reward."""
    obs, _ = env.reset(seed=seed)
    controller.reset(env)
    total = 0.0
    while True:
        obs, reward, term, trunc, info = controller.step(env, obs)
        total += reward
        if term or trunc:
            return info, total


def vecnormalize_path(model_path: str) -> str:
    return model_path[:-len(".zip")] + "_vecnormalize.pkl" if model_path.endswith(".zip") else model_path + "_vecnormalize.pkl"


def policy_controller(model, normalizer=None, name: str = "policy") -> ActionController:
    """A trained model as a controller. normalizer: a VecNormalize whose stats the model was trained with."""
    def choose(obs):
        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)
        return model.predict(obs, deterministic=True)[0]
    return ActionController(name, choose)


def load_policy(path: str):
    """A saved PPO model and its observation normalization (None if there is none)."""
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    normalizer = None
    if os.path.exists(vecnormalize_path(path)):
        with open(vecnormalize_path(path), "rb") as f:
            normalizer = pickle.load(f)
    return model, normalizer


def make_controller(name: str, seed: int = 0, action_dim: int = 3):
    if name == "scripted":
        return ScriptedController()
    if name == "random":
        rng = np.random.default_rng(seed)
        return ActionController("random", lambda obs: rng.uniform(-1, 1, action_dim))
    if name == "straight":
        depth_action = (1.5 - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 2 - 1   # Target 1.5 m
        return ActionController("straight", lambda obs: np.array([0.0, depth_action, 0.6] + [0.0] * (action_dim - 3)))
    if name == "mission":
        from mission import MissionRunner
        return MissionRunner()
    model, normalizer = load_policy(name)
    return policy_controller(model, normalizer, name)
