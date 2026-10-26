#!/usr/bin/env python3
"""The RL skills by name, for the tools' --task option."""
from gate_env import GateEnv, GateReturnEnv
from mission import MissionEnv
from tasks_mission import TasksMissionEnv
from tasks_gate_env import TasksGateEnv
from tasks_slalom_env import TasksSlalomEnv
from tasks_turnaround_env import TasksTurnaroundEnv
from tasks_dance_env import TasksDanceEnv
from orbit_env import OrbitEnv

ENVS = {"gate": GateEnv, "gate_return": GateReturnEnv, "orbit": OrbitEnv, "mission": MissionEnv, "tasks_mission": TasksMissionEnv, "tasks_gate": TasksGateEnv, "tasks_slalom": TasksSlalomEnv, "tasks_turnaround": TasksTurnaroundEnv, "tasks_dance": TasksDanceEnv}


def make(task: str, detector_mode: str = "perfect", **kwargs):
    return ENVS[task](detector_mode=detector_mode, **kwargs)
