#!/usr/bin/env python3
"""
Scores the tasks gate skill from the learned mission's own starts (RoboSim's:
9.144 m before the gate, +-0.914 m sideways, heading 0, 90 or 270 deg, 0.5 m
deep), going out, with its done detector running as in the mission. Counts
the outcomes and how often the detector fires before the sub is through (the
mission then hands the slalom a sub still in front of the gate).

Usage: .venv/bin/python tools/score_gate_starts.py --episodes 200 [--randomize] [--policy runs/x/best_model.zip]
"""
import argparse
import collections
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POLICY = "runs/tgate3_s0/best_model.zip"


def fly(job):
    seeds, policy, randomize = job
    import random
    import envs
    from done_detector import DoneDetector, detector_path
    from gate_controllers import load_policy
    from tasks_gate_env import TasksGateStarts
    model, norm = load_policy(policy)
    detector = DoneDetector(detector_path(policy))
    starts = TasksGateStarts(return_share=0.0, forward_distance_m=(9.144, 9.144), lateral_m=(-0.914, 0.914),
                             course_heading_share=1.0, depth_m=(0.5, 0.5))
    env = envs.make("tasks_gate", "noisy", starts=starts, **({"randomize": True} if randomize else {}))
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        detector.reset()
        early = None
        while True:
            if early is None and detector.update(obs, env.readings.heading):
                early = env._past_gate_m() < 0.0
            obs, _, term, trunc, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            if term or trunc:
                break
        out.append((seed, next(k for k in env.OUTCOMES if info[k]), bool(early)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--policy", default=POLICY)
    args = parser.parse_args()
    seeds = list(range(args.seed, args.seed + args.episodes))
    with Pool(args.workers) as pool:
        results = [r for part in pool.map(fly, [(seeds[i::args.workers], args.policy, args.randomize)
                                                for i in range(args.workers)]) for r in part]
    counts = collections.Counter(r[1] for r in results)
    early = sum(r[2] for r in results)
    n = len(results)
    print(f"{counts['is_success']}/{n} through; " + ", ".join(f"{k} {v}" for k, v in counts.items() if k != "is_success")
          + f"; detector fired before the gate in {early}")


if __name__ == "__main__":
    main()
