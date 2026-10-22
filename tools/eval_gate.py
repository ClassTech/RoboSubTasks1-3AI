#!/usr/bin/env python3
"""
Scores a controller on a skill environment (--task gate or orbit) over a fixed
set of seeds, so different controllers face exactly the same starts and noise.

Controllers:
  scripted   the scripted tasks for that skill (the baseline the policy has to beat)
  random     random actions
  straight   hold the start heading at 1.5 m and drive forward
  <path>     a saved Stable-Baselines3 PPO model (.zip), once training exists

Usage: .venv/bin/python tools/eval_gate.py scripted --task orbit --detector noisy --episodes 50
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs                                             # noqa: E402
from gate_controllers import make_controller, run_episode  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("controller", help="scripted, random, straight, or a path to a PPO model .zip")
    parser.add_argument("--task", choices=sorted(envs.ENVS), default="gate")
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="noisy")
    parser.add_argument("--randomize", action="store_true",
                        help="Vary PID gains, inertia, drag and thrust each episode, as in training")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000, help="First seed; evaluation seeds stay apart from training")
    args = parser.parse_args()

    env = envs.make(args.task, args.detector, **({"randomize": True} if args.randomize else {}))
    OUTCOMES = env.OUTCOMES
    controller = make_controller(args.controller, action_dim=env.ACTION_DIM)
    run = lambda seed: run_episode(env, controller, seed)

    counts = dict.fromkeys(OUTCOMES, 0)
    times, rewards, energies = [], [], []
    extra = {k: [] for k in env.EXTRA_INFO}
    for i in range(args.episodes):
        info, total = run(args.seed + i)
        for k in OUTCOMES:
            counts[k] += bool(info[k])
        if info["is_success"]:
            times.append(info["time_s"])
        rewards.append(total)
        energies.append(info["energy"])
        for k in extra:
            extra[k].append(info[k])

    n = args.episodes
    varied = ", varied dynamics" if args.randomize else ""
    print(f"{args.controller} on {n} {args.task} episodes ({args.detector} detector{varied}, seeds {args.seed}-{args.seed + n - 1}):")
    failures = ", ".join(f"{k.replace('_', ' ')} {counts[k]}" for k in OUTCOMES if k != "is_success")
    print(f"  success {counts['is_success']}/{n} ({100 * counts['is_success'] / n:.0f}%), {failures}")
    if times:
        print(f"  time to success: median {statistics.median(times):.1f} s, max {max(times):.1f} s")
    print(f"  reward: mean {statistics.mean(rewards):.2f}   energy: mean {statistics.mean(energies):.2f}")
    for k, values in extra.items():
        unit = next((u for suffix, u in (("_deg", "deg"), ("_s", "s"), ("_m", "m")) if k.endswith(suffix)), None)
        if unit:
            name = k[:k.rfind("_")].replace("_", " ")
            worst = min(values) if unit == "m" else max(values)   # Closest pass; longest time / largest angle
            print(f"  {name}: mean {statistics.mean(values):.2f} {unit}, worst {worst:.2f} {unit}")
        else:
            print(f"  {k.replace('_', ' ')}: mean {100 * statistics.mean(values):.0f}%")


if __name__ == "__main__":
    main()
