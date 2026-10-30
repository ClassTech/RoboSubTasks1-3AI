#!/usr/bin/env python3
"""
Scores the learned tasks 1-3 mission (TasksMissionRunner) over many starts in
parallel, optionally with other skill policies than the runner's defaults.

Usage: .venv/bin/python tools/score_tasks_mission.py --episodes 60 --seed 10000 [--slalom runs/x/best_model.zip]
"""
import argparse
import collections
import os
import statistics
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fly(job):
    seeds, paths, randomize = job
    import envs
    from tasks_mission import TasksMissionRunner
    env = envs.make("tasks_mission", "noisy", **({"randomize": True} if randomize else {}))
    runner = TasksMissionRunner(**paths)
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        runner.reset(env)
        while True:
            obs, _, term, trunc, info = runner.step(env, obs)
            if term or trunc:
                break
        out.append((seed, next(k for k in env.OUTCOMES if info[k]), info["time_s"], runner.state_name(),
                    info["dance_stops"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--randomize", action="store_true")
    for skill in ("gate", "slalom", "turnaround", "dance"):
        parser.add_argument(f"--{skill}", help=f"The {skill} policy (.zip, with its done_detector.npz beside it)")
    parser.add_argument("--no-dance", action="store_true", help="Leave the dance out")
    args = parser.parse_args()
    paths = {f"{k}_path": getattr(args, k) for k in ("gate", "slalom", "turnaround", "dance") if getattr(args, k)}
    if args.no_dance:
        paths["dance_path"] = None
    seeds = list(range(args.seed, args.seed + args.episodes))
    with Pool(args.workers) as pool:
        results = sorted(r for part in pool.map(fly, [(seeds[i::args.workers], paths, args.randomize)
                                                      for i in range(args.workers)]) for r in part)
    counts = collections.Counter(r[1] for r in results)
    n, ok = len(results), counts["is_success"]
    print(f"{ok}/{n} ({100 * ok / n:.0f}%) " + ", ".join(f"{k} {v}" for k, v in counts.items() if k != "is_success"))
    times = [r[2] for r in results if r[1] == "is_success"]
    if times:
        print(f"time to success: median {statistics.median(times):.0f} s")
    stops = [r[4] for r in results]
    print(f"dance: {sum(s == 16 for s in stops)}/{n} danced in full, mean {statistics.mean(stops):.1f} of 16 stops")
    for seed, outcome, _, step, _ in results:
        if outcome != "is_success":
            print(f"  {seed}: {outcome} during {step}")


if __name__ == "__main__":
    main()
