#!/usr/bin/env python3
"""
Records the states the mission runner hands to each skill: the sub's full
physical state (position, heading, depth, velocities) and what the thrusters
are doing, at every switch into ORBIT and into RETURN. Training can then start
episodes from these real handoffs (train.py --mission-starts), not just from
idealized starts.

Usage: .venv/bin/python tools/record_handoffs.py --missions 200 --out runs/handoffs.json
Seeds start at 40000, apart from training (0+) and the scored tests (10000+, 20000+, 30000+).
"""
import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE_FIELDS = ("x", "y", "z", "heading", "pitch", "velocity_x", "velocity_y", "velocity_z",
                "angular_velocity_z", "angular_velocity_y")
THRUST_FIELDS = ("hfl", "hfr", "hal", "har", "vl", "vr")


def snapshot(core) -> dict:
    s, t = core.state, core.thrust
    return {**{k: getattr(s, k) for k in STATE_FIELDS}, **{f"thrust_{k}": getattr(t, k) for k in THRUST_FIELDS}}


def record(seeds):
    from mission import MissionEnv, MissionRunner
    out = {"ORBIT": [], "RETURN": []}
    runner = MissionRunner()
    original_switch = runner._switch

    def switch(env, phase):
        if phase in out:
            out[phase].append(snapshot(env.core))
        original_switch(env, phase)
    runner._switch = switch

    for seed in seeds:
        env = MissionEnv(detector_mode="noisy", randomize=seed % 2 == 1)
        obs, _ = env.reset(seed=seed)
        runner.reset(env)
        with redirect_stdout(io.StringIO()):
            while True:
                obs, _, term, trunc, _ = runner.step(env, obs)
                if term or trunc:
                    break
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--missions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=40_000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="runs/handoffs.json")
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.missions))
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    with Pool(args.workers) as pool:
        parts = pool.map(record, chunks)
    merged = {"ORBIT": [], "RETURN": []}
    for p in parts:
        for k in merged:
            merged[k] += p[k]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f)
    print(f"{len(merged['ORBIT'])} orbit and {len(merged['RETURN'])} return handoffs from {args.missions} missions -> {args.out}")


if __name__ == "__main__":
    main()
