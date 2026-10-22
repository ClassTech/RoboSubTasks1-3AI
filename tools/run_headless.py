#!/usr/bin/env python3
"""
Runs the scripted mission several times without a window and reports how each
run went. Each run gets new random sensor and detector errors.

Usage: .venv/bin/python tools/run_headless.py --detector noisy --runs 10
"""
import argparse
import io
import math
import os
import random
import sys
from contextlib import redirect_stdout

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.submarine import Submarine           # noqa: E402
from main import build_mission               # noqa: E402
from simulator import SubmarineSimulator     # noqa: E402

DT = 1 / 60


def run_once(detector: str, max_seconds: float, seed: int):
    random.seed(seed)
    log = io.StringIO()
    with redirect_stdout(log):
        sim = SubmarineSimulator(Submarine(mission_plan=build_mission()), detector_mode=detector)
        sim.renderCamera = detector == "opencv"
        t, closest = 0.0, float("inf")
        ai = sim.submarineAI
        m = sim.core.marker
        while t < max_seconds and ai.current_task_index < len(ai.mission_plan):
            sim.step(DT)
            t += DT
            s = sim.core.state
            closest = min(closest, math.hypot(s.x - m.x, s.y - m.y))
    done = ai.current_task_index >= len(ai.mission_plan)
    failed = "FAILED" in log.getvalue()
    last_task = "complete" if done else f"{ai.get_current_task_name()}/{ai.get_current_state_name()}"
    return (done and not failed and sim.core.gate_passes >= 2, t, sim.core.gate_passes, sim.core.collision_count,
            last_task, log.getvalue(), closest)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detector", choices=["opencv", "perfect", "noisy"], default="noisy")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-log", action="store_true", help="Print the mission log of failed runs")
    args = parser.parse_args()

    successes, closest_marker = 0, []
    for i in range(args.runs):
        ok, t, passes, collisions, last, log, closest = run_once(args.detector, args.max_seconds, args.seed + i)
        successes += ok
        closest_marker.append(closest)
        print(f"run {i + 1}: {'OK  ' if ok else 'FAIL'} {t:6.1f}s  gate passes {passes}  "
              f"collisions {collisions}  closest to marker {closest:.2f} m  ended in {last}")
        if not ok and args.show_log:
            print("  " + "\n  ".join(log.strip().splitlines()[-15:]))
    print(f"\n{successes}/{args.runs} runs completed the mission ({args.detector} detector); "
          f"closest pass to the marker {min(closest_marker):.2f} m, "
          f"{sum(c < 1.5 for c in closest_marker)} runs inside the 1.5 m margin")


if __name__ == "__main__":
    main()
