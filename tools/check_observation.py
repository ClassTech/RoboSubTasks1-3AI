#!/usr/bin/env python3
"""
Checks the policy observation (observation.py) against ground truth.

Flies the scripted GateTask through the gate environment and, at every 10 Hz
step, compares what the observation says with the truth:
  - gate bearing and range, while the gate is in view
  - the turn needed to face straight through the gate, and the sideways
    distance from its center line, while the gate is in view
  - whether a pole seen on its own (close to the gate) is labeled the right
    side (left/right)
Then checks that a remembered bearing is corrected as the sub turns.

Usage: .venv/bin/python tools/check_observation.py [--detector noisy|perfect] [--episodes N]
"""
import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_controllers import make_controller          # noqa: E402
from gate_env import GateEnv                           # noqa: E402
from observation import GATE_CONFIRM_FRAMES, BEARING_SCALE, INDEX, MAX_UNSEEN, ObservationBuilder  # noqa: E402


def wrap(deg):
    return (deg + 180) % 360 - 180


def truth(core):
    """True gate bearing/range, approach turn, center-line offset, and each pole's bearing, as the sub sees them."""
    s, g = core.state, core.gate
    def bearing_to(x, y):
        return wrap(math.degrees(math.atan2(y - s.y, x - s.x)) - s.heading)
    facing_plus_x = s.x < g.x               # Which side of the gate the sub is on
    through = 0.0 if facing_plus_x else 180.0
    side = 1 if facing_plus_x else -1       # +y is on the left when going toward +x
    poles = {"left": bearing_to(g.x, g.center_y + side * g.width / 2),
             "right": bearing_to(g.x, g.center_y - side * g.width / 2)}
    return {"bearing": bearing_to(g.x, g.center_y), "range": math.hypot(g.x - s.x, g.center_y - s.y),
            "approach": wrap(through - s.heading), "offset": side * (s.y - g.center_y), "poles": poles}


def summarize(name, values, unit):
    if not values:
        print(f"  {name}: no samples")
        return
    a = sorted(abs(v) for v in values)
    print(f"  {name}: n={len(a)}  median |err| {statistics.median(a):.3f}{unit}  "
          f"95th pct {a[int(0.95 * (len(a) - 1))]:.3f}{unit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detector", choices=["noisy", "perfect"], default="noisy")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    env = GateEnv(detector_mode=args.detector)
    controller = make_controller("scripted")
    print(f"Observation: {env.builder.frame_size} values per frame x {env.builder.stack} frames = {env.builder.size}")
    err = {k: [] for k in ("bearing", "range", "approach", "offset")}
    lone_correct = lone_total = 0
    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        controller.reset(env)
        while True:
            obs, _, term, trunc, info = controller.step(env, obs)
            b, t = env.builder, truth(env.core)
            heading = env.readings.heading
            if b.gate.found:
                err["bearing"].append(b.gate.bearing(heading) - t["bearing"])
                err["range"].append((b.gate.range_m - t["range"]) / t["range"])
                if b.gate.geometry_known:
                    err["approach"].append(wrap(b.gate.abs_approach - heading - t["approach"]))
                    err["offset"].append(b.gate.offset_m - t["offset"])
            elif b.left.found != b.right.found:     # A lone pole: is it labeled the right side?
                side = "left" if b.left.found else "right"
                seen = b.tracks[side].bearing(heading)
                other = "right" if side == "left" else "left"
                lone_total += 1
                lone_correct += abs(wrap(seen - t["poles"][side])) < abs(wrap(seen - t["poles"][other]))
            if term or trunc:
                successes += info["is_success"]
                break

    print(f"{successes}/{args.episodes} scripted episodes succeeded ({args.detector} detector)\n")
    print("Gate while in view:")
    summarize("bearing error", err["bearing"], " deg")
    summarize("range error (fraction of true range)", err["range"], "")
    summarize("approach turn error", err["approach"], " deg")
    summarize("center-line offset error", err["offset"], " m")
    if lone_total:
        print(f"Lone pole labeled the correct side: {lone_correct}/{lone_total} steps "
              f"({100 * lone_correct / lone_total:.1f}%)")
    check_memory(env.core.camera)


def check_memory(camera):
    """See the gate dead ahead at heading 100, lose it, turn to 130: it should now be 30 deg right."""
    b = ObservationBuilder(camera)
    w, h = camera.width, camera.height
    poles = [{'center_x': x, 'center_y': h / 2, 'width': 6, 'height': 200, 'min_x': x - 3, 'max_x': x + 3,
              'min_y': h / 2 - 100, 'max_y': h / 2 + 100, 'area': 1000} for x in (w / 2 - 100, w / 2 + 100)]
    for _ in range(GATE_CONFIRM_FRAMES):   # A pair counts as the gate once seen this many frames in a row
        b.update(1.5, 0.0, 100.0, 0.0, 0.0, 0.0, poles, [0, 0, 0], 0.1)
    obs = b.update(1.5, 0.0, 130.0, 0.0, 0.0, 0.0, [], [0, 0, 0], 0.1)
    found, bearing, unseen = obs[INDEX["gate_found"]], obs[INDEX["gate_bearing"]], obs[INDEX["gate_unseen"]]
    bearing_deg, unseen_s = bearing * BEARING_SCALE, unseen * MAX_UNSEEN
    ok = not found and abs(bearing_deg + 30) < 1e-3 and abs(unseen_s - 0.1) < 1e-6
    print(f"\nMemory while turning: found={found:.0f}, bearing {bearing_deg:+.1f} deg (expect -30, i.e. right), "
          f"unseen {unseen_s:.1f} s -> {'OK' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()
