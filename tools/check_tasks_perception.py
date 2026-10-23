#!/usr/bin/env python3
"""
Checks tasks-course perception (tasks_perception.py) against ground truth: the
scripted tasks mission flies some runs, and at every 10 Hz step the estimates
from the detector's boxes are compared with the simulator's true geometry.

Gate: whether each pole is identified as the right physical pole, its range
error, and the errors in the gate's and each passage's turn and sideways offset.
Slalom: the nearest set's range error, and the error in each lane's offset.

Usage: .venv/bin/python tools/check_tasks_perception.py --detector noisy --episodes 10
"""
import argparse
import io
import math
import os
import statistics
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_controllers import make_controller          # noqa: E402
from tasks_mission import TasksMissionEnv               # noqa: E402
from tasks_perception import find_gate, find_slalom_sets  # noqa: E402

LANE_HALF = 1.524 / 2      # Passage and lane centers are this far from the gate center or red pole


def _wrap(a):
    return (a + 180) % 360 - 180


def truth_lines(s, x, y_center, facing_forward):
    """True turn to face straight through a line at x, and the sub's offset from a center line at y_center
    (+ = left of it, as the sub sees it)."""
    turn = _wrap((0.0 if facing_forward else 180.0) - s.heading)
    offset = (s.y - y_center) if facing_forward else -(s.y - y_center)
    return turn, offset


def summarize(name, vals, unit=""):
    if not vals:
        print(f"  {name}: no samples")
        return
    a = sorted(abs(v) for v in vals)
    print(f"  {name}: n={len(a)}  median |err| {statistics.median(a):.3f}{unit}  95th pct {a[int(0.95 * (len(a) - 1))]:.3f}{unit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="noisy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()

    err = {k: [] for k in ("pole_range", "gate_turn", "gate_offset", "passage_offset",
                           "set_range", "lane_offset", "lane_turn", "gate_offset_smoothed", "lane_offset_smoothed")}
    k = 0.1 / (1.0 + 0.1)   # Exponential smoothing with a 1 s time constant, as the observation will do
    ident_ok = ident_n = sets_wrong = 0
    c = make_controller("scripted")
    for ep in range(args.episodes):
        env = TasksMissionEnv(detector_mode=args.detector)
        obs, _ = env.reset(seed=args.seed + ep)
        c.reset(env)
        info_c = env.core.course.info
        smooth = {}
        cy, gx = info_c["gate_center_y"], info_c["gate_x"]
        with redirect_stdout(io.StringIO()):
            while True:
                obs, _, term, trunc, info = c.step(env, obs)
                d, r, s, cam = env.detections, env.readings, env.core.state, env.core.camera
                if d is not None:
                    fwd = math.cos(math.radians(s.heading)) > 0
                    gate = find_gate(d.red_blobs, d.black_blobs, cam, r.depth, r.pitch)
                    for p in gate.poles:
                        py = cy + (LANE_HALF * 2 if p.course_side == "left" else -LANE_HALF * 2)
                        bearing = _wrap(math.degrees(math.atan2(py - s.y, gx - s.x)) - s.heading)
                        # Identity: the nearest true pole by bearing should be the one named
                        other_y = cy + (-LANE_HALF * 2 if p.course_side == "left" else LANE_HALF * 2)
                        other = _wrap(math.degrees(math.atan2(other_y - s.y, gx - s.x)) - s.heading)
                        ident_n += 1
                        ident_ok += abs(_wrap(p.estimate.bearing_deg - bearing)) < abs(_wrap(p.estimate.bearing_deg - other))
                        if p.estimate.range_m is not None:
                            true_r = math.hypot(gx - s.x, py - s.y)
                            err["pole_range"].append((p.estimate.range_m - true_r) / true_r)
                    if gate.geometry is not None:
                        turn, off = truth_lines(s, gx, cy, fwd)
                        err["gate_turn"].append(_wrap(gate.geometry.approach_deg - turn))
                        err["gate_offset"].append(gate.geometry.offset_m - off)
                        sm = smooth["gate"] = smooth.get("gate", gate.geometry.offset_m) + k * (gate.geometry.offset_m - smooth.get("gate", gate.geometry.offset_m))
                        err["gate_offset_smoothed"].append(sm - off)
                        for side, g in gate.passages.items():
                            if g is None:
                                continue
                            lane_y = cy + (LANE_HALF if (side == "left") == fwd else -LANE_HALF)
                            err["passage_offset"].append(g.offset_m - truth_lines(s, gx, lane_y, fwd)[1])
                    sets = find_slalom_sets(d.red_blobs, d.white_blobs, cam, r.depth, r.pitch, exclude=gate.used_reds)
                    if sets:
                        near = sets[0]
                        # The true set nearest the estimated one's position
                        b = math.radians(s.heading + near.red.estimate.bearing_deg)
                        ex, ey = s.x + near.range_m * math.cos(b), s.y + near.range_m * math.sin(b)
                        sx, sy = min(info_c["slalom_centers"], key=lambda c: math.hypot(c[0] - ex, c[1] - ey))
                        true_r = math.hypot(sx - s.x, sy - s.y)
                        if abs(ex - sx) > 1.5:
                            sets_wrong += 1
                        else:
                            err["set_range"].append((near.range_m - true_r) / true_r)
                            for side, lane in near.lanes.items():
                                lane_y = sy + (LANE_HALF if (side == "left") == fwd else -LANE_HALF)
                                turn, off = truth_lines(s, sx, lane_y, fwd)
                                err["lane_offset"].append(lane.offset_m - off)
                                key = (round(sx), side)
                                sm = smooth[key] = smooth.get(key, lane.offset_m) + k * (lane.offset_m - smooth.get(key, lane.offset_m))
                                err["lane_offset_smoothed"].append(sm - off)
                                err["lane_turn"].append(_wrap(lane.approach_deg - turn))
                if term or trunc:
                    break
    print(f"{args.episodes} scripted runs, {args.detector} detector")
    print("Gate:")
    print(f"  poles identified as the right physical pole: {ident_ok}/{ident_n} ({100 * ident_ok / max(1, ident_n):.1f}%)")
    summarize("pole range error (fraction of range)", err["pole_range"])
    summarize("gate turn error", err["gate_turn"], " deg")
    summarize("gate offset error", err["gate_offset"], " m")
    summarize("passage offset error", err["passage_offset"], " m")
    summarize("gate offset error, smoothed over 1 s", err["gate_offset_smoothed"], " m")
    print("Slalom (nearest set):")
    print(f"  not near a real set: {sets_wrong} steps")
    summarize("set range error (fraction of range)", err["set_range"])
    summarize("lane offset error", err["lane_offset"], " m")
    summarize("lane turn error", err["lane_turn"], " deg")
    summarize("lane offset error, smoothed over 1 s", err["lane_offset_smoothed"], " m")


if __name__ == "__main__":
    main()
