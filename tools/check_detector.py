#!/usr/bin/env python3
"""
Checks the simulated detector against OpenCV, and the bearing/range estimates
against ground truth.

Runs the scripted mission without a window (SDL dummy video driver) at a fixed
time step, using OpenCV on the rendered camera image to fly. Every frame it also
computes the noise-free simulated boxes from the same pose and compares them to
the boxes OpenCV found.

Usage: .venv/bin/python tools/check_detector.py
"""
import math
import os
import statistics
import sys
from collections import defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.submarine import Submarine           # noqa: E402
from detector_model import DetectorModel     # noqa: E402
from main import build_mission               # noqa: E402
from perception import estimate_pole, estimate_gate, bearing_deg  # noqa: E402
from simulator import SubmarineSimulator     # noqa: E402
from world import SubmarinePhysicsState      # noqa: E402

DT = 1 / 60
MAX_SIM_SECONDS = 400


def iou(a, b):
    ix = max(0, min(a['max_x'], b['max_x']) - max(a['min_x'], b['min_x']))
    iy = max(0, min(a['max_y'], b['max_y']) - max(a['min_y'], b['min_y']))
    inter = ix * iy
    union = a['width'] * a['height'] + b['width'] * b['height'] - inter
    return inter / union if union > 0 else 0.0


def true_bearing_range(state, x, y):
    dx, dy = x - state.x, y - state.y
    b = math.degrees(math.atan2(dy, dx)) - state.heading
    b = (b + 180) % 360 - 180
    return b, math.hypot(dx, dy)


def summarize(name, values):
    if not values:
        print(f"  {name}: no samples")
        return
    a = [abs(v) for v in values]
    a.sort()
    p95 = a[min(len(a) - 1, int(0.95 * len(a)))]
    print(f"  {name}: n={len(values)}  median |err|={statistics.median(a):.3f}  95th pct={p95:.3f}  mean={statistics.mean(values):+.3f}")


def main():
    sim = SubmarineSimulator(Submarine(mission_plan=build_mission()), detector_mode="opencv")
    core = sim.core
    cam, cfg = core.camera, core.config
    truth = DetectorModel(cam, core.gate, core.marker, core.pole_bottom_z,
                          cfg.gatePoleRadius, noisy=False)
    pole_info = {p.name: p for p in truth.poles}

    box_err = defaultdict(list)        # (pole, field) -> pixel errors, OpenCV - simulated
    matched, missed_by_cv = defaultdict(int), defaultdict(int)
    extra_cv = defaultdict(int)
    range_err = defaultdict(list)      # (source, method) -> relative range error
    bearing_err = defaultdict(list)    # source -> degrees
    gate_range_err, gate_bearing_err = defaultdict(list), defaultdict(list)

    t, frames = 0.0, 0
    while t < MAX_SIM_SECONDS and sim.submarineAI.current_task_index < len(sim.submarineAI.mission_plan):
        s = core.state
        state = SubmarinePhysicsState(**vars(s))
        sim.step(DT)
        t += DT
        frames += 1
        vision = sim.submarineAI.vision
        cv_blobs = {'red': list(vision.red_blobs), 'green': list(vision.green_blobs)}

        per_pole_est = {}
        for pole in truth.poles:
            box = truth.project_pole(state, pole)
            if box is None or box['area'] < truth.min_area[pole.color]:
                continue
            candidates = cv_blobs[pole.color]
            best = max(candidates, key=lambda c: iou(box, c), default=None)
            if best is None or iou(box, best) < 0.1:
                missed_by_cv[pole.name] += 1
                continue
            candidates.remove(best)
            matched[pole.name] += 1
            for field in ('center_x', 'center_y', 'width', 'height'):
                box_err[(pole.name, field)].append(best[field] - box[field])

            tb, tr = true_bearing_range(state, pole.x, pole.y)
            for source, b in (("simulated", box), ("opencv", best)):
                est = estimate_pole(b, cam, diameter_m=2 * pole.radius,
                                    height_m=pole.z_bottom - pole.z_top,
                                    bottom_depth_m=pole.z_bottom, camera_depth_m=state.z,
                                    pitch_deg=state.pitch)
                bearing_err[source].append(est.bearing_deg - tb)
                if est.range_m is not None:
                    range_err[(source, est.method)].append((est.range_m - tr) / tr)
                per_pole_est[(source, pole.name)] = (b, est)

        for source in ("simulated", "opencv"):
            l = per_pole_est.get((source, "gate left pole"))
            r = per_pole_est.get((source, "gate right pole"))
            if l and r:
                # Left pole in the world (lower y) appears on the right of the image when facing +x.
                (lb, le), (rb, re) = sorted([l, r], key=lambda p: p[0]['center_x'])
                g = estimate_gate(lb, rb, cam, core.gate.width, le, re)
                tb, tr = true_bearing_range(state, core.gate.x, core.gate.center_y)
                gate_bearing_err[source].append(g.bearing_deg - tb)
                if g.range_m is not None:
                    gate_range_err[(source, g.method)].append((g.range_m - tr) / tr)

        for color, rest in cv_blobs.items():
            extra_cv[color] += len(rest)

    done = sim.submarineAI.current_task_index >= len(sim.submarineAI.mission_plan)
    print(f"Mission {'completed' if done else 'DID NOT complete'} in {t:.1f} s sim time ({frames} frames). "
          f"Gate passes: {core.gate_passes}, collisions: {core.collision_count}\n")

    print("Box agreement, OpenCV minus simulated (pixels):")
    for name in pole_info:
        total = matched[name] + missed_by_cv[name]
        print(f" {name}: in view {total} frames, OpenCV matched {matched[name]}, missed {missed_by_cv[name]}")
        for field in ('center_x', 'center_y', 'width', 'height'):
            summarize(field, box_err[(name, field)])
    print(f" OpenCV boxes with no simulated match: {dict(extra_cv)}\n")

    print("Pole bearing error (deg):")
    for source in ("simulated", "opencv"):
        summarize(source, bearing_err[source])
    print("\nPole range error (fraction of true range):")
    for key in sorted(range_err):
        summarize(f"{key[0]} / {key[1]}", range_err[key])
    print("\nGate bearing error (deg):")
    for source in ("simulated", "opencv"):
        summarize(source, gate_bearing_err[source])
    print("\nGate range error (fraction of true range):")
    for key in sorted(gate_range_err):
        summarize(f"{key[0]} / {key[1]}", gate_range_err[key])


if __name__ == "__main__":
    main()
