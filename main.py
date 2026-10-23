#!/usr/bin/env python3
"""
Main entry point for the Autonomous Submarine Simulator.
Assembles and starts the simulation.
"""
import argparse

from simulator import SubmarineSimulator
from ai.submarine import Submarine
# --- Absolute imports from ai.tasks package ---
from ai.tasks import GateTask, StabilizeTask, OrbitTurnTask, ShutdownTask
# ---

def build_mission():
    """The scripted qualification mission (also the baseline for the RL policy)."""
    # --- SET MISSION DEPTH ---
    MISSION_DEPTH = 1.5 # Run 1.5m deep
    SURFACE_DEPTH = 0.1 # Just under the surface
    # ---

    # --- Create task instances, passing target_depth ---
    first_gate_task = GateTask(target_depth=MISSION_DEPTH)
    stabilize_after_gate1 = StabilizeTask(duration=2.0, target_depth=MISSION_DEPTH)

    # --- USE NEW ORBIT TASK ---
    # --- REMOVED initial_align_fraction argument ---
    orbit_task = OrbitTurnTask(target_depth=MISSION_DEPTH,
                               approach_height_fraction=0.75,
                               orbit_target_fraction=0.5,
                               orbit_sway_power=-0.8,      # Negative = right (positive sway moves the sub left).
                                                           # Was -0.5; with the real drag that took most of 90 s
                               orbit_yaw_gain=4.5,
                               orbit_target_width_fraction=0.08, # ~2.7 m from the marker, outside the 1.5 m margin
                               final_sway_power=-0.5,
                               final_extra_sway_seconds=3.0)  # So the drive back to the gate clears the marker      
    # ---

    stabilize_after_turn = StabilizeTask(duration=2.0, target_depth=MISSION_DEPTH)
    return_gate_task = GateTask(target_depth=MISSION_DEPTH)

    stabilize_before_surface = StabilizeTask(duration=2.0, target_depth=MISSION_DEPTH)
    surface_task = StabilizeTask(duration=10.0, target_depth=SURFACE_DEPTH)
    shutdown_task = ShutdownTask()
    # ---

    # --- Set search direction for the return GateTask ---
    return_gate_task.search_direction = -1
    # ---

    # --- Define mission plan using the instances ---
    mission = [
        first_gate_task,
        stabilize_after_gate1,
        orbit_task,
        stabilize_after_turn,
        return_gate_task,
        stabilize_before_surface,
        surface_task,
        shutdown_task
    ]
    return mission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the scripted mission in the simulator.")
    parser.add_argument("--detector", choices=["opencv", "perfect", "noisy"], default="opencv",
                        help="opencv: run OpenCV on the rendered camera image (default). "
                             "perfect/noisy: use the simulated detector, without or with noise.")
    parser.add_argument("--course", choices=["prequal", "tasks"], default="prequal",
                        help="prequal: the qualification run (default). tasks: RoboSub tasks 1-3 "
                             "(divided gate and slalom, out and back), scripted from ClassTech/RoboSim")
    args = parser.parse_args()

    if args.course == "tasks":
        from tasks_mission import build_tasks_mission
        mission = build_tasks_mission()
    else:
        mission = build_mission()
    submarine_ai = Submarine(mission_plan=mission)
    sim = SubmarineSimulator(submarine_ai=submarine_ai, detector_mode=args.detector, course=args.course)
    sim.run()