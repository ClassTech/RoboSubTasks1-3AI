# RoboSubTasks1-3AI

Learned skills, trained in simulation, for the first three tasks of the RoboSub course. From a random orientation in front of the gate, the sub goes through one side of the divided gate, does a victory dance, slaloms through three sets of poles on the same side, then turns around and comes back the same way to the start.

The design, results and lessons are in [docs/design.html](docs/design.html) (open it in a browser). The step-by-step log is in [PLAN.md](PLAN.md). This repo was copied from [RoboSubQualAI](https://github.com/ClassTech/RoboSubQualAI), the qualification run, on 2026-10-22.

## What it does

- **Simulates the pool and the sub realistically.** The simulation covers physics at 60 Hz, the sub's real size and thruster layout, and its measured thrust (0.8 kgf, 7.85 N, per thruster). It includes the drag of its open frame, thrusters that ramp from zero to full in 0.5 s, and roll from the side vertical thrusters. Sensors (a 1280×960 camera, pressure, IMU with magnetometer) have noise and per-run errors. The course follows ClassTech's RoboSim.
- **Flies the course two ways:**
  - **Scripted:** RoboSim's hand-written mission, adapted to this sub. It is the baseline and has no dance.
  - **Learned:** four small neural-network skills, *gate*, *dance*, *slalom* (all three sets) and *turnaround*. A runner just names them in order. Each skill's small learned *done detector* says when it's finished.
- **Trains the skills.** The gate, slalom and turnaround learn by PPO (Stable-Baselines3), with 9 simulators per run. Training varies the sub's gains, inertia, drag and thrust between runs, so the policies don't depend on exact numbers. The dance learned by imitating a hand controller, after PPO alone couldn't learn it.

The policies never drive the thrusters directly. Ten times a second they pick targets (turn, depth, and forward thrust; the dance picks roll instead of forward thrust). The sub's own control loops fly those targets.

## Where it stands

With the noisy detector at the real thrust, on 300 test starts, without the dance:

| Full mission | Success | Typical time |
|---|---|---|
| Scripted mission | 281/300 (93.7%) | ~134 s |
| **Learned skills + done detectors** | **289/300 (96.3%)** | **~119 s** |

With the victory dance after the gate, on another 300 starts, the learned mission succeeds 287 times, against 286 without the dance. That takes about 224 s, and 234 of the dances (78%) score all 16 stops. The dance's points are extra, so a short dance doesn't fail the run.

Still to do: measure the real sub's roll and speed (the sim's values are estimates), and run the policies on the sub's Raspberry Pi.

## Setup

PyTorch's last build for Intel Macs (2.2.2) supports Python up to 3.12, so the project uses Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Things to run

```bash
# Watch the learned mission (SPACE pauses, Q quits)
.venv/bin/python watch.py tasks_mission --task tasks_mission --seed 10001 --speed 2

# The scripted mission in the pygame window
.venv/bin/python main.py --course tasks --detector noisy

# Watch one skill (--task tasks_gate | tasks_slalom | tasks_turnaround | tasks_dance)
.venv/bin/python watch.py runs/tdance_bc8/best_model.zip --task tasks_dance

# Score the learned mission (--no-dance leaves the dance out; --randomize varies the dynamics)
.venv/bin/python tools/score_tasks_mission.py --episodes 300 --seed 10000

# Train or fine-tune a skill
.venv/bin/python train.py --name my_gate --task tasks_gate --detector noisy --randomize --target-kl 0.03 --envs 9
.venv/bin/python train.py --name my_gate_ft --task tasks_gate --detector noisy --randomize --resume runs/my_gate/best_model.zip --learning-rate 1e-4

# Teach the dance by imitation
.venv/bin/python tools/imitate_dance.py --name my_dance --rounds 8

# Follow training at http://localhost:6006
.venv/bin/tensorboard --logdir runs
```

`runs/` (trained models) is not in git. In a fresh clone, train the skills before `watch.py` or the scorers have a model to load. The scripted mission works without one.

## Where things are

| Area | Files |
|---|---|
| Settings and course | `config.py` (pool, course, sub properties, noise ranges), `course.py` |
| Simulation | `sim_core.py` (headless physics, sensors, detector), `collision.py`, `sensor_model.py`, `camera_geometry.py`, `detector_model.py` |
| Runs on the sub (numpy only) | `tasks_perception.py`, `tasks_observation.py`, `controller.py`, `done_detector.py` |
| Skills and training | `sub_env.py`, `tasks_gate_env.py`, `tasks_slalom_env.py`, `tasks_turnaround_env.py`, `tasks_dance_env.py`, `envs.py`, `train.py`, `tools/imitate_dance.py` |
| Missions | `tasks_mission.py` (scripted and learned, and the judging), `ai/tasks/course_tasks.py`, `main.py`, `simulator.py` |
| Viewing | `sim_view.py`, `watch.py` |
| Tools | `tools/score_tasks_mission.py`, `eval_gate.py`, `train_done_detector.py`, `score_gate_starts.py`, `expand_slalom_obs.py`, `check_tasks_perception.py` |

The qualification course and its skills are still here (`--course prequal`, `gate_env.py`, `orbit_env.py`, `mission.py`).

**Conventions:** z is depth (positive down). Heading and bearing are in degrees, with positive meaning left. Roll is positive with the right side down. Training uses seeds 0 and up; scored tests use 10000 and up.

## History

This repository was split from `QualSimSub` (kept as is), which is based on [ClassTech/RoboSim](https://github.com/ClassTech/RoboSim). The scripted task framework, the rendering and the original physics come from there.
