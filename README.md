# RoboSubTasks1-3AI

Learned skills, trained in simulation, for the first three tasks of the RoboSub course: from a random orientation in front of the gate, through one side of the divided gate, the slalom through three sets of poles on the same side, and back the same way to the start.

This repo starts from [RoboSubQualAI](https://github.com/ClassTech/RoboSubQualAI), the qualification run (gate, around a marker, back through the gate), which it was copied from on 2026-10-22. The plan for the new tasks is in [PLAN.md](PLAN.md). Until the new course is built, the code below still describes the qualification run.

## What it does

- **Simulates the pool and the sub realistically.** The simulation covers physics at 60 Hz, the sub's real size and thruster layout, its measured thrust (0.8 lbf per thruster) and the drag of its open frame, and thrusters that ramp from zero to full in 0.5 s. Sensors (a 1280×960 camera, pressure, IMU with magnetometer) have noise and per-run errors. Gate poles and the marker are drawn at their real size.
- **Flies the course two ways:**
  - **Scripted:** the original hand-written mission, kept as the baseline.
  - **Learned:** two small neural-network skills, *gate* and *orbit*. Each skill has a small learned *done detector* that says when it's finished; the runner just runs them in order.
- **Trains the skills with PPO** (Stable-Baselines3), using 16 simulators in parallel at about 290× real time. Training varies the sub's gains, inertia, drag and thrust between runs so the policies don't depend on exact numbers.

The policies never drive the thrusters directly. Ten times a second they pick targets (turn, depth, forward thrust, and sideways thrust for the orbit). The sub's PID loops fly those targets.

## Where it stands

On fresh test starts, with the noisy detector:

| Full mission | Success, nominal | Success, varied dynamics | Typical time |
|---|---|---|---|
| Scripted mission | 57/60 | 56/60 | ~151 s |
| **Learned skills + done detectors** | **59/60** | **59/60** | **~97 s** |

The learned mission is a little more reliable than the scripted one and about a third faster, with no hand-written switching rules: each skill's detector decides when to move on.

Running the policies on the sub's Raspberry Pi is planned but deferred.

## Setup

PyTorch's last build for Intel Macs (2.2.2) supports Python up to 3.12, so the project uses Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Things to run

```bash
# The scripted mission in the pygame window
.venv/bin/python main.py                        # camera image + OpenCV, like the real sub
.venv/bin/python main.py --detector noisy       # simulated detector with realistic noise

# Watch the learned mission, the scripted one, or a single skill
.venv/bin/python watch.py mission --task mission
.venv/bin/python watch.py scripted --task mission
.venv/bin/python watch.py runs/gate_phys6_ft/final_model.zip --task gate
.venv/bin/python watch.py runs/orbit_straight_in2/final_model.zip --task orbit

# Train a skill (--task gate | orbit; --randomize varies the dynamics; --resume fine-tunes)
.venv/bin/python train.py --name my_gate --task gate --detector noisy --wide-starts --randomize --steps 2000000

# Follow training at http://localhost:6006
.venv/bin/tensorboard --logdir runs

# Score a controller on fixed test starts (--task gate | gate_return | orbit | mission)
.venv/bin/python tools/eval_gate.py mission --task mission --episodes 60 --seed 30000
```

`runs/` (trained models) is not in git. In a fresh clone, train a skill before `watch.py` or `eval_gate.py` have a model to load. The `scripted`, `straight` and `random` controllers work without one.

## Where things are

| Area | Files |
|---|---|
| Settings | `config.py` (pool, course, sub properties, noise ranges) |
| Simulation | `sim_core.py` (headless physics, sensors, detector), `collision.py`, `sensor_model.py`, `camera_geometry.py`, `detector_model.py`, `world.py` |
| Runs on the sub (numpy only) | `perception.py`, `controller.py`, `observation.py`, `done_detector.py` |
| RL | `sub_env.py`, `gate_env.py`, `orbit_env.py`, `mission.py`, `envs.py`, `train.py`, `gate_controllers.py` |
| Scripted mission | `main.py`, `simulator.py`, `ai/` |
| Viewing | `sim_view.py`, `watch.py` |
| Tools | `tools/eval_gate.py`, `train_done_detector.py`, `run_headless.py`, `check_detector.py`, `check_observation.py`, `record_handoffs.py` |

**Conventions:** z is depth (positive down). Heading and bearing are in degrees, with positive meaning left. Training uses seeds 0 and up; scored tests use 10000 and up.

## History

This repository was split from `QualSimSub` (kept as is), which is based on [ClassTech/RoboSim](https://github.com/ClassTech/RoboSim). The scripted task framework, the rendering and the original physics come from there.
