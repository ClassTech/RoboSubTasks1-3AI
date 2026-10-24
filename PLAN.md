# RoboSubTasks1-3AI Plan

Goal: learned skills, trained in simulation, that fly the first three tasks of the RoboSub course: start in a random orientation in front of the gate, go through one side of the divided gate, run the slalom through three sets of poles on the same side, then do it all in reverse and end where it started.

This repo starts from RoboSubQualAI (copied 2026-10-22, fresh history). Everything that made the qualification run work carries over: the realistic simulator (measured thrust, frame drag, side vertical thrusters, thruster ramp, noisy sensors and detector, varied dynamics), perception and the observation, the controller, PPO training, done detectors, the plain-sequence mission runner, and the tools. The qualification-course code (gate, marker, orbit) is still here until the new course replaces it. Its history and results are in RoboSubQualAI's PLAN.md and docs/design.html.

## Course (from ClassTech/RoboSim, the 2025 tasks 1-2 simulator; where in doubt, RoboSim decides)

| Item | Spec |
|---|---|
| Gate | 3.048 m (10 ft) wide, 1.524 m (5 ft) poles: red top, black body; a 0.610 m divider hanging from the top bar at the center; top bar depth random 0.6-1.8 m |
| Gate side | The sub chooses an opening; the return goes back through the same physical opening |
| Slalom | 3 sets; first set 8 m past the gate; sets 4 m apart; each set offset sideways up to +-0.38 m from the last |
| Slalom set | Three poles 1.524 m apart, 0.9 m tall: white outside, red center. Each set is passed between the red and one white pole |
| Start | 9.144 m (30 ft) before the gate, +-0.914 m sideways, heading 0, 90 or 270 deg, 0.5 m deep |
| Return | Slalom reversed, in the same physical lane (the opposite side relative to the sub), then back through the gate's same opening; hover 10 s, then surface |
| Victory dance | Deferred (RoboSim spins 360 deg in yaw, then rolls, between the gate and the slalom) |

## Phases (to agree on before building)

### Phase 1: Course and sim
- [x] Confirm the course spec above (user: where in doubt, use RoboSim; dance deferred).
- [x] Course geometry: `course.py` describes each course as segments (collisions, the camera and top views, and the simulated detector all build from them) and pass lines with named lanes; `config.TasksConfig` has RoboSim's numbers; `SubSim(course="tasks")`. The qualification course is unchanged (its scripted 57/60 and learned 60/60 scores are identical after the refactor).
  - One departure from RoboSim: the gate's top bar is 0.6-1.1 m deep, not 0.6-1.8 m (with 1.52 m poles in a 2.1 m pool, a 1.8 m bar leaves a 0.3 m opening above the floor).
- [x] Rendering and the simulated detector for the new objects: red, white and black boxes (the gate poles are half red, half black, mirrored; the divider is red; the slalom is white-red-white).
- [x] OpenCV on the rendered image agrees with the simulated detector, apart from a wide, flat dark patch in the background photo and far white poles below the simulated detector's minimum size; neither is pole-shaped enough to matter.

### Phase 2: Perception and observation
- [x] `tasks_perception.py` (numpy only): gate poles are a red and a black box stacked end to end, and the color order says which physical pole (the right one is red over black), from either direction; the divider is the red box between them, level with their tops; ranges from the poles' feet on the floor. Slalom poles range by their known 0.9 m length; a set is a red pole with a white at about the same range (lone reds are the gate's). Passages and lanes are small gates (`gate_geometry` on their two edges): turn to face through and sideways offset.
  - `tools/check_tasks_perception.py`, against ground truth. Perfect detector: every gate pole identified correctly; gate turn ~0.4 deg and offset ~5 cm typical; lane offset ~3 cm. Noisy, per frame: gate offset ~0.7 m typical (it rests on the difference of two ~5% ranges), ~0.17 m smoothed over 1 s, as on the qualification course; lanes are only 1.52 m wide, so their offset stays rough (~0.26 m smoothed, worse at times) while bearings stay precise.
- [x] `tasks_observation.py`: one builder, two skills, each with a side input (+1 left, -1 right, relative to the sub) and 3 actions (turn, depth, surge; sway was dropped, see the gate skill). Gate frame (40): own state, side, left and right poles (tracked by physical identity, shown as the sub sees them), divider, gate (with turn and offset), the commanded passage (with turn and offset), last action. Slalom frame (31): own state, side, the nearest set's red, our white, the other white, our lane (bearing to its center from image positions, its width in degrees, smoothed turn and offset), last action. 4 frames stacked: 160 and 124 values.
  - Noisy: the gate track is confirmed in 54% of approach steps (the rest is mostly the last meters, with one pole in view; pole tracks cover those), gate offset error ~0.13 m typical; the lane bearing ~0.7 deg typical.

### Phase 3: Scripted baseline
- [x] Ported RoboSim's mission and tasks: `tasks_mission.build_tasks_mission()` (gate, stabilize, slalom, stabilize, slalom reversed, gate, hover 10 s, surface; no dance) and `ai/tasks/course_tasks.py`. Run it with `.venv/bin/python main.py --course tasks`.
- [x] Adapted to this sub (details in the module's docstring): this repo's helpers and mixer, no pitch or roll loops, a fixed 1.3 m depth, timings for 0.34 m/s, the slalom on the gate's side, the reversed slalom waiting until it has turned around.
  - Fixes found by testing: approaches square up, slide sideways onto the passage or lane, then drive straight in correcting with sway (RoboSim turned toward the target, came in at an angle and hit poles once they left the view); losing a target only counts as passing it if it was close; the slalom finishes after 3 s with no set in view; a set counts as passed when the next one takes its place (the lane narrows by half for 3 frames); the SIM ONLY damping turns the true velocity by the true heading (by the measured one, off up to 15 deg, it read forward speed as sideways).
- [x] `TasksMissionEnv` judges a run on ground truth: the gate, the three sets out and back, and the gate again, all in one lane. **Scripted, noisy detector: 57/60, 57/60 varied, 56/60 on 10000+ (170/180, 94%), ~200 s.** Failures: scattered pole collisions and a few wrong-course runs.
  - The slalom at full thrust instead of 60% (user: run the script at full thrust where appropriate): **58/60, 58/60 varied, 57/60 on 10000+ (173/180, 96%), ~180 s** (was ~200 s). Most of the remaining time is the sim's top speed, 0.34 m/s from the measured thrust and the estimated drag; the real sub's speed hasn't been measured yet, and it would calibrate the drag.

### Phase 4: Learned skills
- [x] Gate through a chosen side, from a random orientation (search included). `tasks_gate_env.TasksGateEnv` (`--task tasks_gate`): starts 3-10 m before the gate (RoboSim's headings or any) or 2-7 m past it facing back; the side (+1 left, -1 right, relative) drawn per episode. Reward as agreed with the user: progress to the commanded passage's center (with depth, center line and, added later, facing terms), +20 through it, +5 at 1.5 m beyond, -5 for any failure (collision, wrong passage, missed, surfaced, timeout). The mission takes the side the scripted way (the passage nearer the image center at first sight).
  - With sway (4 actions), from scratch, 4M steps: 70% and 50% (two seeds); it went through crabwise, ~52 deg off, and hit the divider it couldn't see (19 of 25 collisions). A facing cost (1/rad over the last 3 m) changed nothing.
  - Without sway (3 actions, as the qualification gate; sliding through would be legal in competition), from scratch: **`tgate3_s0` best: 90/100 nominal, 93/100 varied, 4-7 collisions, ~8 deg off straight at the gate line** (scripted gate task on these starts: 82/100). Seed 1 reached 57-67%.
  - The user's fallback, not needed: a two-step gate (line up with turn and sway by maximizing the gate's apparent width, which puts the sub on its center line using only precise bearings; then drive straight through).
- [x] Slalom: one set on a chosen side, run three times each way. `tasks_slalom_env.TasksSlalomEnv` (`--task tasks_slalom`): starts 1.5-3.5 m before a set, no sway; reward as the gate's (center line 2.0: policies shaved the red pole); backwards (crossing facing more than 90 deg from the way through) fails, after a policy learned to reverse blind through set 3. **`tslalom4_ft` best: 83/100 nominal, 82/100 varied, 13 collisions (mostly the red pole).** The weak link: six sets in a row at 83% is about a third.
- [x] Turnaround, its own skill (user: "Why not?"). Inside the slalom (30-50% of starts) it turned around 8-14% of the time and the rest fell 88% -> 73%: a lane needs its red and a white in the 70 deg view, so from closer than ~2.2 m it's invisible. `tasks_turnaround_env.TasksTurnaroundEnv` (`--task tasks_turnaround`): starts 1-3.5 m past set 3 facing away, memory filled as the forward pass left it; done 3 m back in front of the commanded lane, facing it within 15 deg, red and our white in view; going through the set fails. 2M steps, ~20 min: **`tturn_s1` best: 97/100 nominal, 98/100 varied, ~6.5 s.**
- [x] Thrusters named as on the real sub (user: angled across the diagonals; front-left forward pushes forward-right and turns right). The sim and mixer had them mirrored, consistently, so nothing flew differently (forces match to 1e-15; gate 91/100, was 90).
- [x] Slalom collisions, diagnosed: all 13 came from starts >= 0.5 m off the lane's line (cutting in at 30-57 deg, blind at the end), which the mission doesn't produce; mission-like starts were 22/22. The real gap was the gate's handoff, ~6.5 m before set 1, beyond the 1.5-3.5 m training starts: 31/50. 30% of starts now 5-7.5 m before set 1 going out; fine-tuned from tslalom4_ft (`tslalom6_ft_s0/s1`): **`tslalom6_ft_s1` best: after the gate 47/50, between sets 50/50; training-start mix 79/100 (81 varied)** (s0: 46/50, 50/50, 79/100).
  - Found on the way: resuming from a checkpoint restored its seed, so `--seed` did nothing on fine-tunes (fixed in both repos; earlier "two-seed" fine-tunes were one run twice).
- [ ] A done detector for each; the runner stays a plain sequence.

### Phase 5: Full run
- [ ] Score the full out-and-back run against the scripted baseline; watch it.

## Lessons carried over (see RoboSubQualAI's docs/design.html)
- Make the big payoff the actual goal; check every reward term for loopholes ("you get what you ask for").
- Watch runs, and when behavior is odd, ask what the policy is paid for and what it actually perceives.
- Confirm detections over several frames; don't trust ranges the sensors can't support.
- Watch for outputs pinned past their +-1 limit.
- Keep the runner free of hand-written rules; fix problems in training or perception.
