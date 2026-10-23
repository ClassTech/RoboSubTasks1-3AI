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
- [ ] Tell apart the gate's poles and divider from the slalom poles; estimate the chosen gate opening's center and the chosen slalom lane.
- [ ] Observation for each new skill, built only from what the sub measures.

### Phase 3: Scripted baseline
- [x] Ported RoboSim's mission and tasks: `tasks_mission.build_tasks_mission()` (gate, stabilize, slalom, stabilize, slalom reversed, gate, hover 10 s, surface; no dance) and `ai/tasks/course_tasks.py`. Run it with `.venv/bin/python main.py --course tasks`.
- [x] Adapted to this sub (details in the module's docstring): this repo's helpers and mixer, no pitch or roll loops, a fixed 1.3 m depth, timings for 0.34 m/s, the slalom on the gate's side, the reversed slalom waiting until it has turned around.
  - Fixes found by testing: approaches square up, slide sideways onto the passage or lane, then drive straight in correcting with sway (RoboSim turned toward the target, came in at an angle and hit poles once they left the view); losing a target only counts as passing it if it was close; the slalom finishes after 3 s with no set in view; a set counts as passed when the next one takes its place (the lane narrows by half for 3 frames); the SIM ONLY damping turns the true velocity by the true heading (by the measured one, off up to 15 deg, it read forward speed as sideways).
- [x] `TasksMissionEnv` judges a run on ground truth: the gate, the three sets out and back, and the gate again, all in one lane. **Scripted, noisy detector: 57/60, 57/60 varied, 56/60 on 10000+ (170/180, 94%), ~200 s.** Failures: scattered pole collisions and a few wrong-course runs.
  - The slalom at full thrust instead of 60% (user: run the script at full thrust where appropriate): **58/60, 58/60 varied, 57/60 on 10000+ (173/180, 96%), ~180 s** (was ~200 s). Most of the remaining time is the sim's top speed, 0.34 m/s from the measured thrust and the estimated drag; the real sub's speed hasn't been measured yet, and it would calibrate the drag.

### Phase 4: Learned skills
- [ ] Gate through a chosen side, from a random orientation (search included).
- [ ] Slalom: one set on a chosen side, chained three times (or one skill for all three).
- [ ] Turn-around and the return legs.
- [ ] A done detector for each; the runner stays a plain sequence.

### Phase 5: Full run
- [ ] Score the full out-and-back run against the scripted baseline; watch it.

## Lessons carried over (see RoboSubQualAI's docs/design.html)
- Make the big payoff the actual goal; check every reward term for loopholes ("you get what you ask for").
- Watch runs, and when behavior is odd, ask what the policy is paid for and what it actually perceives.
- Confirm detections over several frames; don't trust ranges the sensors can't support.
- Watch for outputs pinned past their +-1 limit.
- Keep the runner free of hand-written rules; fix problems in training or perception.
