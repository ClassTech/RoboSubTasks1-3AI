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
- [ ] Course geometry in `world.py`/`config.py`: the divided gate and slalom poles; collisions and pass checks (which gate opening, which side of each set's center pole).
- [ ] Rendering and the simulated detector for the new objects: white poles (`WHITE_HSV_RANGE` is kept for this), the gate's black and red parts, the divider.
- [ ] Check the simulated detector against OpenCV on the rendered image, as before.

### Phase 2: Perception and observation
- [ ] Tell apart the gate's poles and divider from the slalom poles; estimate the chosen gate opening's center and the chosen slalom lane.
- [ ] Observation for each new skill, built only from what the sub measures.

### Phase 3: Scripted baseline
- [ ] Port RoboSim's mission definition and scripted tasks, adapted to this simulator: the plan in its `main.py` is GateTask, StabilizeTask (3 s), SlalomTask, StabilizeTask (3 s), SlalomTask (reversed), GateTask, TimedHoverTask (10 s), SurfaceTask (VictoryDanceTask left out for now). Tasks come from its `ai/tasks/` (gate, slalom, stabilize, hover, timed hover, surface) and `ai/vision.py`. The forward slalom records its side so the reversed one uses the same physical lane, and the second GateTask goes back through the same opening.
- [ ] Adapt them to this repo's sub: measured sensing only (no ground-truth helpers in the scripted loops where avoidable), the shared controller (side vertical thrusters, no pitch or roll loop), the real thrust and drag (timings and gains).
- [ ] Score the scripted mission on fixed test starts; it's the baseline for the learned skills.

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
