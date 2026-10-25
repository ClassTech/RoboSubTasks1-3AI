#!/usr/bin/env python3
"""
Watch a controller fly a skill environment (--task gate or orbit) in the pygame
window: the same episodes, starts and noise that training and
tools/eval_gate.py use.

The panel shows the controller's action, the reward, and what the policy
believes about the gate (and the marker, for the orbit) next to the truth.

Usage:
  .venv/bin/python watch.py scripted
  .venv/bin/python watch.py straight --detector perfect
  .venv/bin/python watch.py runs/gate/best_model.zip --speed 2
  .venv/bin/python watch.py scripted --task orbit

Keys: SPACE pause, N single physics step while paused, R restart the episode,
ENTER next episode, Q or ESC quit.
"""
import argparse
import math

import pygame

import envs
from gate_controllers import ScriptedController, make_controller
from perception import find_gate_pair
from sim_view import SimView
from sub_env import DEPTH_MAX, DEPTH_MIN, HEADING_CHANGE_DEG


class _Restart(Exception): pass
class _Next(Exception): pass
class _Quit(Exception): pass


class Viewer:
    def __init__(self, args):
        self.args = args
        self.env = envs.make(args.task, args.detector)
        self.title = f"{args.task.capitalize()} skill viewer"
        self.view = SimView(self.env.core.config, title=self.title)
        self.controller = make_controller(args.controller, action_dim=self.env.ACTION_DIM)
        self.env.on_physics_step = self._on_physics_step
        self.paused = False
        self.frame = 0
        self.outcome_names = {k: "SUCCESS" if k == "is_success" else k.replace("_", " ").upper()
                              for k in self.env.OUTCOMES}
        self.tally = dict.fromkeys(self.outcome_names, 0)
        self.last_outcome = ""

    # --- Input ---
    def _handle_events(self) -> bool:
        """Processes keys. Returns True if a single step was requested while paused."""
        step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise _Quit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE): raise _Quit()
                if event.key == pygame.K_SPACE: self.paused = not self.paused
                elif event.key == pygame.K_n: step = True
                elif event.key == pygame.K_r: raise _Restart()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER): raise _Next()
        return step

    # --- Drawing ---
    def _lines(self):
        env, a = self.env, self.args
        r, s, g = env.readings, env.core.state, env.core.gate
        lines = [f"Controller: {self.controller.name}   Detector: {a.detector}",
                 f"Episode {self.episode + 1}, seed {self.seed}   Time {env.time:4.1f} / {env.max_seconds:.0f} s",
                 f"Reward: last step {self.step_reward:+.2f}, total {self.total_reward:+.2f}"]
        if hasattr(self.controller, "state_name"):
            label = "Scripted state" if isinstance(self.controller, ScriptedController) else "Phase"
            lines.append(f"{label}: {self.controller.state_name()}")
        act = self.controller.last_action
        if act is not None:
            depth = DEPTH_MIN + (act[1] + 1) / 2 * (DEPTH_MAX - DEPTH_MIN)
            sway = f", sway {act[3]:+.2f}" if len(act) > 3 else ""
            lines.append(f"Action: turn {HEADING_CHANGE_DEG * act[0]:+5.1f} deg, depth {depth:.2f} m, surge {act[2]:+.2f}{sway}")
        rel_heading = (r.heading - env.builder.start_heading + 180) % 360 - 180 if env.builder.start_heading is not None else 0.0
        lines += [f"Sub: depth {r.depth:.2f} m, heading {rel_heading:+.0f} deg from start, pitch {r.pitch:+.1f} deg",
                  ""]
        if env.core.course_kind == "tasks":
            return lines + self._tasks_lines()
        b = env.builder
        facing_plus_x = s.x < g.x
        side = 1 if facing_plus_x else -1
        dx, dy = g.x - s.x, g.center_y - s.y
        true_bearing = (math.degrees(math.atan2(dy, dx)) - s.heading + 180) % 360 - 180
        true_approach = ((0.0 if facing_plus_x else 180.0) - s.heading + 180) % 360 - 180
        true_offset = side * (s.y - g.center_y)
        poles = "  ".join(f"{name} pole: {'in view' if t.found else ('remembered' if t.seen_ever else 'not seen')}"
                          for name, t in (("Left", b.left), ("Right", b.right)))
        lines.append(poles)
        if not b.gate.seen_ever:
            lines.append("Gate: not seen yet")
        else:
            seen = "in view" if b.gate.found else f"remembered, last seen {b.gate.unseen:.1f} s ago"
            lines += [f"Gate: {seen}",
                      f"  believes  bearing {b.gate.bearing(r.heading):+6.1f} deg, range {b.gate.range_m:5.2f} m"]
            if b.gate.geometry_known:
                approach = (b.gate.abs_approach - r.heading + 180) % 360 - 180
                lines.append(f"            square-on turn {approach:+6.1f} deg, off center line {b.gate.offset_m:+5.2f} m")
        lines += [f"  truth     bearing {true_bearing:+6.1f} deg, range {math.hypot(dx, dy):5.2f} m",
                  f"            square-on turn {true_approach:+6.1f} deg, off center line {true_offset:+5.2f} m"]
        if "marker" in b.tracks:
            m, mt = env.core.marker, b.marker
            m_bearing = (math.degrees(math.atan2(m.y - s.y, m.x - s.x)) - s.heading + 180) % 360 - 180
            seen = ("in view" if mt.found else f"remembered, {mt.unseen:.1f} s ago") if mt.seen_ever else "not seen yet"
            lines.append(f"Marker: {seen}")
            if mt.seen_ever:
                lines.append(f"  believes  bearing {mt.bearing(r.heading):+6.1f} deg, range {mt.range_m:5.2f} m")
            lines.append(f"  truth     bearing {m_bearing:+6.1f} deg, range {math.hypot(m.x - s.x, m.y - s.y):5.2f} m")
            handoff = f"   handoff ready: {env.handoff_ready()}" if hasattr(env, "handoff_ready") else ""
            lines.append(f"Swept around marker {env.swept:+5.0f} deg{handoff}")
        lines += ["",
                  f"Last episode: {self.last_outcome or '-'}",
                  "Totals: " + ", ".join(f"{self.outcome_names[k].lower()} {v}" for k, v in self.tally.items())]
        return lines

    def _tasks_lines(self):
        """The tasks course: the commanded passage, what the gate skill believes about it, and the truth."""
        env, r, s = self.env, self.env.readings, self.env.core.state
        lines = []
        if hasattr(env, "direction"):
            way = "forward" if env.direction > 0 else "return"
            lines.append(f"Commanded: {'left' if env.side > 0 else 'right'} passage ({way}; the course's {env.lane} lane)")
        b = getattr(env, "builder", None)
        if b is not None and getattr(b, "skill", None) == "gate":
            ours = "left" if env.side > 0 else "right"
            p = b.passages[ours]
            if not p.seen_ever:
                lines.append("Passage: not seen yet")
            else:
                seen = "in view" if p.found else f"remembered, last seen {p.unseen:.1f} s ago"
                lines += [f"Passage: {seen}", f"  believes  bearing {p.bearing(r.heading):+6.1f} deg, range {p.range_m:5.2f} m"]
                if p.geometry_known:
                    turn = (p.abs_approach - r.heading + 180) % 360 - 180
                    lines.append(f"            square-on turn {turn:+6.1f} deg, off center line {p.offset_m:+5.2f} m")
            gx = env.core.course.info["gate_x"]
            dx, dy = gx - s.x, env.lane_y - s.y
            fwd = env.direction > 0
            lines += [f"  truth     bearing {(math.degrees(math.atan2(dy, dx)) - s.heading + 180) % 360 - 180:+6.1f} deg, "
                      f"range {math.hypot(dx, dy):5.2f} m",
                      f"            square-on turn {((0.0 if fwd else 180.0) - s.heading + 180) % 360 - 180:+6.1f} deg, "
                      f"off center line {(s.y - env.lane_y) * (1 if fwd else -1):+5.2f} m"]
        passes = " ".join(f"{'G' if p.line == 'gate' else p.line[-1]}{'>' if p.direction > 0 else '<'}"
                          f"{p.lane[0].upper() if p.lane else 'X'}" for p in env.core.passes) or "-"
        lines += ["", f"Passes: {passes}", "",
                  f"Last episode: {self.last_outcome or '-'}",
                  "Totals: " + ", ".join(f"{self.outcome_names[k].lower()} {v}" for k, v in self.tally.items())]
        return lines

    def _draw(self, thrusters):
        env = self.env
        self.view.draw_camera(env.core)
        red = env.detections.red_blobs if env.detections else []
        self.view.draw_boxes(red, find_gate_pair(red, env.core.camera.height))
        self.view.render(env.core, self._lines(), thrusters, title=self.title, trail=self.trail,
                         controls=("SPACE - Pause", "N - Step (paused)", "R - Restart episode",
                                   "ENTER - Next episode", "Q - Quit"))

    def _on_physics_step(self, env, thrusters):
        self.trail.append((env.core.state.x, env.core.state.y))
        self.thrusters = thrusters
        self.frame += 1
        if self.frame % self.args.render_every == 0 or self.paused:
            self._draw(thrusters)
            self.view.clock.tick(60 * self.args.speed / self.args.render_every)
        stepping = self._handle_events()
        while self.paused and not stepping:
            self._draw(thrusters)
            self.view.clock.tick(30)
            stepping = self._handle_events()

    # --- Episodes ---
    def run_episode(self):
        env = self.env
        obs, _ = env.reset(seed=self.seed)
        self.controller.reset(env)
        self.trail, self.step_reward, self.total_reward = [], 0.0, 0.0
        while True:
            obs, reward, term, trunc, info = self.controller.step(env, obs)
            self.step_reward, self.total_reward = reward, self.total_reward + reward
            if term or trunc:
                outcome = next(k for k in self.outcome_names if info[k])
                self.tally[outcome] += 1
                self.last_outcome = f"{self.outcome_names[outcome]} after {info['time_s']:.1f} s, reward {self.total_reward:+.1f}"
                print(f"Episode {self.episode + 1} (seed {self.seed}): {self.last_outcome}")
                end = pygame.time.get_ticks() + 1500   # Show the end for a moment
                while pygame.time.get_ticks() < end:
                    self._draw(self.thrusters)
                    self._handle_events()
                    self.view.clock.tick(30)
                return

    def run(self):
        self.episode, self.seed = 0, self.args.seed
        try:
            while self.args.episodes == 0 or self.episode < self.args.episodes:
                try:
                    self.run_episode()
                except _Restart:
                    continue
                except _Next:
                    pass
                self.episode += 1
                self.seed += 1
        except _Quit:
            pass
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("controller", help="scripted, straight, random, or a path to a PPO model .zip")
    parser.add_argument("--task", choices=sorted(envs.ENVS), default="gate")
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="noisy")
    parser.add_argument("--seed", type=int, default=10_000, help="First episode's seed (eval seeds start at 10000)")
    parser.add_argument("--episodes", type=int, default=0, help="Stop after this many episodes (0 = keep going)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed; above 1 is faster than real time")
    parser.add_argument("--render-every", type=int, default=2,
                        help="Draw every Nth physics step (the 1280x960 camera is slow to draw every step)")
    Viewer(parser.parse_args()).run()


if __name__ == "__main__":
    main()
