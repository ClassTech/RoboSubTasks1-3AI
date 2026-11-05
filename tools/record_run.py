#!/usr/bin/env python3
"""
Records one episode of watch.py's view to an MP4, without a window (needs
ffmpeg on the PATH). The controllers are watch.py's, plus tasks_mission_nodance
for the learned mission without the dance.

Usage:
  .venv/bin/python tools/record_run.py scripted --task tasks_mission --seed 10001 --out site/scripted.mp4
  .venv/bin/python tools/record_run.py tasks_mission --task tasks_mission --seed 10001 --out site/learned.mp4
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

import envs
import watch
from tasks_mission import TasksMissionRunner


class Recorder(watch.Viewer):
    def __init__(self, args):
        if args.controller == "tasks_mission_nodance":
            args.controller = "tasks_mission"
            super().__init__(args)
            self.controller = TasksMissionRunner(dance_path=None)
        else:
            super().__init__(args)
        self.title = args.title or self.title
        self.episode, self.seed = 0, args.seed
        w, h = self.view.width, self.view.height
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(args.fps), "-i", "-", "-vf", f"scale={args.width}:-2", "-c:v", "libx264",
             "-crf", str(args.crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
            stdin=subprocess.PIPE)

    def _draw(self, thrusters):
        env = self.env
        self.view.draw_camera(env.core)
        red = env.detections.red_blobs if env.detections else []
        self.view.draw_boxes(red, watch.find_gate_pair(red, env.core.camera.height))
        self.view.render(env.core, self._lines(), thrusters, title=self.title, trail=self.trail, controls=())

    def _write_frame(self):
        self.ffmpeg.stdin.write(pygame.image.tobytes(self.view.screen, "RGB"))

    def _on_physics_step(self, env, thrusters):
        self.trail.append((env.core.state.x, env.core.state.y))
        self.thrusters = thrusters
        self.frame += 1
        if self.frame % self.args.render_every == 0:
            self._draw(thrusters)
            self._write_frame()
        pygame.event.pump()

    def record(self):
        env = self.env
        obs, _ = env.reset(seed=self.seed)
        self.controller.reset(env)
        self.trail, self.step_reward, self.total_reward = [], 0.0, 0.0
        while True:
            obs, reward, term, trunc, info = self.controller.step(env, obs)
            self.step_reward, self.total_reward = reward, self.total_reward + reward
            if term or trunc:
                break
        outcome = next(k for k in self.outcome_names if info[k])
        self.tally[outcome] += 1
        self.last_outcome = f"{self.outcome_names[outcome]} after {info['time_s']:.1f} s"
        print(f"seed {self.seed}: {self.last_outcome}, dance stops {info.get('dance_stops', '-')}")
        self._draw(self.thrusters)
        for _ in range(int(self.args.fps * 3)):   # Hold the end for 3 s of video
            self._write_frame()
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("controller", help="as for watch.py, or tasks_mission_nodance")
    parser.add_argument("--task", choices=sorted(envs.ENVS), default="tasks_mission")
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="noisy")
    parser.add_argument("--seed", type=int, default=10_001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--render-every", type=int, default=4, help="Physics steps (60 Hz) per video frame")
    parser.add_argument("--fps", type=float, default=30, help="Video frame rate; with --render-every 4, 30 is 2x real time")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--crf", type=int, default=28)
    args = parser.parse_args()
    args.speed, args.episodes = 1.0, 1
    Recorder(args).record()


if __name__ == "__main__":
    main()
