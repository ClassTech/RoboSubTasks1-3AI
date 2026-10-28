#!/usr/bin/env python3
"""
Trains a PPO policy on a skill environment (--task gate or orbit).

Everything for a run goes in runs/<name>/:
  tb/                          TensorBoard logs (view with: .venv/bin/tensorboard --logdir runs)
  best_model.zip               the checkpoint with the best scored-test result so far
  latest_model.zip             the most recent checkpoint
  *_vecnormalize.pkl           the observation normalization each checkpoint needs
  eval.csv                     every scored test, one row each
  config.json                  this run's settings

Scored tests: every --eval-every steps the current policy flies --eval-episodes
fixed starts (seeds 10000+, never used for training) with no randomness in its
actions, scored like tools/eval_gate.py. The scripted GateTask's result on the
same starts is logged alongside for comparison.

Watch a checkpoint at any time, even while training runs:
  .venv/bin/python watch.py runs/<name>/best_model.zip

Usage: .venv/bin/python train.py --name gate_perfect --task gate --detector perfect --steps 3000000
"""
import argparse
import copy
import csv
import json
import os
import statistics
import time
from collections import deque

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import envs
from gate_controllers import make_controller, policy_controller, run_episode, vecnormalize_path
from gate_env import RewardConfig, StartConfig

EVAL_SEED = 10_000


def env_kwargs(task: str, args, scoring: bool = False) -> dict:
    """Environment settings for training, or for scoring (always the real-world margin rule)."""
    randomize = {"randomize": True} if args.randomize and not scoring else {}
    recorded = {}
    if args.mission_starts and not scoring:
        with open(args.mission_starts) as f:
            handoffs = json.load(f)
        recorded = ({"mission_starts": handoffs["ORBIT"]} if task == "orbit"
                    else {"mission_return_starts": handoffs["RETURN"]} if task == "gate" else {})
    if task in ("gate", "gate_return"):
        kwargs = {"reward": RewardConfig(center_line_weight=args.center_line_weight), **randomize, **recorded}
        if task == "gate" and not scoring and (args.wide_starts or args.search_starts):
            start = {"search_share": args.search_starts}
            if args.wide_starts:
                # Covers both the forward pass and the return from the orbit's handoff
                start.update(distance_m=(3.0, 11.0), lateral_m=(-3.5, 3.5), depth_m=(0.1, 1.9))
            kwargs["start"] = StartConfig(**start)
        return kwargs
    if task == "orbit":
        return {"margin_rule": "soft" if scoring else args.margin_rule, "practice_starts": not scoring,
                **randomize, **recorded}
    if task == "tasks_turnaround" and not scoring and args.scramble_starts:
        from tasks_turnaround_env import TurnaroundStarts
        return {"starts": TurnaroundStarts(scramble_share=args.scramble_starts), **randomize}
    if task == "tasks_slalom" and not scoring and (args.back_lead_ins or args.gate_lead_ins):
        from tasks_slalom_env import TasksSlalomStarts
        return {"starts": TasksSlalomStarts(back_lead_in_share=args.back_lead_ins,
                                            gate_lead_in_share=args.gate_lead_ins), **randomize}
    return randomize


def make_env(task: str, detector: str, seed: int, kwargs: dict):
    def _init():
        env = envs.make(task, detector, **kwargs)
        env = Monitor(env, info_keywords=env.OUTCOMES + ("time_s", "energy") + env.EXTRA_INFO)
        env.reset(seed=seed)
        return env
    return _init


UNITS = {"_s": "s", "_m": "m", "_deg": "deg"}


def format_extra(key: str, value: float) -> str:
    """Extra per-episode numbers: names ending in _s, _m or _deg carry that unit; others are fractions."""
    for suffix, unit in UNITS.items():
        if key.endswith(suffix):
            return f"{key[:-len(suffix)].replace('_', ' ')} {value:.2f} {unit}"
    return f"{key.replace('_', ' ')} {100 * value:.0f}%"


def time_key(env) -> str:
    """Name of the median-time result (kept as time_to_gate for the gate skill, for old runs)."""
    return "time_to_gate" if isinstance(env, envs.GateEnv) else "time_to_success"


def score(env, controller, episodes: int) -> dict:
    """Flies the fixed evaluation starts and summarizes the outcomes."""
    counts = dict.fromkeys(env.OUTCOMES, 0)
    times, rewards = [], []
    extra = {k: [] for k in env.EXTRA_INFO}
    for i in range(episodes):
        info, total = run_episode(env, controller, EVAL_SEED + i)
        for k in extra:
            extra[k].append(info[k])
        for k in env.OUTCOMES:
            counts[k] += bool(info[k])
        if info["is_success"]:
            times.append(info["time_s"])
        rewards.append(total)
    result = {"success_rate": counts["is_success"] / episodes}
    result.update({f"{k}_rate": counts[k] / episodes for k in env.OUTCOMES if k != "is_success"})
    result[time_key(env)] = statistics.median(times) if times else float("nan")
    result["mean_reward"] = statistics.mean(rewards)
    result.update({k: statistics.mean(v) for k, v in extra.items()})
    return result


class OutcomeLogger(BaseCallback):
    """Logs how training episodes end (each failure kind, time to success, energy) to TensorBoard."""
    def __init__(self, outcomes, time_name: str, extra=(), window: int = 200):
        super().__init__()
        self.outcomes = [k for k in outcomes if k != "is_success"]
        self.time_name = time_name
        self.extra = extra
        self.recent = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "collision" in info:
                self.recent.append(info)
        return True

    def _on_rollout_end(self):
        if not self.recent:
            return
        n = len(self.recent)
        for key in self.outcomes:
            self.logger.record(f"outcomes/{key}_rate", sum(bool(i[key]) for i in self.recent) / n)
        times = [i["time_s"] for i in self.recent if i["is_success"]]
        if times:
            self.logger.record(f"outcomes/{self.time_name}", statistics.median(times))
        self.logger.record("outcomes/energy", statistics.mean(i["energy"] for i in self.recent))
        for key in self.extra:
            self.logger.record(f"outcomes/{key}", statistics.mean(i[key] for i in self.recent))


class ScoredTest(BaseCallback):
    """Every eval_every steps, scores the current policy on the fixed starts and keeps the best checkpoint."""
    def __init__(self, run_dir: str, env, eval_every: int, episodes: int, baseline: dict):
        super().__init__()
        self.run_dir, self.eval_every, self.episodes, self.baseline = run_dir, eval_every, episodes, baseline
        self.env = env
        self.time_key = time_key(env)
        self.next_eval = eval_every
        self.best = (-1.0, float("inf"))   # (success rate, time to gate)
        self.csv_path = os.path.join(run_dir, "eval.csv")
        self.started = time.time()

    def _save(self, name: str):
        path = os.path.join(self.run_dir, f"{name}.zip")
        self.model.save(path)
        self.model.get_vec_normalize_env().save(vecnormalize_path(path))

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.eval_every
        normalizer = copy.deepcopy(self.model.get_vec_normalize_env())
        normalizer.venv = None   # Only its statistics are needed
        result = score(self.env, policy_controller(self.model, normalizer), self.episodes)
        for key, value in result.items():
            self.logger.record(f"eval/{key}", value)
        self.logger.record("eval/scripted_success_rate", self.baseline["success_rate"])
        self.logger.record(f"eval/scripted_{self.time_key}", self.baseline[self.time_key])
        self.logger.dump(self.num_timesteps)

        self._save("latest_model")
        t = result[self.time_key] if result["success_rate"] > 0 else float("inf")
        is_best = (result["success_rate"], -t) > (self.best[0], -self.best[1])
        if is_best:
            self.best = (result["success_rate"], t)
            self._save("best_model")
        new_file = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["steps", "minutes", *result.keys(), "best"])
            w.writerow([self.num_timesteps, f"{(time.time() - self.started) / 60:.1f}",
                        *[f"{v:.3f}" for v in result.values()], int(is_best)])
        failures = ", ".join(f"{k[:-5].replace('_', ' ')} {100 * v:.0f}%" for k, v in result.items()
                             if k.endswith("_rate") and k != "success_rate")
        extras = "".join(f", {format_extra(k, result[k])}" for k in self.env.EXTRA_INFO)
        print(f"[{(time.time() - self.started) / 60:5.1f} min] {self.num_timesteps:>9,} steps: "
              f"success {100 * result['success_rate']:3.0f}% (scripted {100 * self.baseline['success_rate']:.0f}%), "
              f"{failures}, time {result[self.time_key]:.1f} s{extras}{'  <- best' if is_best else ''}", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="gate", help="Run directory name under runs/")
    parser.add_argument("--task", choices=sorted(envs.ENVS), default="gate")
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="perfect")
    parser.add_argument("--steps", type=float, default=3e6, help="Total environment steps (10 per simulated second)")
    parser.add_argument("--envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch threads for the PPO update")
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="Stop each PPO update once the policy has moved this far (KL divergence); 0 = no limit")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Starting learning rate; falls linearly to 0")
    parser.add_argument("--resume", help="Continue training from this checkpoint .zip (and its _vecnormalize.pkl) "
                                         "instead of starting fresh; --task must match")
    parser.add_argument("--margin-rule", choices=["hard", "near", "soft"], default="hard",
                        help="Orbit skill, training only: whether entering the marker's safety margin ends the "
                             "episode (hard, learns more reliably), only coming within 1 m does (near), or it just "
                             "costs reward (soft). Scoring always uses soft")
    parser.add_argument("--randomize", action="store_true",
                        help="Training only: vary PID gains, inertia, drag and thrust each episode "
                             "(config.RandomizationConfig). Scored tests use nominal values")
    parser.add_argument("--mission-starts", help="Training only: JSON of recorded mission handoffs "
                        "(tools/record_handoffs.py); a share of episodes start from them")
    parser.add_argument("--wide-starts", action="store_true",
                        help="Gate skill, training only: starts 3-11 m out, up to 3.5 m off center, 0.1-1.9 m deep, "
                             "covering the return from the orbit as well as the forward pass")
    parser.add_argument("--search-starts", type=float, default=0.0,
                        help="Gate skill, training only: share of episodes turned 40-110 deg away from the gate, "
                             "so it starts out of view and has to be found")
    parser.add_argument("--scramble-starts", type=float, default=0.0,
                        help="Turnaround training only: share of starts whose white poles' stale range details are redrawn")
    parser.add_argument("--back-lead-ins", type=float, default=0.0,
                        help="Slalom training only: share of back starts flown in through the whole slalom and the turnaround")
    parser.add_argument("--gate-lead-ins", type=float, default=0.0,
                        help="Slalom training only: share of gate-handoff starts flown in by the gate skill from RoboSim's start")
    parser.add_argument("--log-std-init", type=float, default=0.0,
                        help="Starting exploration spread, as a log (0 = spread 1). The dance needs less: with a "
                             "spread of 1 on the turn (+-45 deg) the sub never held a stop still long enough to count")
    parser.add_argument("--center-line-weight", type=float, default=RewardConfig.center_line_weight,
                        help="Gate skill: reward weight for distance from the gate's center line")
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    # Each update uses 8192 steps however many environments collect them
    hyper = dict(n_steps=max(64, 8192 // args.envs), batch_size=256, n_epochs=10, gamma=0.995, gae_lambda=0.95,
                 learning_rate=args.learning_rate, clip_range=0.2, ent_coef=0.005,
                 target_kl=args.target_kl or None,
                 policy_kwargs=dict(net_arch=[128, 128], log_std_init=args.log_std_init))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "ppo": hyper}, f, indent=2)
    torch.set_num_threads(args.threads)

    print("Scoring the scripted baseline on the evaluation starts...", flush=True)
    kwargs = env_kwargs(args.task, args)
    eval_env = envs.make(args.task, args.detector, **env_kwargs(args.task, args, scoring=True))
    baseline = score(eval_env, make_controller("scripted"), args.eval_episodes)
    print(f"Scripted: success {100 * baseline['success_rate']:.0f}%, time {baseline[time_key(eval_env)]:.1f} s", flush=True)

    venv = SubprocVecEnv([make_env(args.task, args.detector, args.seed * 1000 + i, kwargs) for i in range(args.envs)])
    # Learning rate falls linearly to 0 over the run, which steadies the late updates
    schedule = lambda progress_left: args.learning_rate * progress_left
    if args.resume:
        venv = VecNormalize.load(vecnormalize_path(args.resume), venv)
        venv.training, venv.norm_reward = True, True
        model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=os.path.join(run_dir, "tb"),
                         custom_objects={"learning_rate": schedule, "lr_schedule": schedule,
                                         "target_kl": hyper["target_kl"]})
        # PPO.load restores the checkpoint's own seed (environments and PyTorch), so without this every
        # fine-tune from one checkpoint was the same run, whatever --seed said
        model.set_random_seed(args.seed)
        print(f"Resuming from {args.resume}", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=hyper["gamma"])
        model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0,
                    tensorboard_log=os.path.join(run_dir, "tb"), **{**hyper, "learning_rate": schedule})
    callbacks = [OutcomeLogger(eval_env.OUTCOMES, time_key(eval_env), eval_env.EXTRA_INFO),
                 ScoredTest(run_dir, eval_env, args.eval_every, args.eval_episodes, baseline)]
    print(f"Training {int(args.steps):,} steps with {args.envs} environments; logs in {run_dir}/tb", flush=True)
    model.learn(total_timesteps=int(args.steps), callback=callbacks, tb_log_name="ppo")
    model.save(os.path.join(run_dir, "final_model.zip"))
    venv.save(vecnormalize_path(os.path.join(run_dir, "final_model.zip")))
    venv.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
