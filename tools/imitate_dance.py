#!/usr/bin/env python3
"""
Teaches the dance by imitation, as a start for PPO (train.py --resume).

PPO alone never learned more than the first stop (tdance2-9, PLAN.md): exploring shook the sub
off its stops and the turn-ons were never found. Here a teacher that can see the judges' count
(stage, true roll) shows the policy what to do, and the policy, which sees only its measured
observation, copies it:

  round 0     the teacher flies, with noise on what it does (so the data has recoveries in it)
  rounds 1+   the student flies (with a little noise) and the teacher labels each step: DAgger,
              so the student learns to recover from its own mistakes, not only the teacher's
  each round  fit the policy's mean action to the teacher's on everything so far

Saves a PPO checkpoint (the same network as train.py's) and its VecNormalize, and scores it on
train.py's evaluation starts. Then: train.py --task tasks_dance --resume runs/NAME/best_model.zip

    .venv/bin/python tools/imitate_dance.py --name tdance_bc1
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks_dance_env import STAGES, TasksDanceEnv   # noqa: E402

LOG_STD = -1.0   # The spread the checkpoint starts PPO with


def teacher_action(env: TasksDanceEnv) -> np.ndarray:
    """The hand controller: turn the target to the stage's stop, then roll right (deep first) and back.
    It moves on from a stop only once the student's own stop_done says so: moving on the moment the
    judges counted, a step or two before stop_done, gave the same observation both labels, and the
    student, averaging them, pushed on at half strength."""
    stage = min(env.stage, len(STAGES) - 1)
    axis, angle = STAGES[stage]
    b = env.builder
    want = angle if axis == "yaw" else 0.0
    off = env.turn_target - want
    turn = 1.0 if off > 20 else -1.0 if off < -20 else 0.0
    if env.turn_target % 90.0 == 0.0 and not b.stop_done:   # At a stop, not yet done by its own count
        turn = 0.0
    if axis == "yaw":
        return np.array([turn, 0.0, 0.0])
    err = angle - b.rolled
    if stage == len(STAGES) - 1:   # Level again: the env's leveler settles it
        roll = 0.0
    else:   # Each full roll starts deep: rolling uses all the vertical thrust and the sub rises
        starting = stage in (8, 12) and abs(b.rolled - (0.0 if stage == 8 else 360.0)) <= 20
        roll = float(np.sign(err)) if (env.readings.depth > 1.6 and b.stop_done) or not starting else 0.0
    return np.array([turn, 1.0, roll])


def _load_student(path):
    if path is None:
        return None
    from gate_controllers import load_policy
    return load_policy(path)


def collect(job):
    """Flies episodes, returning (observations, teacher actions, successes)."""
    seeds, student_path, noise, stage_share = job
    student = _load_student(student_path)
    rng = np.random.default_rng(seeds[0])
    obs_all, act_all, wins = [], [], 0
    for seed in seeds:
        env = TasksDanceEnv(randomize=True, stage_share=stage_share)
        obs, _ = env.reset(seed=seed)
        while True:
            label = teacher_action(env)
            obs_all.append(obs)
            act_all.append(label)
            if student is None:
                act = label
            else:
                model, normalizer = student
                act = model.predict(normalizer.normalize_obs(obs[None]), deterministic=True)[0][0]
            act = np.clip(act + rng.normal(0.0, noise, act.shape), -1, 1)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                wins += bool(info.get("is_success"))
                break
    return np.asarray(obs_all, np.float32), np.asarray(act_all, np.float32), wins


def new_model():
    """A PPO model and VecNormalize built as train.py builds them."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    venv = VecNormalize(DummyVecEnv([lambda: TasksDanceEnv()]), norm_obs=True, norm_reward=True, clip_obs=10.0,
                        gamma=0.995)
    model = PPO("MlpPolicy", venv, device="cpu", n_steps=910, batch_size=256, n_epochs=10, gamma=0.995,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
                policy_kwargs=dict(net_arch=[128, 128], log_std_init=LOG_STD))
    return model, venv


def fit(model, venv, obs, act, epochs: int, seed: int):
    """Fits the policy's mean action to the teacher's, on observations normalized from this data."""
    venv.obs_rms.mean = obs.mean(axis=0).astype(np.float64)
    venv.obs_rms.var = obs.var(axis=0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))
    x = torch.as_tensor(venv.normalize_obs(obs), dtype=torch.float32)
    y = torch.as_tensor(act, dtype=torch.float32)
    policy = model.policy
    params = (list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())
              + list(policy.features_extractor.parameters()))
    opt = torch.optim.Adam(params, lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        total = 0.0
        for i in range(0, len(x), 1024):
            idx = order[i:i + 1024]
            latent = policy.mlp_extractor.forward_actor(policy.extract_features(x[idx], policy.pi_features_extractor))
            loss = ((policy.action_net(latent) - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 9 or epoch == epochs - 1:
            print(f"    epoch {epoch + 1}: mse {total / len(x):.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rounds", type=int, default=5, help="Rounds of collecting and fitting (round 0 is the teacher)")
    parser.add_argument("--episodes", type=int, default=180, help="Episodes per round")
    parser.add_argument("--teacher-noise", type=float, default=0.25)
    parser.add_argument("--student-noise", type=float, default=0.1)
    parser.add_argument("--stage-starts", type=float, default=0.5, help="Share of episodes starting at a random stop")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "best_model.zip")
    torch.manual_seed(args.seed)
    model, venv = new_model()
    obs_all, act_all = [], []
    import envs
    from gate_controllers import make_controller, vecnormalize_path
    from train import score
    for r in range(args.rounds):
        student = None if r == 0 else path
        noise = args.teacher_noise if r == 0 else args.student_noise
        base = 1_000_000 + args.seed * 100_000 + r * 10_000
        seeds = [base + i for i in range(args.episodes)]
        jobs = [(seeds[w::args.workers], student, noise, args.stage_starts) for w in range(args.workers)]
        with Pool(args.workers) as pool:
            parts = pool.map(collect, jobs)
        wins = sum(p[2] for p in parts)
        obs_all += [p[0] for p in parts]
        act_all += [p[1] for p in parts]
        obs, act = np.concatenate(obs_all), np.concatenate(act_all)
        print(f"Round {r} ({'teacher' if r == 0 else 'student'} flying): {wins}/{args.episodes} danced, "
              f"{len(obs):,} samples so far", flush=True)
        fit(model, venv, obs, act, args.epochs, args.seed + r)
        model.save(path)
        venv.save(vecnormalize_path(path))
        result = score(envs.make("tasks_dance", "noisy"), make_controller(path), args.eval_episodes)
        print(f"  scored: success {100 * result['success_rate']:.0f}%, stops {result['stops']:.1f}, "
              f"lost {100 * result['lost_rate']:.0f}%, collision {100 * result['collision_rate']:.0f}%, "
              f"timeout {100 * result['timeout_rate']:.0f}%, time {result['time_to_success']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
