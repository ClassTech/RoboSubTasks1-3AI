#!/usr/bin/env python3
"""
Copies a slalom or turnaround policy trained on the 31-value slalom frame into
the 32-value frame with red_travel (dead reckoning since the set was last seen,
index 28 of each frame). The new input starts with zero weight and neutral
normalization, so the copy flies exactly as the original; fine-tuning then
learns to use it. Saves best_model.zip and its _vecnormalize.pkl in --out.

Usage: .venv/bin/python tools/expand_slalom_obs.py runs/tslalom8_ft_s4/best_model.zip --task tasks_slalom --out runs/tslalom8_dr
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLD_F, NEW_F, AT, STACK = 31, 32, 28, 4


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("policy")
    parser.add_argument("--task", choices=["tasks_slalom", "tasks_turnaround"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import envs
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    src = args.policy[:-len(".zip")]
    old = PPO.load(src + ".zip", device="cpu")
    old_norm = pickle.load(open(src + "_vecnormalize.pkl", "rb"))
    venv = VecNormalize(DummyVecEnv([lambda: envs.make(args.task, "noisy")]), norm_obs=True, norm_reward=True,
                        clip_obs=10.0, gamma=old.gamma)
    new = PPO("MlpPolicy", venv, device="cpu", policy_kwargs=old.policy_kwargs,
              **{k: getattr(old, k) for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef")})
    new_idx = np.array([k * NEW_F + (i if i < AT else i + 1) for k in range(STACK) for i in range(OLD_F)])
    sd_old, sd_new = old.policy.state_dict(), new.policy.state_dict()
    for k, v in sd_old.items():
        if v.shape == sd_new[k].shape:
            sd_new[k] = v.clone()
        else:   # A first layer: the old columns in their new places, zeros for the new input
            w = torch.zeros_like(sd_new[k])
            w[:, new_idx] = v
            sd_new[k] = w
    new.policy.load_state_dict(sd_new)
    mean, var = np.zeros(STACK * NEW_F), np.ones(STACK * NEW_F)
    mean[new_idx], var[new_idx] = old_norm.obs_rms.mean, old_norm.obs_rms.var
    venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = mean, var, old_norm.obs_rms.count
    venv.ret_rms = old_norm.ret_rms
    os.makedirs(args.out, exist_ok=True)
    new.save(os.path.join(args.out, "best_model.zip"))
    venv.save(os.path.join(args.out, "best_model_vecnormalize.pkl"))

    env = envs.make(args.task, "noisy")
    o, _ = env.reset(seed=3)
    worst = 0.0
    for _ in range(200):
        a_new = new.predict(venv.normalize_obs(o), deterministic=True)[0]
        a_old = old.predict(old_norm.normalize_obs(np.delete(o, [k * NEW_F + AT for k in range(STACK)])),
                            deterministic=True)[0]
        worst = max(worst, float(np.abs(a_new - a_old).max()))
        o, _, term, trunc, _ = env.step(np.clip(a_new, -1, 1))
        if term or trunc:
            o, _ = env.reset()
    print(f"Saved to {args.out}; largest action difference from the original over 200 steps: {worst:.1e}")


if __name__ == "__main__":
    main()
