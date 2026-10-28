#!/usr/bin/env python3
"""
Trains a skill's done detector (done_detector.py) by supervised learning.

1. Collect: the skill's policy flies --episodes training-style starts (noisy
   detector, varied dynamics on every other episode). Each 10 Hz step records
   the observation and the ground-truth label (env.done_label()). Episodes keep
   flying EXTRA_S past the moment the label turns true, so the detector sees
   finished states too, not just the instant of finishing.
2. Train: a 128-128 tanh network with a sigmoid output. A fifth of the episodes
   are held out. (Weighting the rarer "done" moments up made it fire on
   ambiguous states, such as facing the gate halfway around the marker.)
3. Report how it fires: for each threshold and number of steps in a row, when it
   first fires on the held-out episodes compared with the true finish. This is
   information for choosing --threshold and --confirm, judged in the full mission.
   Offline, firing a little early looks as bad as firing far too early, though in
   the mission it's harmless; missing the finish is costly (the orbit keeps
   circling). For the orbit, the cautious settings this table favors scored
   115/120 in the mission, the default (0.5, 1) 119/120.

Writes done_detector.npz next to the policy.

Usage: .venv/bin/python tools/train_done_detector.py --task gate --policy runs/gate_carry_ft/final_model.zip
Seeds start at 60000, apart from training (0+), handoff recording (40000+) and the scored tests.
"""
import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXTRA_S = 3.0          # Keep flying this long after the label turns true
ON_TIME_S = 1.0        # A first firing within this of the true finish, either way, counts as on time
THRESHOLDS = (0.5, 0.7, 0.9, 0.97)
CONFIRMS = (1, 3, 5, 10)


def make_env(task: str, handoffs: dict, randomize: bool, mission_start: bool = False):
    if task == "tasks_gate" and mission_start:
        import envs   # The learned mission's own start (RoboSim's), going out
        from tasks_gate_env import TasksGateStarts
        starts = TasksGateStarts(return_share=0.0, forward_distance_m=(9.144, 9.144), lateral_m=(-0.914, 0.914),
                                 course_heading_share=1.0, depth_m=(0.5, 0.5))
        env = envs.make(task, "noisy", starts=starts, randomize=randomize)
    elif task.startswith("tasks_"):
        import envs   # The tasks skills' own training starts (the slalom's include the gate's handoff)
        env = envs.make(task, "noisy", randomize=randomize)
    elif task == "gate":
        from gate_env import GateEnv, StartConfig
        start = StartConfig(distance_m=(3.0, 11.0), lateral_m=(-3.5, 3.5), depth_m=(0.1, 1.9), search_share=0.3)
        env = GateEnv("noisy", start=start, randomize=randomize, mission_return_starts=handoffs.get("RETURN"))
    else:
        from orbit_env import OrbitEnv
        env = OrbitEnv("noisy", practice_starts=True, randomize=randomize, mission_starts=handoffs.get("ORBIT"))
    env.end_on_complete = False
    return env


HISTORY_TARGET = {"tasks_gate": "passage", "tasks_slalom": "red"}   # Whose closest range the detector keeps


def history_fields(task: str):
    """Observation indices (newest frame) of the target's found and range and the surge action, or None."""
    if task not in HISTORY_TARGET:
        return None
    from tasks_observation import GATE_FIELDS, SLALOM_FIELDS
    fields = GATE_FIELDS if task == "tasks_gate" else SLALOM_FIELDS
    target = HISTORY_TARGET[task]
    return np.array([fields.index(f"{target}_found"), fields.index(f"{target}_range"), fields.index("action_surge")])


def collect(job):
    task, policy, handoffs, seeds, mission_share = job
    from done_detector import History
    from gate_controllers import load_policy
    model, norm = load_policy(policy)
    episodes = []
    for seed in seeds:
        env = make_env(task, handoffs, randomize=seed % 2 == 1,
                       mission_start=(seed // 2) % 100 < 100 * mission_share)
        obs, _ = env.reset(seed=seed)
        rows, turned, labels, before, onset = [], [], [], [], None
        # Practice starts begin partway around the marker, facing it, so the heading has
        # already turned about as far as the sweep
        history = History(history_fields(task))
        history.reset(float(getattr(env, "_preset_swept", 0.0)))
        with redirect_stdout(io.StringIO()):
            while True:
                history.update(obs, env.readings.heading)   # Measured heading, as on the sub
                rows.append(obs)
                turned.append(history.features())
                label = bool(env.done_label())
                labels.append(label)
                before.append(not label and not getattr(env, "passed", True))   # Not yet through the gate or set
                if label and onset is None:
                    onset = env.time
                if onset is not None and env.time >= onset + EXTRA_S:
                    break
                action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
        episodes.append((np.array(rows, dtype=np.float32), np.array(turned, dtype=np.float32), np.array(labels), onset,
                         np.array(before)))
    return episodes


def train(x, y, epochs: int, seed: int = 0, w=None):
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.Tanh(),
                              torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1))
    xt, yt = torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    wt = torch.ones(len(yt)) if w is None else torch.tensor(w, dtype=torch.float32)
    loss_fn = lambda logits, target, weight: (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none") * weight).sum() / weight.sum()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(xt)).split(1024):
            opt.zero_grad()
            loss_fn(net(xt[idx]).squeeze(1), yt[idx], wt[idx]).backward()
            opt.step()
    return [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
            for m in net if isinstance(m, torch.nn.Linear)]


def first_firings(probs: list, threshold: float, confirm: int):
    """Per episode: the time the detector would first fire with these settings, or None."""
    out = []
    for p in probs:
        above = p > threshold
        streak, fired = 0, None
        for i, a in enumerate(above):
            streak = streak + 1 if a else 0
            if streak >= confirm:
                fired = i * 0.1
                break
        out.append(fired)
    return out


def judge(fired: list, onsets: list) -> dict:
    counts = dict.fromkeys(["early", "on time", "late", "never", "false alarm", "quiet"], 0)
    lag = []
    for t_fire, t_true in zip(fired, onsets):
        if t_true is None:
            counts["false alarm" if t_fire is not None else "quiet"] += 1
        elif t_fire is None:
            counts["never"] += 1
        else:
            lag.append(t_fire - t_true)
            counts["early" if t_fire < t_true - ON_TIME_S else "late" if t_fire > t_true + ON_TIME_S else "on time"] += 1
    counts["median_lag_s"] = float(np.median(lag)) if lag else float("nan")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=["gate", "orbit", "tasks_gate", "tasks_slalom", "tasks_turnaround"],
                        required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seed", type=int, default=60_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--handoffs", default="runs/handoffs.json")
    parser.add_argument("--mission-starts", type=float, default=0.0,
                        help="tasks_gate: share of episodes from the mission's own start, going out. From there the gate "
                             "skill can glimpse its passage far off and drift for a while, and a detector trained on the "
                             "training starts fired during the drift (~4%%)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability above which a step counts as done")
    parser.add_argument("--confirm", type=int, default=1, help="Steps in a row above the threshold before it fires")
    parser.add_argument("--before-weight", type=float, default=1.0,
                        help="Weight of the moments before the pass (tasks gate and slalom). Firing before the set "
                             "breaks the mission (the next slalom step goes through the same set), firing late doesn't")
    args = parser.parse_args()

    from done_detector import DoneDetector, detector_path
    from gate_controllers import load_policy
    handoffs = json.load(open(args.handoffs)) if os.path.exists(args.handoffs) else {}
    seeds = list(range(args.seed, args.seed + args.episodes))
    jobs = [(args.task, args.policy, handoffs, seeds[i::args.workers], args.mission_starts) for i in range(args.workers)]
    with Pool(args.workers) as pool:
        episodes = [e for part in pool.map(collect, jobs) for e in part]
    finished = sum(e[3] is not None for e in episodes)
    print(f"{len(episodes)} episodes, {finished} reached the finish; "
          f"{sum(len(e[0]) for e in episodes):,} labeled moments", flush=True)

    _, norm = load_policy(args.policy)
    mean, var, clip = norm.obs_rms.mean.astype(np.float32), norm.obs_rms.var.astype(np.float32), norm.clip_obs
    features = lambda rows, extra: np.column_stack([np.clip((rows - mean) / np.sqrt(var + 1e-8), -clip, clip),
                                                    extra]).astype(np.float32)
    held_out = episodes[::5]
    train_eps = [e for i, e in enumerate(episodes) if i % 5]
    x = np.concatenate([features(e[0], e[1]) for e in train_eps])
    y = np.concatenate([e[2] for e in train_eps]).astype(np.float32)
    w = np.where(np.concatenate([e[4] for e in train_eps]), args.before_weight, 1.0)
    layers = train(x, y, args.epochs, w=w)

    out = detector_path(args.policy)
    arrays = {f"w{i}": w for i, (w, _) in enumerate(layers)}
    arrays.update({f"b{i}": b for i, (_, b) in enumerate(layers)})
    fields = history_fields(args.task)
    if fields is not None:
        arrays["history_fields"] = fields
    save = lambda threshold, confirm: np.savez(out, n_layers=len(layers), obs_mean=mean, obs_var=var, clip_obs=clip,
                                               threshold=threshold, confirm=confirm, **arrays)
    save(args.threshold, args.confirm)

    # When it would first fire on episodes it never saw, for each setting
    det = DoneDetector(out)
    probs = [np.array([det.probability(o, t) for o, t in zip(e[0], e[1])]) for e in held_out]
    onsets = [None if e[3] is None else float(np.argmax(e[2])) * 0.1 for e in held_out]
    acc = np.mean([(p > 0.5) == l for ps, e in zip(probs, held_out) for p, l in zip(ps, e[2])])
    before_fire = sum(any(p > args.threshold for p in ps[b]) if b.any() else False
                      for ps, b in ((ps, e[4]) for ps, e in zip(probs, held_out)))
    print(f"Held out {len(held_out)} episodes ({sum(o is not None for o in onsets)} finished); "
          f"per-moment accuracy {100 * acc:.1f}%; above {args.threshold} before the pass in {before_fire}")
    for threshold in THRESHOLDS:
        for confirm in CONFIRMS:
            c = judge(first_firings(probs, threshold, confirm), onsets)
            print(f"  threshold {threshold:.2f}, {confirm:2d} in a row: early {c['early']}, on time {c['on time']}, "
                  f"late {c['late']}, never {c['never']}, false alarm {c['false alarm']}, "
                  f"median lag {c['median_lag_s']:+.1f} s")
    save(args.threshold, args.confirm)
    print(f"Saved with threshold {args.threshold}, {args.confirm} in a row -> {out}")


if __name__ == "__main__":
    main()
