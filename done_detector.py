#!/usr/bin/env python3
"""
A skill's done detector: a small network that looks at the same observation
as the skill's policy (only what the real sub measures) and says whether the
skill's job is finished. The mission runner moves to the next skill when it
fires, so the switching is learned, not hand-written.

It's trained by supervised learning (tools/train_done_detector.py) on moments
flown by the skill's own policy, each labeled with the simulator's ground
truth (the environment's done_label()): for the gate, through the opening and
1.5 m beyond; for the orbit, 270° around the marker and facing the gate.
Learning this from the policy's reward instead failed: the cost of saying done
too early came every step and the payoff for the right moment rarely, so the
policies learned never to say it.

Besides the observation it gets one more measured number: the total heading
turned since the skill began (magnetometer, unwrapped). The observation only
covers the last 0.4 s, and without this the orbit's detector mistook facing
the gate a quarter of the way around for the finish three quarters of the way
around: mirror images, told apart only by which way the sub has been turning.

Detectors for the tasks skills also get a little history about their target
(the gate skill's passage, the slalom's red pole), from the observation alone:
the closest range seen while it was in view, seconds since then, and roughly
how far the sub has driven since (its own surge commands added up). Close to a
slalom set its poles leave the 70 deg view, so the last meter before the set
and the meters after it look alike for 0.4 s; without this the slalom's
detector fired seconds early, about 2 m short of the set.

numpy only, so it runs on the Pi unchanged. A detector file (.npz) carries its
own copy of the observation normalization the policy was trained with.
"""
import numpy as np

TURNED_SCALE_DEG = 360.0
SINCE_SCALE_S = 10.0
TRAVEL_SCALE_M = 3.0
TOP_SPEED_MPS = 0.34    # Full surge, for the rough distance driven
DT_S = 0.1


class History:
    """The extra inputs, kept step by step from the raw observation: the heading turned, and (if `fields`
    names the target's found/range and the surge action in the observation) the target's closest range
    in view, seconds since then, and distance driven since then."""
    def __init__(self, fields=None):
        self.fields = None if fields is None or len(fields) == 0 else [int(i) for i in fields]
        self.reset()

    def reset(self, turned: float = 0.0):
        self.turned, self._last_heading = turned, None
        self.closest, self.since, self.travel = 3.0, 0.0, 0.0

    def update(self, obs, heading: float):
        if self._last_heading is not None:
            self.turned += (heading - self._last_heading + 180.0) % 360.0 - 180.0
        self._last_heading = heading
        if self.fields:
            found, rng, surge = (float(obs[i]) for i in self.fields)
            self.since += DT_S
            self.travel += surge * TOP_SPEED_MPS * DT_S
            if found > 0.5 and rng <= self.closest:
                self.closest, self.since, self.travel = rng, 0.0, 0.0

    def features(self) -> list:
        f = [self.turned / TURNED_SCALE_DEG]
        if self.fields:
            f += [self.closest, min(self.since, 60.0) / SINCE_SCALE_S, float(np.clip(self.travel, -6, 12)) / TRAVEL_SCALE_M]
        return f


class DoneDetector:
    """Fires once the probability has stayed above `threshold` for `confirm` steps in a row (both saved
    with the detector), so one odd moment can't end a skill early. Call reset() when the skill starts."""
    def __init__(self, path: str, threshold: float = None, confirm: int = None):
        d = np.load(path)
        self.mean, self.std, self.clip = d["obs_mean"], np.sqrt(d["obs_var"] + 1e-8), float(d["clip_obs"])
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))]
        self.threshold = float(d["threshold"]) if threshold is None else threshold
        self.confirm = int(d["confirm"]) if confirm is None else confirm
        self.history = History(d["history_fields"] if "history_fields" in d else None)
        self.streak = 0

    def reset(self):
        self.streak = 0
        self.history.reset()

    @property
    def turned(self) -> float:
        return self.history.turned

    def probability(self, obs: np.ndarray, extra: list = None) -> float:
        """Chance the skill is finished, from one raw (unnormalized) observation and the history inputs
        (by default the ones kept by update())."""
        extra = self.history.features() if extra is None else extra
        x = np.clip((np.asarray(obs, dtype=np.float32) - self.mean) / self.std, -self.clip, self.clip)
        x = np.append(x, np.asarray(extra, dtype=np.float32))
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return float(1.0 / (1.0 + np.exp(-x[0])))

    def update(self, obs: np.ndarray, heading: float) -> bool:
        """Feed this step's observation and measured heading; True once the skill is judged finished."""
        self.history.update(obs, heading)
        self.streak = self.streak + 1 if self.probability(obs) > self.threshold else 0
        return self.streak >= self.confirm


def detector_path(policy_path: str) -> str:
    """Where a policy's done detector lives: next to it, as done_detector.npz."""
    import os
    return os.path.join(os.path.dirname(policy_path), "done_detector.npz")
