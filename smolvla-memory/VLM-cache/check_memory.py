"""Two checks of a memory model on its own training data, no sim:
  swap  -> action loss late in each episode (carry back, set down) with its own start frame as the memory, another
           episode's, and none: if the three match, the policy ignores the memory
  cache -> eval's cached actions (memory.CACHE) against the uncached ones, which run as in training: they should match

Usage: ./run memory-check [libero-4s-dart-xl-sd-mem ...]
"""
import argparse

import pyarrow.parquet as pq
import torch
from lerobot.datasets import LeRobotDataset
from torch.utils.data import default_collate

import evaluate as ev
import memory
import runs

EPISODES = 128     # episodes sampled, one late frame each
BATCH = 16         # samples per forward pass
LATE = (0.6, 0.95)  # share of the episode the frame is drawn from (shortcut: not the detected grasp, the late part of every demo is after it)
SEED = 0


def samples(ds, rng):
    """One random late frame per sampled episode -> dataset indices."""
    t = pq.read_table(ds.root / "data", columns=["index", "episode_index"]).to_pandas()
    eps = t.groupby("episode_index")["index"].agg(["min", "max"])
    eps = eps.iloc[torch.randperm(len(eps), generator=rng)[:EPISODES].numpy()]
    lo, hi = eps["min"] + LATE[0] * (eps["max"] - eps["min"]), eps["min"] + LATE[1] * (eps["max"] - eps["min"])
    return [int(a + torch.rand(1, generator=rng) * (b - a)) for a, b in zip(lo, hi)]


def loss(policy, batch, episode_index, blank=False):
    """Mean action loss with the memory of episode_index (B,); same noise and times every call."""
    memory.BLANK = blank
    torch.manual_seed(SEED)
    with torch.no_grad():
        out, _ = policy.forward({**batch, "episode_index": episode_index})
    memory.BLANK = False
    return float(out)


def actions(policy, batch, cache):
    """A chunk per sample, with the memory cached or not; same noise both times."""
    memory.CACHE = cache
    policy.reset()
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = policy.predict_action_chunk(batch)
    memory.CACHE = True
    return out


def check(name):
    if runs.MODELS[name.partition("@")[0]].get("script", "").endswith("isolated.py"):
        import isolated  # noqa: F401  (its patches, at import)
    policy, pre, _ = ev.load_model(name)
    data = runs.MODELS[name.partition("@")[0]]["data"]
    memory.TABLE = memory.memory_table(data)
    ds = LeRobotDataset(data, video_backend="pyav",
                        delta_timestamps={"action": [i / ev.CONTROL_HZ for i in range(policy.config.chunk_size)]})
    idx = samples(ds, torch.Generator().manual_seed(SEED))

    sums, gap = {"own": 0.0, "swapped": 0.0, "none": 0.0}, 0.0
    for k in range(0, len(idx), BATCH):
        batch = pre(default_collate([ds[i] for i in idx[k:k + BATCH]]))
        own = batch["episode_index"]
        sums["own"] += loss(policy, batch, own)
        sums["swapped"] += loss(policy, batch, own.roll(1))  # each sample gets the next sample's (other) episode
        sums["none"] += loss(policy, batch, own, blank=True)
        if k == 0:  # as in eval: each chunk remembers its own frame (a trial's first)
            gap = float((actions(policy, batch, cache=True) - actions(policy, batch, cache=False)).abs().max())
    n = -(-len(idx) // BATCH)
    print(f"\n=== {name}: action loss on {len(idx)} late frames (lower is better)")
    for key, s in sums.items():
        print(f"  memory {key:8s} {s / n:.4f}")
    print(f"  cached vs uncached actions: max difference {gap:.2e} (normalized units; ~1e-3 or less: the same)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", default=["libero-4s-dart-xl-sd-mem"])
    for name in p.parse_args().models:
        check(name)


if __name__ == "__main__":
    main()
