"""Outcome judge (E3): learn to score SmolVLA's candidate chunks from the cameras, as the cheat does with the sim.

Step 1, data: pour-only trials at the trained spots 1-4 where a random one of the K chunks runs (ARM=random: the plain
policy's states, good and bad ones). At every replan: the camera frames and state, the K chunks' first LOOKAHEAD actions
and, per chunk, where the water and the mug mouth were after them (sim) -> results/judge-data/<model>/data/<unit>.npz.

Usage: ./run judge-record [--jobs 8]
"""
import numpy as np

import cheat
import pour  # noqa: F401 (at import: the pour-only benchmark, the chunk choice; patched further below)
import evaluate as ev
from dataset.export_lerobot import CAMERAS

SPOTS = (1, 2, 3, 4)    # trained spots: the pour benchmark's 5-8 stay unseen
TRIALS = 10             # per plant x spot
MUG_JITTER = 0.03       # m, as DAgger's rollouts (eval: 0.01)
SEED = 20_000           # added to evaluate.trial_seed: neither an eval trial's seed nor distill.py's (10_000)

_reset, _trial, _run_unit, _seed = ev.reset, cheat.trial, ev.run_unit, ev.trial_seed
_rec = {}  # this process's trial: unit, one dict per choice


def on_choice(frame, chunks, outcomes, pick):
    """One replan: what the judge sees (frames, state, the K chunks) and its labels (each chunk's state() after 1 s)."""
    o = lambda k: [s[k] for s in outcomes]
    _rec["rows"].append(dict(
        **{cam: frame[cam] for cam in CAMERAS}, state=frame["observation.state"],
        chunks=chunks[:, :cheat.LOOKAHEAD].astype(np.float32), pick=pick, step=cheat._run["env"].env.timestep,
        over=o("over"), elsewhere=o("elsewhere"), in_mug=o("in_mug"), mouth_cm=o("mouth_cm"), tilt=o("tilt")))


def reset(env, spot, rng, jitter=MUG_JITTER):
    obs = _reset(env, spot, rng, MUG_JITTER)
    _rec["rows"] = []
    return obs


def trial(*args, **kwargs):
    """pour's trial; its choices -> <model>/data/<unit>.npz, with the trial's plant and outcome."""
    r = _trial(*args, **kwargs)
    if _rec["rows"]:
        out = cheat.RESULTS / cheat.MODEL / "data"
        out.mkdir(parents=True, exist_ok=True)
        rows = _rec["rows"]
        np.savez_compressed(out / f"{_rec['unit']}.npz", plant=args[2], watered=r["watered"], in_target=r["in_target"],
                            **{k: np.array([row[k] for row in rows]) for k in rows[0]})
    return r


def run_unit(unit, *args, **kwargs):
    _rec["unit"] = unit
    return _run_unit(unit, *args, **kwargs)


# at import, so the spawned eval workers get them too
cheat.on_choice = on_choice
ev.reset, cheat.trial, ev.run_unit = reset, trial, run_unit
ev.trial_seed = lambda plant, spot, t: SEED + _seed(plant, spot, t)
cheat.RESULTS = ev.RESULTS = cheat.RESULTS.parent / "judge-data"
ev.UNITS = [f"{p}-{s}-{t}" for p in ev.PLANTS for s in SPOTS for t in range(TRIALS)]

if __name__ == "__main__":
    ev.main()
