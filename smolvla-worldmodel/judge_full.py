"""Outcome judge data, whole episodes: grasp -> pour -> put back at the trained spots 1-4, a random one of the K chunks
runs at every replan (ARM=random: the plain policy's states). At every replan: the camera frames and state, the K
chunks' first LOOKAHEAD actions and, per chunk, the sim state() after its look-ahead; plus the phase the replan was in
and the state then. The phase only sets how far each chunk is looked ahead (grasp, start of the pour, put back: HORIZON
more steps of the plain policy, their effect shows late; the tip: 1 s) and how judge_check.py scores it; the judge
never sees it.
-> results/judge-full/<model>/data/<unit>.npz

Usage: ./run judge-full [--jobs 8]
"""
import numpy as np

import cheat
import evaluate as ev
from dataset.export_lerobot import CAMERAS
from judge_check import TIP_DEG  # carried mug tilted at least this: the tip (1 s look-ahead); less: the start of the pour

SPOTS = (1, 2, 3, 4)        # trained spots: 5-8 stay unseen
TRIALS = range(0, 2)        # per plant x spot; pilot 0-1, then range(2, 27) adds new trials (new names and seeds)
MUG_JITTER = 0.03           # m, as DAgger's rollouts (eval: 0.01)
SEED = 30_000               # added to evaluate.trial_seed: no eval, distill.py (10_000) or judge.py (20_000) trial's seed
HORIZON = 40                # plain-policy steps after the chunk, except at the tip (+ LOOKAHEAD: 3 s)

_reset, _trial, _run_unit, _seed = ev.reset, cheat.trial, ev.run_unit, ev.trial_seed
_rec = {}  # this process's trial: unit, one dict per choice


def phase(s):
    """grasp (not lifted yet), pour start (water in, tilt < TIP_DEG), tip, put back (mug empty). s: state() now."""
    if not cheat._run["lifted"]:
        return "grasp"
    if s["in_mug"] == 0:
        return "put back"
    return "pour start" if s["tilt"] < TIP_DEG else "tip"


def on_choice(frame, chunks, outcomes, pick):
    """One replan: what the judge sees (frames, state, the K chunks) and its labels (each chunk's state() after its
    look-ahead); the sim is back at the replan here, so state() is the state then."""
    now = cheat.state(cheat._run["env"].env)
    o = lambda k: [s[k] for s in outcomes]
    _rec["rows"].append(dict(
        **{cam: frame[cam] for cam in CAMERAS}, state=frame["observation.state"],
        chunks=chunks[:, :cheat.LOOKAHEAD].astype(np.float32), pick=pick, step=cheat._run["env"].env.timestep,
        phase=phase(now), horizon=cheat.LOOKAHEAD + cheat.horizon(now), now_tilt=now["tilt"], now_in_mug=now["in_mug"],
        over=o("over"), elsewhere=o("elsewhere"), in_mug=o("in_mug"), mouth_cm=o("mouth_cm"), tilt=o("tilt"), hand_cm=o("hand_cm"),
        home_cm=o("home_cm"), up_cm=o("up_cm")))


def reset(env, spot, rng, jitter=MUG_JITTER):
    obs = _reset(env, spot, rng, MUG_JITTER)
    _rec["rows"] = []
    return obs


def trial(*args, **kwargs):
    """cheat's trial; its choices -> <model>/data/<unit>.npz, with the trial's plant and outcome."""
    r = _trial(*args, **kwargs)
    if _rec["rows"]:
        out = cheat.RESULTS / cheat.MODEL / "data"
        out.mkdir(parents=True, exist_ok=True)
        rows = _rec["rows"]
        np.savez_compressed(out / f"{_rec['unit']}.npz", plant=args[2], watered=r["watered"], in_target=r["in_target"],
                            success=r["success"], **{k: np.array([row[k] for row in rows]) for k in rows[0]})
    return r


def run_unit(unit, *args, **kwargs):
    _rec["unit"] = unit
    return _run_unit(unit, *args, **kwargs)


# at import, so the spawned eval workers get them too
cheat.on_choice = on_choice
cheat.look = lambda s: True
cheat.horizon = lambda s: 0 if phase(s) == "tip" else HORIZON
ev.reset, cheat.trial, ev.run_unit = reset, trial, run_unit
ev.trial_seed = lambda plant, spot, t: SEED + _seed(plant, spot, t)
cheat.RESULTS = ev.RESULTS = cheat.RESULTS.parent / "judge-full"
ev.UNITS = [f"{p}-{s}-{t}" for p in ev.PLANTS for s in SPOTS for t in TRIALS]

if __name__ == "__main__":
    ev.main()
