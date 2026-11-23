"""Outcome judge data, whole episodes: grasp -> pour -> put back at the trained spots 1-4, a random one of the K chunks
runs at every replan of LOOK_PHASES (ARM=random: the plain policy's states; elsewhere the plain policy alone). At
those replans: the camera frames and state, the K chunks' first LOOKAHEAD actions and, per chunk, the sim state() after
its look-ahead; plus the phase the replan was in and the state then. The phase only sets how far each chunk is looked
ahead (the start of the pour: HORIZON more steps of the plain policy, its effect shows late; elsewhere 1 s) and how
judge_check.py scores it; the judge never sees it.
The pilot (trials 0-1) looked ahead in every phase, 3 s also in grasp and put back: those rows stay in the data.
Spot 0 (units <plant>-0-<t>): a random mug start per trial anywhere in AREA, KEEP_OUT from the pour benchmark's spots
5-8, so the judge can't learn 4 fixed scenes. Each replan also keeps the mug's start and the policy's grasp memory.
-> results/judge-full/<model>/data/<unit>.npz

Usage: ./run judge-full [--jobs 8]
       PART=2/3 ./run judge-full [--jobs 8]   this machine's share of the trials (the 2nd of 3), for several pods
"""
import os

import numpy as np

import cheat
import evaluate as ev
import grasp_position as gp
from dataset.export_lerobot import CAMERAS
from judge_check import TIP_DEG  # carried mug tilted at least this: the tip (1 s look-ahead); less: the start of the pour

SPOTS = (0,)                # 0: a random start per trial (AREA); before: the trained spots 1-4 (5-8 stay unseen)
TRIALS = range(0, 20)       # per plant x spot (new names and seeds): 60 episodes; spots 1-4: pilot 0-1, main run 2-6, then 7-16
AREA = ((-24.0, 14.0), (27.0, 30.0))  # cm, real table frame (x, y) min and max: the strip spots 1-8 lie in
KEEP_OUT = 4.0              # cm: no random start this close to spots 5-8, the pour benchmark's unseen spots
LOOK_PHASES = ("pour start", "tip")  # pilot: grasp's choices were mostly distance noise; the put back is left to the policy
MUG_JITTER = 0.03           # m, as DAgger's rollouts (eval: 0.01)
SEED = 30_000               # added to evaluate.trial_seed: no eval, distill.py (10_000) or judge.py (20_000) trial's seed
HORIZON = 40                # plain-policy steps after the chunk at the start of the pour (+ LOOKAHEAD: 3 s)

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
        home_cm=o("home_cm"), up_cm=o("up_cm"), start=_rec["start"], grasp=gp.CURRENT.numpy().copy()))


def random_start(rng):
    """A mug start (cm, real table frame) in AREA, at least KEEP_OUT from spots 5-8."""
    while True:
        xy = tuple(rng.uniform(*AREA))
        if min(np.hypot(*np.subtract(xy, ev.SPOTS[s])) for s in (5, 6, 7, 8)) > KEEP_OUT:
            return xy


def reset(env, spot, rng, jitter=MUG_JITTER):
    if spot == 0:  # evaluate.reset reads the start from SPOTS
        ev.SPOTS[0] = random_start(rng)
    obs = _reset(env, spot, rng, 0.0 if spot == 0 else MUG_JITTER)
    _rec.update(rows=[], start=ev.SPOTS[spot])
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
cheat.look = lambda s: phase(s) in LOOK_PHASES
cheat.horizon = lambda s: HORIZON if phase(s) == "pour start" else 0
ev.reset, cheat.trial, ev.run_unit = reset, trial, run_unit
ev.trial_seed = lambda plant, spot, t: SEED + _seed(plant, spot, t)
cheat.RESULTS = ev.RESULTS = cheat.RESULTS.parent / "judge-full"
ev.UNITS = [f"{p}-{s}-{t}" for t in TRIALS for p in ev.PLANTS for s in SPOTS]  # trial by trial: stopped early, every plant and spot
if "PART" in os.environ:  # every n-th trial from the k-th: each pod a mix of plants and spots
    k, n = map(int, os.environ["PART"].split("/"))
    ev.UNITS = ev.UNITS[k - 1::n]

if __name__ == "__main__":
    ev.main()
