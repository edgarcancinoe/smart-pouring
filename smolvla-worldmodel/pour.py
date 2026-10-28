"""Pour-only benchmark: each trial replays the plain policy until the mug is carried (exact: same seed as the main eval),
then the arm under test pours, and the trial ends once the mug is empty and the drops have landed. No grasp or
put-back: the score is drops in the target pot (of 10) over the carried trials, and a trial costs the pour only.
Same per-replan log as cheat.py -> results/pour-<arm>/<model>/<plant>-<spot>-<trial>.json, plus carried/ended.

Usage: ./run pour [--jobs 30]         the cheat (best of K chunks), SPOTS x TRIALS
       ./run pour-plain [--jobs 30]   the plain policy, same trials
"""
import numpy as np

import cheat  # patches evaluate (per-replan log, ARM's chunk choice); this file patches on top
import evaluate as ev

TRIALS = 5              # per plant x spot, the main eval's seeds
CARRIED_CM = 5.0        # the pour starts once the mug is this far above where it stood...
UPRIGHT_DEG = 30.0      # ...and tilted less than this (a mug knocked onto its side also rises ~3-4 cm)
POUR_STEPS = 600        # control steps the pour may take once carried (30 s)
SETTLE_STEPS = 20       # steps run once the mug is empty, for the drops to land (1 s)
PRE_STEPS = 500         # not carried by then: not a pour trial
MAX_STEPS = ev.MAX_STEPS

_act, _trial = cheat.act, cheat.trial


def carried(e):
    """The mug up and upright: held, not knocked over."""
    pos, rot, _ = ev.mug_pose(e)
    tilt = np.degrees(np.arccos(np.clip(rot.apply([0, 0, 1])[2], -1, 1)))
    return pos[2] - cheat._run["start"][2] > CARRIED_CM / 100 and tilt < UPRIGHT_DEG


def act(model, frame, task):
    """cheat.act, then end the trial once the pour is over (or there is none)."""
    chunk = _act(model, frame, task)
    run, e = cheat._run, cheat._run["env"].env
    t, s = e.timestep, run["log"][-1]
    if run["lifted"]:
        run.setdefault("carried", t)
        if s["in_mug"] == 0:
            run.setdefault("empty", t)
    on_table = ev.mug_pose(e)[0][2] - run["start"][2] < 0.01
    end = ("carried" not in run and (s["tilt"] > ev.TOPPLED_DEG or t > PRE_STEPS) and "no pour"  # knocked over / never held
           or "carried" in run and on_table and "mug back on the table"   # dropped, or set down
           or "carried" in run and t - run["carried"] > POUR_STEPS and "time"
           or "empty" in run and t - run["empty"] >= SETTLE_STEPS and "poured")
    if end:
        run["ended"] = end
        ev.MAX_STEPS = t  # evaluate.trial's loop reads it every step: it stops after this one
    return chunk


def trial(*args, **kwargs):
    """cheat.trial, plus when the mug was carried (control step, None: not a pour trial) and why the trial ended."""
    try:
        r = _trial(*args, **kwargs)
    finally:
        ev.MAX_STEPS = MAX_STEPS
    return dict(r, carried=cheat._run.get("carried"), ended=cheat._run.get("ended", "max steps"))


def reset(env, spot, rng, jitter=ev.MUG_JITTER):
    obs = _reset(env, spot, rng, jitter)
    cheat._run.pop("carried", None), cheat._run.pop("empty", None), cheat._run.pop("ended", None)
    return obs


_reset = cheat.reset
# at import, so the spawned eval workers get them too
cheat.carried, cheat.trial = carried, trial
ev.act, ev.reset = act, reset
cheat.RESULTS = ev.RESULTS = cheat.RESULTS.parent / f"pour-{cheat.ARM}"
ev.UNITS = [f"{p}-{s}-{t}" for p in ev.PLANTS for s in cheat.SPOTS for t in range(TRIALS)]

if __name__ == "__main__":
    ev.main()
