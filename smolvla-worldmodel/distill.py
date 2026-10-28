"""Control B, the cheat distilled: the cheat pours at the trained spots 1-4 (other seeds than the eval), each pour is
recorded from the carry on like a DAgger takeover, the watered ones are added to dagger1's data and dagger1 trains on.
Then the pour-only benchmark on spots 5-8: near the cheat's 40/41 means no judge or world model is needed at run time.

Usage: ./run distill [--jobs 16]   the cheat's pours, SPOTS x TRIALS -> dataset/data/cheat/<clip>/ + results/distill/
       ./run distill-export        the watered pours -> dataset -> merged with dagger1's -> Hub (then ./run train MODEL)
       ./run pour-distilled        the pour-only benchmark of MODEL (plain policy) -> results/pour-plain/MODEL/
"""
import json
import sys

import numpy as np

import cheat
import pour  # noqa: F401 (at import: the pour-only benchmark, the cheat's chunk choice; patched further below)
import dagger
import evaluate as ev
from dataset import episodes
from dataset.export_lerobot import export
from dataset.sim_runs import playback
from dataset.upload import upload
from sim.watering import CONTROL_HZ

MODEL = "libero-4s-dart-full-xxl-sd-rel-b64-dagger1-cheat"  # runs.MODELS: dagger1 trained on with these pours
SPOTS = (1, 2, 3, 4)    # trained spots only: the eval's new spots 5-8 stay unseen
TRIALS = 10             # per plant x spot
MUG_JITTER = 0.03       # m, as DAgger's rollouts (eval: 0.01)
SEED = 10_000           # added to evaluate.trial_seed, as DAgger: never an eval trial's seed
NAME = "water_pour_cheat"  # the pours alone: local dataset folder and Hub dataset
OUT = episodes.DATA / "cheat"

_reset, _trial, _run_unit, _seed = ev.reset, cheat.trial, ev.run_unit, ev.trial_seed
_rec = {}  # this process's recording: unit, Recorder (from the carry on), grasp memory per step


def reset(env, spot, rng, jitter=MUG_JITTER):
    """pour's reset with DAgger's jitter; records env.step from the carry on (not the cheat's look-aheads)."""
    obs = _reset(env, spot, rng, MUG_JITTER)
    if not hasattr(env, "_step"):
        env._step = env.step

        def step(a):
            looking = cheat._run.get("looking")
            if _rec.get("rec") is None and cheat._run.get("carried") is not None and not looking:  # carried: set by pour.act
                _rec.update(rec=playback.Recorder(env.env), memory=[])
            recording = _rec.get("rec") is not None and not looking
            if recording:
                _rec["memory"].append(dagger.grasp_memory())  # before the step, as DAgger
            out = env._step(a)
            if recording:
                _rec["rec"].add(a, "pour")
            return out
        env.step = step
    _rec.update(rec=None, memory=[])
    return obs


def trial(*args, **kwargs):
    """pour's trial; the recorded pour -> data/cheat/cheat-<unit>/ (robot.json success: watered)."""
    r = _trial(*args, **kwargs)
    if _rec["rec"] is not None and len(_rec["memory"]) >= 2:
        name = f"cheat-{_rec['unit']}"
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "robot.json").write_text(json.dumps(dict(clip=name, plant=args[2], success=r["watered"]), indent=2))
        np.save(OUT / name / "grasp_memory.npy", np.array(_rec["memory"], np.float32))
        _rec["rec"].save(name, args[2], 1.0, CONTROL_HZ)
    return r


def run_unit(unit, *args, **kwargs):
    _rec["unit"] = unit
    return _run_unit(unit, *args, **kwargs)


def export_all():
    """Watered pours -> dataset NAME -> Hub; merged with dagger1's data -> MODEL's data -> Hub."""
    clips = sorted(d.name for d in OUT.glob("cheat-*") if (d / playback.FILE).exists())
    export(clips, NAME, memory=dagger.memory_column)  # keeps robot.json success only
    upload(NAME, f"edgarcancinoe/{NAME}")
    dagger.MERGE, dagger.MERGED = ("water_pour_dagger1", NAME), MODEL
    dagger.merge()


# at import, so the spawned eval workers get them too
ev.reset, cheat.trial, ev.run_unit = reset, trial, run_unit
ev.trial_seed = lambda plant, spot, t: SEED + _seed(plant, spot, t)
cheat.RESULTS = ev.RESULTS = cheat.RESULTS.parent / "distill"
ev.UNITS = [f"{p}-{s}-{t}" for p in ev.PLANTS for s in SPOTS for t in range(TRIALS)]

if __name__ == "__main__":
    if "--export" in sys.argv:
        export_all()
    else:
        ev.main()
