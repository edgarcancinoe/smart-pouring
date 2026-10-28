"""Cheat test: how much could a judge that picks among SmolVLA's own chunks add? See README.md.

After the lift, at each replan: save the sim, sample K chunks (different noise), run LOOKAHEAD actions of each, score
it with sim state (drops over the target pot, mug mouth's distance to it), rewind, run the best one's first
N_ACTION_STEPS. Until the lift and once the mug is empty: the plain policy. ARM=plain runs the same trials without
the cheat, with the same per-replan log (mug vs pot, tilt, drops) -> results/<arm>/<model>/<plant>-<spot>.json.

Usage: ./run cheat [--jobs 8]         the cheat, MODEL, 3 plants x spots 5-8 x TRIALS
       ./run cheat-plain [--jobs 8]   the same trials, plain policy, logged
"""
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

import eval_relative  # noqa: F401 (at import: evaluate's scene, reset and load_model keep the grasp memory)
import eval_grasp_position as egp
import evaluate as ev
import grasp_position as gp
from dataset.export_lerobot import CAMERAS
from sim.watering.assets import MUG, POTS_DIMS

ARM = os.environ.get("ARM", "cheat")  # "cheat" or "plain" (./run cheat / cheat-plain; env var so the workers see it)
MODEL = "libero-4s-dart-full-xxl-sd-rel-b64-dagger1"
SPOTS = (5, 6, 7, 8)    # unseen spots only
TRIALS = 3              # per plant x spot: 36 in all, the first 3 seeds of the main eval
K = 8                   # chunks sampled per replan
LOOKAHEAD = 20          # actions of each chunk run before scoring it (1 s)
DROP_WEIGHT = 10.0      # score per drop over the target pot (minus per drop out of the mug elsewhere), vs cm of mouth distance
RESULTS = Path(__file__).parent / "results" / ARM

_act, _reset, _trial = ev.act, ev.reset, ev.trial
_run = {}  # this process's running trial: env, mug start, lifted, log


def snapshot(e):
    """Everything a rewind restores: MuJoCo data, the robot's and controller's arrays/numbers, the grasp memory."""
    if "data" not in _run:
        _run["data"] = mujoco.MjData(e.sim.model._model)
    mujoco.mj_copyData(_run["data"], e.sim.model._model, e.sim.data._data)
    r = e.robots[0]
    keep = lambda o: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(o).items()
                      if isinstance(v, (np.ndarray, int, float, bool, np.generic))}
    return [(o, keep(o)) for o in (e, r, r.controller, r.gripper)], dict(egp._last), gp.CURRENT.clone()


def restore(e, snap):
    objs, last, current = snap
    mujoco.mj_copyData(e.sim.data._data, e.sim.model._model, _run["data"])
    for o, fields in objs:
        for k, v in fields.items():
            setattr(o, k, v.copy() if isinstance(v, np.ndarray) else v)
    egp._last.clear()
    egp._last.update(last)
    gp.CURRENT = current.clone()


def state(e):
    """Mug vs target pot now: mouth offset (cm, xyz), tilt (deg), drops in the mug / over the target pot / elsewhere."""
    pos, rot, _ = ev.mug_pose(e)
    pot = e.sim.data.body_xpos[e.obj_body_id[e.target_pot]]
    dims = POTS_DIMS[e.target_pot.split("_")[-1]]
    mouth = pos + rot.apply([0, 0, MUG["height"]])
    in_mug = e.watering_stats()[e.mug_name]
    drops = e.droplet_positions()
    out = np.ones(len(drops), bool)
    out[np.argsort(np.linalg.norm(drops - pos, axis=1))[:in_mug]] = False  # shortcut: the in_mug nearest the mug are the ones in it
    over = out & (np.linalg.norm(drops[:, :2] - pot[:2], axis=1) < dims.get("r_in_top", dims["r_in"]))  # in the pot or falling into it
    tilt = np.degrees(np.arccos(np.clip(rot.apply([0, 0, 1])[2], -1, 1)))
    return dict(mouth_cm=np.round((mouth - pot) * 100, 1).tolist(), tilt=round(float(tilt), 1),
                in_mug=int(in_mug), over=int(over.sum()), elsewhere=int((out & ~over).sum()))


def score(s):
    """Higher is better: drops over the target pot, minus drops elsewhere, then the mouth's xy distance to the pot (cm)."""
    return DROP_WEIGHT * (s["over"] - s["elsewhere"]) - float(np.linalg.norm(s["mouth_cm"][:2]))


def lookahead(env, chunk):
    """Run the chunk's first LOOKAHEAD actions; returns the state after them and [gripper xyz, droplet xyz...] (m)."""
    for a in chunk[:LOOKAHEAD]:
        obs, *_ = env.step(a)
    return state(env.env), np.r_[obs["robot0_eef_pos"], env.env.droplet_positions().ravel()]


def act(model, frame, task):
    """evaluate.act, logged; after the lift (mug not empty yet) and for ARM=cheat: the best of K chunks."""
    env, e = _run["env"], _run["env"].env
    s = state(e)
    _run["lifted"] |= ev.mug_pose(e)[0][2] - _run["start"][2] > ev.LIFT_CM / 100
    entry = dict(step=int(e.timestep), **s)
    _run["log"].append(entry)
    if ARM != "cheat" or not _run["lifted"] or s["in_mug"] == 0:
        return _act(model, frame, task)

    policy, pre, post = model
    batch = {key: torch.from_numpy(frame[key]).permute(2, 0, 1).float().div(255)[None].repeat(K, 1, 1, 1) for key in CAMERAS}
    batch["observation.state"] = torch.from_numpy(frame["observation.state"])[None].repeat(K, 1)
    batch["task"] = [task] * K
    with torch.inference_mode():
        chunks = post(policy.predict_action_chunk(pre(batch))).cpu().numpy()  # (K, 50, 7), K noise draws

    snap = snapshot(e)
    if "rewind_mm" not in _run:  # once per trial: the same chunk twice from the same state must end the same
        ends = []
        for _ in range(2):
            ends.append(lookahead(env, chunks[0])[1])
            restore(e, snap)
        _run["rewind_mm"] = round(float(np.abs(ends[0] - ends[1]).max() * 1000), 3)  # gripper or any droplet; 0 = exact
    ends, scores = [], []
    for c in chunks:
        s_end, end = lookahead(env, c)
        restore(e, snap)
        ends.append(end[:3])
        scores.append(score(s_end))
    best = int(np.argmax(scores))
    entry.update(scores=np.round(scores, 1).tolist(), pick=best,
                 spread_cm=round(float(np.linalg.norm(np.array(ends) - np.mean(ends, 0), axis=1).mean() * 100), 2))
    return chunks[best, :ev.N_ACTION_STEPS]


def reset(env, spot, rng, jitter=ev.MUG_JITTER):
    """evaluate.reset (with the grasp memory), then a fresh trial log."""
    obs = _reset(env, spot, rng, jitter)
    _run.update(env=env, start=ev.mug_pose(env.env)[0], lifted=False, log=[])
    _run.pop("rewind_mm", None)
    return obs


def trial(*args, **kwargs):
    """evaluate.trial, plus the per-replan log and the rewind check."""
    r = _trial(*args, **kwargs)
    return dict(r, arm=ARM, rewind_mm=_run.get("rewind_mm"), log=_run["log"])


# at import, so the spawned eval workers get them too
ev.act, ev.reset, ev.trial = act, reset, trial
ev.RESULTS, ev.MODELS, ev.TRIALS = RESULTS, [MODEL], TRIALS
ev.UNITS = [f"{p}-{s}" for p in ev.PLANTS for s in SPOTS]
ev.VERBOSE = True  # a line per trial (time per trial: the cheat's cost)

if __name__ == "__main__":
    ev.main()
