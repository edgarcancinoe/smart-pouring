"""Grasp bank: the plain policy grasps and lifts the mug, and the sim is saved at the replan where the pour benchmark
hands over to the arm under test. Any later run can then pour from these exact states: no grasp phase, no wasted
non-carried trial, and every arm pours from the identical state.

Per carried trial -> results/grasps-<bank>/<model>/<unit>.pkl (see save()), beside cheat's <unit>.json (the outcome;
a unit with a json and no pkl was not carried). Each grasp is reloaded into the env right after, and a fixed chunk
run from both the in-memory state and the reloaded file: reload_mm (gripper or any droplet, 0 = exact) in the pkl and
on the unit's line.

Two banks (GRASPS):
  bench (default): the pour benchmark's trials, same names and seeds (SPOTS, SEED as pour-*), pairs with its tables
                   -> grasps[-s<seed>][-spots<spots>]
  train:           random training-style starts (as dsrl.py's), new seeds -> grasps-train

Usage: ./run grasps [--jobs 22]                   bench bank, spots 5-8, seed set 0
       SEED=10000 ./run grasps [--jobs 22]        another seed set
       GRASPS=train ./run grasps [--jobs 22]      training bank
"""
import os
import pickle

import mujoco
import numpy as np
import torch

import pour  # noqa: F401 (patches evaluate and cheat: pour-only trials, old rules by default); this file patches on top
import cheat
import eval_grasp_position as egp
import evaluate as ev
import grasp_position as gp

BANK = os.environ.get("GRASPS", "bench")  # env var so the workers see it
TRAIN_SEED = 100_000    # train bank seeds: none of eval (< 30_000), judge, DSRL training (40_000-65_000) or starts (70_000+)
TRAIN_TRIALS = 900      # train bank trials per plant (~75% carried: ~2,000 grasps)
AREA = ((-24.0, 14.0), (27.0, 30.0))  # cm: random mug starts, as dsrl.py (importing it would patch evaluate)
KEEP_OUT = 4.0          # cm from spots 5-8, the benchmark's unseen spots
CHECK_STEPS = ev.N_ACTION_STEPS  # actions run from the saved and the reloaded state to compare them
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION  # everything mj_step reads: qpos, qvel, act, warm start, ctrl, mocap, time...
EGP_KEYS = ("held", "closing", "width", "start", "obs")  # the grasp memory's per-trial part (not "stats": the model's)

_act, _reset, _run_unit = ev.act, ev.reset, ev.run_unit
_g = {}  # this process's running trial: the saved grasp and the in-memory check


def random_start(rng):
    """A mug start (cm) in AREA, at least KEEP_OUT from spots 5-8 (as dsrl.py)."""
    while True:
        xy = tuple(rng.uniform(*AREA))
        if min(np.hypot(*np.subtract(xy, ev.SPOTS[s])) for s in (5, 6, 7, 8)) > KEEP_OUT:
            return xy


def reset(env, spot, rng, jitter=ev.MUG_JITTER):
    _g.clear()
    if spot == 0:  # evaluate.reset reads the start from SPOTS
        ev.SPOTS[0] = random_start(rng)
    return _reset(env, spot, rng, 0.0 if spot == 0 else jitter)


def fields(e):
    """The Python-side numbers a rewind restores (cheat.snapshot's), by object: env, robot, controller, gripper."""
    keep = lambda o: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(o).items()
                      if isinstance(v, (np.ndarray, int, float, bool, np.generic))}
    r = e.robots[0]
    return [keep(o) for o in (e, r, r.controller, r.gripper)]


def save(e):
    """The state now -> dict: MuJoCo's full integration state, the Python-side fields, the grasp memory, the mug's
    start (pour's carried/on-table read it) and where the gripper and droplets are (for checks)."""
    m, d = e.sim.model._model, e.sim.data._data
    mj = np.empty(mujoco.mj_stateSize(m, SPEC))
    mujoco.mj_getState(m, d, mj, SPEC)
    return dict(mj=mj, fields=fields(e),
                egp={k: egp._last[k] for k in EGP_KEYS}, gp=gp.CURRENT.numpy().copy(),
                mug_start=cheat._run["start"].copy(), step=int(e.timestep),
                ends=np.r_[e.sim.data.site_xpos[e.robots[0].eef_site_id], e.droplet_positions().ravel()])


def load(env, g):
    """Put a saved grasp into env (from scene(plant), after evaluate.reset for that trial); returns its observation."""
    e, r = env.env, env.env.robots[0]
    m, d = e.sim.model._model, e.sim.data._data
    mujoco.mj_setState(m, d, g["mj"], SPEC)
    mujoco.mj_forward(m, d)
    for o, f in zip((e, r, r.controller, r.gripper), g["fields"]):
        for k, v in f.items():
            setattr(o, k, v.copy() if isinstance(v, np.ndarray) else v)
    egp._last.update({k: g["egp"][k] for k in EGP_KEYS})
    gp.CURRENT = torch.from_numpy(g["gp"].copy())
    cheat._run.update(start=g["mug_start"].copy(), lifted=False)
    return g["egp"]["obs"]


def run(env, chunk):
    """Run the chunk's first CHECK_STEPS actions; returns [gripper xyz, droplet xyz...] (m)."""
    for a in chunk[:CHECK_STEPS]:
        env.step(a)
    e = env.env
    return np.r_[e.sim.data.site_xpos[e.robots[0].eef_site_id], e.droplet_positions().ravel()]


def act(model, frame, task):
    """pour's act; at the hand-over replan (mug carried for the first time): save, check the in-memory rewind, end."""
    run_, env = cheat._run, cheat._run["env"]
    e = env.env
    if run_["lifted"] or "grasp" in _g or not cheat.carried(e):
        return _act(model, frame, task)
    g = save(e)
    chunk = _act(model, frame, task)  # the plain policy's chunk here: the check runs it (cameras/policy don't move the sim)
    snap = cheat.snapshot(e)
    g["chunk"], g["check_end"] = chunk, run(env, chunk)
    cheat.restore(e, snap)
    _g["grasp"] = g
    ev.MAX_STEPS = e.timestep  # evaluate.trial's loop reads it every step: it stops after this one (pour.trial resets it)
    return chunk


def run_unit(unit, name, *args, **kwargs):
    """cheat's run_unit, then the grasp (if carried) -> <unit>.pkl, reloaded and checked first."""
    out = _run_unit(unit, name, *args, **kwargs)
    g = _g.get("grasp")
    if g is None:
        return out
    plant, spot, t = unit.split("-")
    env = ev.scene(plant)
    seed = ev.trial_seed(plant, int(spot), int(t))
    _reset(env, int(spot), np.random.default_rng(seed))  # another start first, so nothing carries over by chance
    load(env, g)
    end = run(env, g["chunk"])
    g.update(unit=unit, plant=plant, spot=int(spot), spot_cm=tuple(ev.SPOTS[int(spot)]), seed=seed, model=name,
             reload_mm=round(float(np.abs(end - g["check_end"]).max() * 1000), 3))
    path = cheat.RESULTS / name / f"{unit}.pkl"
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(g))
    tmp.rename(path)  # a killed worker never leaves half a file
    print(f"  {unit}: grasp saved at step {g['step']}, reload_mm {g['reload_mm']}", flush=True)
    return out


# at import, so the spawned eval workers get them too
ev.act, ev.reset, ev.run_unit = act, reset, run_unit
if BANK == "train":
    ev.trial_seed = lambda plant, spot, t: TRAIN_SEED + 10_000 * ev.PLANTS.index(plant) + t
    ev.UNITS = [f"{p}-0-{t}" for t in range(TRAIN_TRIALS) for p in ev.PLANTS]
    cheat.RESULTS = ev.RESULTS = pour.RESULTS_ROOT / "grasps-train"
else:
    cheat.RESULTS = ev.RESULTS = pour.RESULTS_ROOT / f"grasps{pour.tag()}"

if __name__ == "__main__":
    assert cheat.ARM == "plain", "the bank is the plain policy's grasps"
    ev.main()
