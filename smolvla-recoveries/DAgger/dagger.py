"""Takeover DAgger: POLICY drives, teacher.py takes over when it's off the path (GATE); takeovers -> data/dagger/. See README.md.

Usage: ./run dagger [--jobs 8]         rollouts of POLICY for round ROUND
       ./run dagger-check [--jobs 8]   the teacher alone, from pushed-off starts: must water and put back ~every time
       ./run dagger-export             the demos + every takeover up to ROUND -> dataset (with grasp_memory) -> Hub
"""
import argparse
import json

import numpy as np
from lerobot.datasets import LeRobotDatasetMetadata

import eval_relative  # noqa: F401 (at import: evaluate's scene, reset and load_model keep the grasp memory)
import eval_grasp_position as egp
import evaluate as ev
import grasp_position as gp
import runs
from dataset import episodes
from dataset.export_lerobot import export, observe
from dataset.retarget.robot_replay import MAX_STEP
from dataset.sim_runs import playback
from dataset.upload import upload
from sim.watering import CONTROL_HZ
from teacher import Teacher

ROUND = 1                           # DAgger round
MODEL = f"libero-4s-dart-full-anchor-xl-sd-rel-b64-dagger{ROUND}"  # trained on this round's data (runs.MODELS: its "data" is the Hub repo)
POLICY = runs.MODELS[MODEL]["start"]  # model rolled out this round: the one MODEL continues
WITH_DART = False                   # also export the DART replays (if POLICY was trained on them)
SPOTS = (1, 2, 3, 4)                # trained spots only: eval's new spots 5-8 stay unseen
ROLLOUTS = 5                        # per plant x spot
MUG_JITTER = 0.03                   # m, +- around the spot, as the shifted demos (eval: 0.01)
SEED = 10_000                       # added to evaluate.trial_seed: never the seed of an eval trial (< 3000)
GATE = {"reach": (1.0, 0.3),        # per teacher stage: cm off its path at which the teacher takes over,
        "carry": (2.0, 0.5),        # ... and back within which it hands back (the pour moves fast: looser)
        "put-back": (1.0, 0.3),     # the carry's last 5 cm (teacher.PUT_BACK_CM): placed means within 3 cm
        "release": (2.0, 0.5)}
MIN_TAKEOVER = 10                   # control steps a takeover lasts at least (no flicker between drivers)
CHECK_PUSH_CM = 3.0                 # dagger-check: how far the hand wanders off the start pose before the teacher drives, cm
CHECK_PUSH_STEPS = 40               # ... over this many control steps
NAME = f"water_pour_anchor_dagger{ROUND}"  # local dataset folder (dataset/data/lerobot/<NAME>)
OUT = episodes.DATA / "dagger-anchor"  # this run's takeovers (the xxl run's are in dagger/)


def start(job):
    """Scene + teacher for one rollout.

    job -> "left-2-0": plant, spot, rollout
    returns plant, env, obs, mug start, teacher, rng
    """
    plant, spot, i = job.split("-")
    spot, i = int(spot), int(i)
    rng = np.random.default_rng(SEED + ev.trial_seed(plant, spot, i))
    if egp._last["stats"] is None:  # the grasp memory's scale, as egp.load_model sets it (dagger-check loads no policy)
        egp._last["stats"] = LeRobotDatasetMetadata(runs.MODELS[POLICY]["data"]).stats["observation.state"]
    env = ev.scene(plant)
    obs = ev.reset(env, spot, rng, MUG_JITTER)
    mug, *_ = ev.mug_pose(env.env)
    return plant, env, obs, mug, Teacher(env.env, plant, spot), rng


def ended(e, obs, mug, end):
    """evaluate.trial's end rule: mug handled, released and resting.

    end -> state dict kept between steps
    """
    pos, rot, speed = ev.mug_pose(e)
    end["lifted"] = end.get("lifted", False) or pos[2] - mug[2] > ev.LIFT_CM / 100
    end["handled"] = end.get("handled", False) or end["lifted"] or rot.apply([0, 0, 1])[2] < np.cos(np.radians(ev.TOPPLED_DEG))
    released = np.linalg.norm(obs["robot0_eef_pos"] - pos) > ev.RELEASE_CM / 100
    end["rest"] = end.get("rest", 0) + 1 if end["handled"] and released and speed < 0.01 else 0
    return end["rest"] >= ev.REST_S * CONTROL_HZ


def outcome(e, obs, mug, end):
    """lifted, watered, placed (as evaluate.trial) + why not placed: released, off_cm, tilt."""
    pos, rot, _ = ev.mug_pose(e)
    tilt = float(np.degrees(np.arccos(np.clip(rot.apply([0, 0, 1])[2], -1, 1))))
    released = bool(np.linalg.norm(obs["robot0_eef_pos"] - pos) > ev.RELEASE_CM / 100)
    off_cm = float(np.linalg.norm(pos[:2] - mug[:2]) * 100)
    placed = released and off_cm <= ev.PLACED_CM and tilt <= ev.PLACED_DEG
    return dict(lifted=bool(end.get("lifted")), watered=bool(e._check_success()), placed=placed,
                released=released, off_cm=round(off_cm, 1), tilt=round(tilt))


def grasp_memory():
    """Policy's grasp memory now: [held, x, y] m."""
    held, x, y = gp.CURRENT.tolist()
    if not held:
        return np.zeros(3, np.float32)
    stats = egp._last["stats"]
    return np.r_[1.0, np.array([x, y]) * stats["std"][:2] + stats["mean"][:2]].astype(np.float32)


def save(takeover, memory, name, plant, recovered):
    """One takeover -> data/dagger/<name>/.

    takeover  -> Recorder
    memory    -> grasp memory per step
    name      -> "r1-left-2-0-t1"
    recovered -> back on the path (or finished the task): export keeps it
    """
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "robot.json").write_text(json.dumps(dict(clip=name, plant=plant, success=recovered), indent=2))
    np.save(OUT / name / "grasp_memory.npy", np.array(memory, np.float32))
    takeover.save(name, plant, 1.0, CONTROL_HZ)


def rollout(job):
    """One rollout: policy drives, teacher takes over when off the path. Only takeovers are saved.

    job -> "left-2-0"
    """
    plant, env, obs, mug, teacher, _ = start(job)
    e = env.env
    model = ev.load_model(POLICY)
    model[0].reset()
    queue = []       # policy actions still to run from its last plan
    takeover = None  # Recorder while the teacher drives, None while the policy does
    memory = []      # the grasp memory of each recorded step
    took = n = 0     # step the current takeover began; takeovers so far
    end = {}
    for k in range(ev.MAX_STEPS):  # one control step (1/20 s) per loop
        # who drives this step
        teacher.observe(obs)               # the teacher follows the rollout whoever drives: stage, carry progress
        off = teacher.off_track_cm(obs)    # cm between the hand and the teacher's path, now
        take, back = GATE[teacher.gate]

        # policy driving and too far off: the teacher takes the wheel
        if takeover is None and off > take:
            takeover, memory, took = playback.Recorder(e), [], k  # recording starts from the state the policy left

        # teacher driving, for long enough, and the hand is back on the path: the policy takes the wheel again
        elif takeover is not None and k - took >= MIN_TAKEOVER and off < back and not teacher.busy:
            save(takeover, memory, f"r{ROUND}-{job}-t{n}", plant, True)  # this takeover -> its own short demo (t0, t1, ...)
            takeover, queue, n = None, [], n + 1  # the policy plans again from where the teacher left it

        # otherwise the same driver goes on (back < take: a hand between the two never flips drivers)

        # one control step
        if takeover is not None:
            a, phase = teacher.act(obs)
            memory.append(grasp_memory())  # before the step: what the memory held when this action was chosen
            obs, *_ = env.step(a)
            takeover.add(a, phase)
        else:
            if not queue:
                queue = list(ev.act(model, observe(e), f"Water the {plant} plant"))
            obs, *_ = env.step(queue.pop(0))

        if ended(e, obs, mug, end):
            break

    out = outcome(e, obs, mug, end)
    if takeover is not None:  # the rollout ended with the teacher driving
        save(takeover, memory, f"r{ROUND}-{job}-t{n}", plant, out["watered"] and out["placed"])
        n += 1
    return dict(clip=job, **out, steps=k + 1, takeovers=n)


def check(job):
    """Teacher alone, whole task, after a random wander off the start pose.

    job -> "left-2-0"
    """
    _, env, obs, mug, teacher, rng = start(job)
    # random walk of CHECK_PUSH_STEPS steps, ~CHECK_PUSH_CM in all (position only, gripper open)
    push = np.r_[[CHECK_PUSH_CM / 100] * 3, [0.0] * 3] / MAX_STEP / np.sqrt(CHECK_PUSH_STEPS)
    for _ in range(CHECK_PUSH_STEPS):
        obs, *_ = env.step(np.r_[np.clip(rng.normal(0, push), -1, 1), -1])
    end = {}
    for k in range(ev.MAX_STEPS):
        teacher.observe(obs)
        obs, *_ = env.step(teacher.act(obs)[0])
        if ended(env.env, obs, mug, end):
            break
    return dict(clip=job, **outcome(env.env, obs, mug, end), steps=k + 1)


def memory_column(clip, actions, states):
    """grasp_memory column for one episode: the takeover's saved one, else derived like training does."""
    path = episodes.clip_dir(clip) / "grasp_memory.npy"
    if path.exists():
        return np.load(path)
    return gp.grasp_positions(actions[:, 6] > 0, states[:, :2], states[:, 6] - states[:, 7])


def dagger_clips():
    """Saved takeovers up to ROUND."""
    return sorted(d.name for d in OUT.glob("r*") if (d / playback.FILE).exists() and int(d.name.split("-")[0][1:]) <= ROUND)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=0, help="rollouts run in parallel (default: up to half the CPU cores)")
    p.add_argument("--check", action="store_true", help="the teacher alone (passed by ./run dagger-check)")
    p.add_argument("--export", action="store_true", help="export + upload (passed by ./run dagger-export)")
    args = p.parse_args()

    if args.export:
        demos = [c for c in playback.saved() if (p := episodes.parse(c)) and (WITH_DART or not p["dart"])]  # takeovers: dagger_clips()
        export(demos + dagger_clips(), NAME, memory=memory_column)
        upload(NAME, runs.MODELS[MODEL]["data"])
        return

    jobs = [f"{plant}-{spot}-{i}" for plant in ev.PLANTS for spot in SPOTS for i in range(ROLLOUTS)]
    results = list(episodes.parallel_map(check if args.check else rollout, jobs, {}, args.jobs, "dagger"))
    for r in results:
        if "error" in r:
            print(f"{r['clip']}: {r['error']}")
        elif not r["placed"]:  # why: still held (out of time?), too far, or tipped
            print(f"{r['clip']}: not placed: released {r['released']}, {r['off_cm']} cm off, {r['tilt']} deg, {r['steps']} steps")
    count = lambda key: sum(bool(r.get(key)) for r in results)
    print(f"\nlifted {count('lifted')}, watered {count('watered')}, placed {count('placed')} of {len(jobs)}"
          + ("" if args.check else f"; {sum(r.get('takeovers', 0) for r in results)} takeovers saved in {OUT}"))


if __name__ == "__main__":
    main()
