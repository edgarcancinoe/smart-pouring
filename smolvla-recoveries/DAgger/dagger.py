"""Takeover DAgger: the fine-tuned POLICY reaches for the mug; once its hand is OFF_CM off the recipe's path the
teacher (teacher.py) drives until it is back, and each takeover is saved as a short demo -> data/dagger/<name>/.
See README.md.

Usage: ./run dagger [--jobs 8]         rollouts of POLICY for round ROUND
       ./run dagger-check [--jobs 8]   the teacher alone, from pushed-off starts: must lift the mug ~every time
       ./run dagger-export             the demos + every takeover up to ROUND -> dataset -> Hub
"""
import argparse
import json

import numpy as np

import evaluate as ev
import runs
from dataset import episodes
from dataset.export_lerobot import export, observe
from dataset.retarget.robot_replay import MAX_STEP
from dataset.sim_runs import playback
from dataset.upload import upload
from sim.watering import CONTROL_HZ
from teacher import CLOSE_STEPS, Teacher

ROUND = 1                           # DAgger round
MODEL = f"libero-4s-dagger{ROUND}"  # model trained on this round's data (runs.MODELS: its "data" is the Hub repo)
POLICY = runs.MODELS[MODEL]["start"]  # model rolled out this round: the one MODEL continues (libero-4s; round 2: libero-4s-dagger1)
WITH_DART = False                   # also export the DART replays (if POLICY was trained on them)
SPOTS = (1, 2, 3, 4)                # trained spots only: eval's new spots 5-8 stay unseen
ROLLOUTS = 5                        # per plant x spot
MUG_JITTER = 0.03                   # m, +- around the spot, as the shifted demos (eval: 0.01)
SEED = 10_000                       # added to evaluate.trial_seed: never the seed of an eval trial (< 3000)
OFF_CM = 1.0                        # the teacher takes over when the hand is this far off the recipe's path
BACK_CM = 0.3                       # ... and hands back once within this
MIN_TAKEOVER = 10                   # control steps a takeover lasts at least (no flicker between drivers)
CHECK_PUSH_CM = 3.0                 # dagger-check: how far the hand wanders off the start pose before the teacher drives, cm
CHECK_PUSH_STEPS = 40               # ... over this many control steps
LIFT = np.r_[0, 0, 0.5, 0, 0, 0, 1]  # dagger-check: straight up with the gripper closed, once the teacher has closed it
NAME = f"water_pour_dagger{ROUND}"  # local dataset folder (dataset/data/lerobot/<NAME>)
OUT = episodes.DATA / "dagger"


def start(job):
    """A rollout's scene, as an eval trial starts but with its own seed and MUG_JITTER, and its teacher.

    job -> "<plant>-<spot>-<i>", e.g. "left-2-0"
    returns plant, env, first observation, mug start position (3,) m, Teacher, seeded random generator
    """
    plant, spot, i = job.split("-")
    spot, i = int(spot), int(i)
    rng = np.random.default_rng(SEED + ev.trial_seed(plant, spot, i))
    env = ev.scene(plant)
    obs = ev.reset(env, spot, rng, MUG_JITTER)
    mug, *_ = ev.mug_pose(env.env)
    return plant, env, obs, mug, Teacher(env.env, plant, spot), rng


def lifted(e, mug):
    """The mug is LIFT_CM above where it started.

    e   -> the LIBERO problem (env.env)
    mug -> (3,) m, the mug's start position
    """
    return ev.mug_pose(e)[0][2] - mug[2] > ev.LIFT_CM / 100


def save(takeover, name, plant, recovered):
    """One takeover -> data/dagger/<name>/ (robot_states.npz + robot.json; export keeps successful ones only).

    takeover  -> playback.Recorder started when the teacher took over
    name      -> e.g. "r1-left-2-0-t1": round, plant, spot, rollout, takeover
    plant     -> plant name (the command)
    recovered -> the teacher got the hand back on the path (or the mug lifted) before the rollout ended
    """
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "robot.json").write_text(json.dumps(dict(clip=name, plant=plant, success=recovered), indent=2))
    takeover.save(name, plant, 1.0, CONTROL_HZ)


def rollout(job):
    """The policy drives; whenever its hand is OFF_CM off the path the teacher takes over until it is back, then the
    policy goes on. Each takeover is saved; the policy's own steps are not.

    job -> "<plant>-<spot>-<i>", e.g. "left-2-0"
    returns {clip, lifted, takeovers}
    """
    plant, env, obs, mug, teacher, _ = start(job)
    e = env.env
    model = ev.load_model(POLICY)
    model[0].reset()
    queue = []       # policy actions still to run from its last plan
    takeover = None  # Recorder while the teacher drives, None while the policy does
    took = n = 0     # step the current takeover began; takeovers so far
    up = False
    for k in range(ev.GRASP_STEPS):  # one control step (1/20 s) per loop, at most 15 s
        # who drives this step
        off = teacher.off_track_cm(obs)  # cm between the hand and the recipe's path, now

        # policy driving and too far off: the teacher takes the wheel
        if takeover is None and off > OFF_CM:
            takeover, took = playback.Recorder(e), k  # recording starts from the state the policy left

        # teacher driving, for long enough, and the hand is back on the path: the policy takes the wheel again
        elif takeover is not None and k - took >= MIN_TAKEOVER and off < BACK_CM and not teacher.busy:
            save(takeover, f"r{ROUND}-{job}-t{n}", plant, True)  # this takeover -> its own short demo (t0, t1, ...)
            takeover, queue, n = None, [], n + 1  # the policy plans again from where the teacher left it

        # otherwise the same driver goes on (BACK_CM < OFF_CM: a hand between the two never flips drivers)

        # one control step
        if takeover is not None:
            a, phase = teacher.act(obs)
            obs, *_ = env.step(a)
            takeover.add(a, phase)
        else:
            if not queue:
                queue = list(ev.act(model, observe(e), f"Water the {plant} plant"))
            obs, *_ = env.step(queue.pop(0))

        # done once the mug is lifted or knocked over
        up = lifted(e, mug)
        if up or ev.mug_pose(e)[1].apply([0, 0, 1])[2] < np.cos(np.radians(ev.TOPPLED_DEG)):
            break

    if takeover is not None:  # time ran out (or the mug went up) with the teacher driving
        save(takeover, f"r{ROUND}-{job}-t{n}", plant, up)
        n += 1
    return dict(clip=job, lifted=up, takeovers=n)


def check(job):
    """The teacher alone: the hand first wanders off the start pose at random, then the teacher grasps and lifts.

    job -> "<plant>-<spot>-<i>"
    returns {clip, lifted}
    """
    _, env, obs, mug, teacher, rng = start(job)
    # random walk of CHECK_PUSH_STEPS steps, ~CHECK_PUSH_CM in all (position only, gripper open)
    push = np.r_[[CHECK_PUSH_CM / 100] * 3, [0.0] * 3] / MAX_STEP / np.sqrt(CHECK_PUSH_STEPS)
    for _ in range(CHECK_PUSH_STEPS):
        obs, *_ = env.step(np.r_[np.clip(rng.normal(0, push), -1, 1), -1])
    for _ in range(ev.GRASP_STEPS):
        a, _ = teacher.act(obs)
        obs, *_ = env.step(LIFT if teacher.closed >= CLOSE_STEPS else a)  # the teacher stops at the grasp
        if lifted(env.env, mug):
            return dict(clip=job, lifted=True)
    return dict(clip=job, lifted=False)


def dagger_clips():
    """Saved takeovers of rounds 1..ROUND, in order."""
    return sorted(d.name for d in OUT.glob("r*") if (d / playback.FILE).exists() and int(d.name.split("-")[0][1:]) <= ROUND)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=0, help="rollouts run in parallel (default: up to half the CPU cores)")
    p.add_argument("--check", action="store_true", help="the teacher alone (passed by ./run dagger-check)")
    p.add_argument("--export", action="store_true", help="export + upload (passed by ./run dagger-export)")
    args = p.parse_args()

    if args.export:
        demos = [c for c in playback.saved() if WITH_DART or not episodes.parse(c)["dart"]]
        export(demos + dagger_clips(), NAME)
        upload(NAME, runs.MODELS[MODEL]["data"])
        return

    jobs = [f"{plant}-{spot}-{i}" for plant in ev.PLANTS for spot in SPOTS for i in range(ROLLOUTS)]
    results = list(episodes.parallel_map(check if args.check else rollout, jobs, {}, args.jobs, "dagger"))
    for r in results:
        if "error" in r:
            print(f"{r['clip']}: {r['error']}")
    up = sum(bool(r.get("lifted")) for r in results)
    print(f"\n{up}/{len(jobs)} lifted" + ("" if args.check else
          f", {sum(r.get('takeovers', 0) for r in results)} takeovers saved in {OUT}"))


if __name__ == "__main__":
    main()
