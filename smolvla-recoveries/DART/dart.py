"""DART: each phone recording replayed RUNS more times with the hand pushed off its path (PHASES),
recording the script's clean (correcting) action -> data/dart/<clip>~d<i>/. --export: the saved runs -> the DART datasets (EXPORTS). See README.md.

Usage: ./run dart [A1-0 ...] [--jobs 4]   (default: every tracked episode)
       ./run dart-export
"""
import argparse
import zlib

import numpy as np

from dataset import episodes
from dataset.export_lerobot import export
from dataset.retarget import robot_replay
from dataset.retarget.robot_replay import MAX_STEP
from dataset.sim_runs import playback
from dataset.upload import upload

RUNS = 32           # noisy replays per recording (<clip>~d1 ... ~d32), each with its own mug shift as well
NOISE_CM = 1.5      # std of the push added to each step's position command, cm
NOISE_DEG = 3.0     # ... to its rotation command, deg
PHASES = ("reach", "follow")  # pushed: the way to the mug and the carry, pour and put-back (xl: reach only, 1 cm, 2 deg, 16 runs)
RHO = 0.9           # push correlation from step to step (~0.5 s at 20 Hz): slow drifts, like a policy's errors
# datasets --export makes -> DART runs per recording each takes (~d1 ... ~d<n>), in dataset/data/lerobot/ and on the Hub
EXPORTS = {"water_pour_dart_full-xl": 16, "water_pour_dart_full-xxl": RUNS}

SIGMA = np.r_[[NOISE_CM / 100] * 3, [np.radians(NOISE_DEG)] * 3] / MAX_STEP  # push std in action units
_state = {}  # per clip: random generator and the current push


def perturb(action, phase, steps):
    """robot_replay.perturb: the clean action plus a slowly drifting push, on the scripted lines of the reach only.

    action -> (7,) clean action
    phase  -> step label
    steps  -> step budget of the drive() call (1: a step along a scripted line)
    """
    if phase not in PHASES or steps != 1:  # the final line-up, the grasp and the release stay clean: no demo closes the gripper off-centre
        return action
    push = _state["push"] = RHO * _state["push"] + np.sqrt(1 - RHO ** 2) * _state["rng"].normal(0, SIGMA)
    return np.r_[np.clip(action[:6] + push, -1, 1), action[6]]


robot_replay.perturb = perturb  # at import, so the spawned workers get it too


def run(clip):
    """One noisy replay (robot_replay.run) in data/dart/<clip>/.

    clip -> e.g. "A1-0~d1"
    returns robot.json dict
    """
    (episodes.DATA / "dart" / clip).mkdir(parents=True, exist_ok=True)  # robot_replay writes into the existing folder
    _state.update(rng=np.random.default_rng(zlib.crc32(clip.encode())), push=np.zeros(6))  # same pushes on a re-run
    return robot_replay.run(clip)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    p.add_argument("--jobs", type=int, default=0, help="clips run in parallel (default: up to half the CPU cores)")
    p.add_argument("--export", action="store_true", help="export + upload (passed by ./run dart-export)")
    args = p.parse_args()
    if args.export:
        for name, n in EXPORTS.items():  # the recordings, their shifted replays and DART runs 1 ... n
            export([c for c in playback.saved() if not episodes.parse(c)["dart"] or episodes.parse(c)["shift"] <= n], name)
            upload(name, f"edgarcancinoe/{name}")
        return
    clips = [f"{c}~d{i}" for c in args.clips or episodes.tracked() for i in range(1, RUNS + 1)
             if not (episodes.DATA / "dart" / f"{c}~d{i}" / "robot.json").exists()]  # done before (same seed): skip
    done = 0
    for r in episodes.parallel_map(run, clips, {}, args.jobs, "dart"):
        if "error" in r:
            print(f"{r['clip']}: {r['error']}")
        done += bool(r.get("success"))
    print(f"\n{done}/{len(clips)} noisy replays succeeded (only those are exported)")


if __name__ == "__main__":
    main()
