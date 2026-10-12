"""DART: each phone recording replayed RUNS more times with the hand pushed off its path on the way to the mug,
recording the script's clean (correcting) action -> data/dart/<clip>~d<i>/. --export: every saved run -> the DART dataset. See README.md.

Usage: ./run dart [A1-0 ...] [--jobs 4]   (default: every tracked episode)
       ./run dart-export
"""
import argparse
import zlib

import numpy as np

import runs
from dataset import episodes
from dataset.export_lerobot import export
from dataset.retarget import robot_replay
from dataset.retarget.robot_replay import MAX_STEP
from dataset.sim_runs import playback
from dataset.upload import upload

RUNS = 2            # noisy replays per recording (<clip>~d1, ~d2), each with its own mug shift as well
NOISE_CM = 1.0      # std of the push added to each step's position command, cm
NOISE_DEG = 2.0     # ... to its rotation command, deg
RHO = 0.9           # push correlation from step to step (~0.5 s at 20 Hz): slow drifts, like a policy's errors
NAME = "water_pour_dart"  # local dataset folder (dataset/data/lerobot/<NAME>); Hub repo: runs.DART

SIGMA = np.r_[[NOISE_CM / 100] * 3, [np.radians(NOISE_DEG)] * 3] / MAX_STEP  # push std in action units
_state = {}  # per clip: random generator and the current push


def perturb(action, phase, steps):
    """robot_replay.perturb: the clean action plus a slowly drifting push, on the scripted lines of the reach only.

    action -> (7,) clean action
    phase  -> step label
    steps  -> step budget of the drive() call (1: a step along a scripted line)
    """
    if phase != "reach" or steps != 1:  # the final line-up and the grasp stay clean: no demo closes the gripper off-centre
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
        export(playback.saved(), NAME)  # the recordings, their shifted replays and the DART replays
        upload(NAME, runs.DART)
        return
    clips = [f"{c}~d{i}" for c in args.clips or episodes.tracked() for i in range(1, RUNS + 1)]
    done = 0
    for r in episodes.parallel_map(run, clips, {}, args.jobs, "dart"):
        if "error" in r:
            print(f"{r['clip']}: {r['error']}")
        done += bool(r.get("success"))
    print(f"\n{done}/{len(clips)} noisy replays succeeded (only those are exported)")


if __name__ == "__main__":
    main()
