"""SmolVLA models drive the Panda in the watering scene: 3 plants x 8 start spots x TRIALS. See README.md.

Writes results/<model>/<plant>-<spot>.json (one dict per trial) and videos/<plant>-<spot>.mp4; results/n<N>/<model>/ for N_ACTION_STEPS != 10.
./run eval-grasp: the same trials, ended once the mug is lifted (or after GRASP_STEPS) -> results/grasp-n<N_ACTION_STEPS>/<model>/.

Usage: ./run eval [libero-0 libero-4 libero-4@10000 ...] [--jobs 2]   (default: MODELS; <model>@<step>: the checkpoint at that step)
       N_ACTION_STEPS=5 ./run eval [...]
       ./run eval-grasp [libero-4 ...] [--jobs 2]
(eval_watch.py: the same trials, live in the viewer)
"""
import argparse
import contextlib
import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import runs
from dataset import episodes
from dataset.export_lerobot import CAMERAS, observe
from dataset.retarget.robot_replay import PLACED_CM, PLACED_DEG
from dataset.tracking.mug_tracker import to_h264
from sim import watering
from sim.watering import CONTROL_HZ
from sim.watering.assets import DROPLET
from sim.watering.bddl import POTS, real_to_sim
from sim.watering.robot import START_Q

PLANTS = ("left", "middle", "right")
# Mug start spots in the real floor frame (cm, as dataset/setup.json): 1-4 trained, 5-8 new.
SPOTS = {**{i + 1: tuple(xy) for i, xy in enumerate(sorted(episodes.SETUP["mug_starts"]))},
         5: (-21.0, 24.0), 6: (-8.0, 16.0), 7: (4.0, 28.0), 8: (24.0, 18.0)}
TRAINED_SPOTS = (1, 2, 3, 4)
# models evaluated when ./run eval names none (<init>-<k>: README.md); one not on the Hub yet is skipped
MODELS = ["libero-4", "base-4", "libero-4s", "base-4s", "libero-4s-b64", "base-4s-b64"]
UNITS = [f"{plant}-{spot}" for plant in PLANTS for spot in SPOTS]  # every plant x spot; or e.g. ["left-1", "right-6"]
TRIALS = 5              # trials per plant x spot (24 x 5 = 120 per model; the first evals ran 7)
SEED = 0                # trial seeds derive from it (mug jitter, policy noise): the same for every model
MUG_JITTER = 0.01       # m, uniform +- around the spot
N_ACTION_STEPS = int(os.environ.get("N_ACTION_STEPS", 10))  # actions run from each predicted chunk (50) before the policy looks again;
                        # env var so two evals with different N can run side by side (the spawned workers re-read it)
MAX_STEPS = 1200        # control steps (60 s; the longest demo is 958)
GRASP_STEPS = 300       # control steps for eval-grasp (15 s; the demos hold the mug by step 96-153)
LIFT_CM = 3.0           # the mug counts as lifted this far above where it stood
TOPPLED_DEG = 45.0      # ... as knocked over tilted this far
RELEASE_CM = 8.0        # ... as released with the gripper point this far from it
REST_S = 1.0            # end the trial once a lifted or knocked-over mug has rested, released, this long
VIDEO_TRIALS = 1        # trials per unit filmed
PRINT_EVERY = 30        # s between the progress lines (trials done, lifted/succeeded so far, time left)
VERBOSE = False         # also print every trial, and a clock line every PROGRESS_EVERY steps of it
PROGRESS_EVERY = 200    # control steps between those clock lines
VIDEO_EVERY = 2         # film every Nth control step
VIDEO_SIZE = 256        # px per camera panel
RESULTS = Path(__file__).parent / "results" / ("" if N_ACTION_STEPS == 10 else f"n{N_ACTION_STEPS}")  # n10: the main results
OPEN = np.r_[np.zeros(6), -1.0]  # no motion, gripper open

_cache = {}  # per process: the loaded model and one env per plant


def load_model(name):
    """Policy and its LeRobot pre/post-processors (normalization, tokenizer, camera renames), on this machine's device.

    name -> model name, e.g. "libero-4"
    returns policy, preprocessor, postprocessor
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if _cache.get("name") != name:
        repo, rev = runs.repo(name)
        cfg = PreTrainedConfig.from_pretrained(repo, revision=rev)
        cfg.device = runs.device()
        policy = SmolVLAPolicy.from_pretrained(repo, config=cfg, revision=rev).eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=repo, pretrained_revision=rev,
                                             preprocessor_overrides={"device_processor": {"device": cfg.device}})
        _cache.update(name=name, model=(policy, pre, post))
    return _cache["model"]


def scene(plant):
    """The watering env for one command, built once per process; the robot starts in START_Q."""
    if plant not in _cache:
        env = watering.make_env(plant, hard_reset=False, initialization_noise=None, use_camera_obs=False,
                                horizon=10 ** 6, ignore_done=True)
        env.env.robots[0].init_qpos = START_Q
        _cache[plant] = env
    return _cache[plant]


def act(model, frame, task):
    """The next N_ACTION_STEPS actions (N, 7) for an observation.

    model -> (policy, preprocessor, postprocessor)
    frame -> dict from observe(): camera images (H, W, 3) uint8, observation.state (8,)
    task  -> command text
    """
    policy, pre, post = model
    batch = {key: torch.from_numpy(frame[key]).permute(2, 0, 1).float().div(255)[None] for key in CAMERAS}
    batch["observation.state"] = torch.from_numpy(frame["observation.state"])[None]
    batch["task"] = [task]
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(batch))
    return post(chunk)[0, :N_ACTION_STEPS].cpu().numpy()


def mug_pose(e):
    """Mug position (3,) m, orientation (Rotation) and speed (m/s)."""
    joint = e.get_object(e.mug_name).joints[-1]
    q = e.sim.data.get_joint_qpos(joint)
    return q[:3].copy(), R.from_quat(q[[4, 5, 6, 3]]), np.linalg.norm(e.sim.data.get_joint_qvel(joint)[:3])


def film(e, caption):
    """One video frame: scene camera | wrist camera, upright, with a caption."""
    imgs = [e.sim.render(camera_name=c, width=VIDEO_SIZE, height=VIDEO_SIZE)[::-1, :, ::-1].copy()
            for c in ("agentview", "robot0_eye_in_hand")]
    for i, line in enumerate(caption):
        cv2.putText(imgs[0], line, (6, 16 + 16 * i), 0, 0.42, (255, 255, 255), 1)
    return np.hstack(imgs)


def reset(env, spot, rng, jitter=MUG_JITTER):
    """A trial's start: robot in START_Q, mug on its spot (+- jitter), droplets settled.

    env    -> env from scene(plant)
    spot   -> mug start spot 1-8
    rng    -> numpy random generator (mug jitter)
    jitter -> m, uniform +- around the spot
    returns the first observation
    """
    e = env.env
    env.reset()
    e.robots[0].controller.update_initial_joints(START_Q)  # the posture the controller relaxes toward
    e.set_layout(np.add(real_to_sim(SPOTS[spot]), rng.uniform(-jitter, jitter, 2)), POTS)

    
    # Safe droplet settling for 200 steps.
    idx, vel = e.robots[0]._ref_joint_pos_indexes, e.robots[0]._ref_joint_vel_indexes
    q_arm = e.sim.data.qpos[idx].copy()
    for _ in range(200):
        e.sim.data.qpos[idx], e.sim.data.qvel[vel] = q_arm, 0
        e.sim.step()
    
    obs, *_ = env.step(OPEN)  # one open-gripper step
    return obs


def trial(env, model, plant, spot, seed, video=None, viewer=None, grasp=False):
    """One episode: place the scene, let the policy act, measure where the water and the mug ended.

    env    -> env from scene(plant)
    model  -> (policy, preprocessor, postprocessor)
    plant  -> "left" / "middle" / "right"
    spot   -> mug start spot 1-8
    seed   -> mug jitter and policy noise
    video  -> mp4 path to film the trial to, or None
    viewer -> MuJoCo passive viewer to show it in, or None
    grasp  -> end once the mug is lifted or knocked over, within GRASP_STEPS
    returns dict of the trial's results (see the module docstring)
    """
    e = env.env
    lock = viewer.lock if viewer else contextlib.nullcontext
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model[0].reset()
    task = f"Water the {plant} plant"
    with lock():
        obs = reset(env, spot, rng)
    start, *_ = mug_pose(e)

    frames, queue = [], []
    lifted = handled = False  # handled: lifted or knocked over, so the mug resting means the robot is done with it
    rest = steps = 0
    t0, t_policy = time.time(), 0.0  # t_policy: cameras + policy, the rest is physics (+ filming)
    while steps < (GRASP_STEPS if grasp else MAX_STEPS):
        if not queue:
            t1 = time.time()
            with lock():
                frame = observe(e)
            queue = list(act(model, frame, task))
            t_policy += time.time() - t1
        with lock():
            obs, *_ = env.step(queue.pop(0))
        steps += 1
        if VERBOSE and steps % PROGRESS_EVERY == 0:
            print(f"    {_cache['name']} {plant}-{spot}: step {steps}/{MAX_STEPS}, {time.time() - t0:.0f} s ({t_policy:.0f} s cameras + policy)", flush=True)
        if viewer:
            viewer.sync()
            if not viewer.is_running():
                break
        pos, rot, speed = mug_pose(e)
        lifted |= pos[2] - start[2] > LIFT_CM / 100
        handled |= lifted or rot.apply([0, 0, 1])[2] < np.cos(np.radians(TOPPLED_DEG))
        released = np.linalg.norm(obs["robot0_eef_pos"] - pos) > RELEASE_CM / 100
        rest = rest + 1 if handled and released and speed < 0.01 else 0
        if video and steps % VIDEO_EVERY == 0:
            n = e.watering_stats()["target"]
            frames.append(film(e, [_cache["name"], task, f"spot {spot}: {n}/{DROPLET['count']} in plant"]))
        if rest >= REST_S * CONTROL_HZ or grasp and handled:
            break

    stats = e.watering_stats()  # pos, rot, released: as at the loop's last step
    tilt = float(np.degrees(np.arccos(np.clip(rot.apply([0, 0, 1])[2], -1, 1))))
    off_cm = float(np.linalg.norm(pos[:2] - start[:2]) * 100)
    watered = bool(e._check_success())
    placed = bool(released) and off_cm <= PLACED_CM and tilt <= PLACED_DEG
    if video and frames:
        tmp = video.with_suffix(".tmp.mp4")
        wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_HZ / VIDEO_EVERY, frames[0].shape[1::-1])
        for f in frames:
            wr.write(f)
        wr.release()
        to_h264(tmp, video)
    return dict(seed=seed, in_target=stats["target"],
                in_wrong=sum(stats[p] for p in e.pot_names if p != e.target_pot),
                spilled=stats["spilled"], in_mug=stats[e.mug_name], watered=watered, lifted=bool(lifted),
                placed=placed, success=watered and placed, mug_off_cm=round(off_cm, 1),
                mug_tilt_deg=round(tilt, 1), steps=steps, seconds=round(time.time() - t0, 1))


def trial_seed(plant, spot, t):
    """Seed of trial t on a plant x spot: the same for every model."""
    return SEED + 1000 * PLANTS.index(plant) + 100 * spot + t


def describe(r, plant, grasp=False):
    """One line on a trial's result."""
    if grasp:
        return f"mug {'LIFTED' if r['lifted'] else 'not lifted'}, tilted {r['mug_tilt_deg']:.0f} deg ({r['steps']} steps, {r['seconds']:.0f} s)"
    return (f"{r['in_target']}/{DROPLET['count']} in the {plant} plant, {r['in_wrong']} in a wrong one, "
            f"mug {'placed' if r['placed'] else 'not placed'} -> {'SUCCESS' if r['success'] else 'fail'} "
            f"({r['steps']} steps, {r['seconds']:.0f} s)")


def run_unit(unit, name, grasp=False, tally=None, lock=None):
    """TRIALS trials of one plant x spot -> results/<model>/<unit>.json.

    unit  -> "<plant>-<spot>", e.g. "left-6"
    name  -> model name
    grasp -> grasp-only trials -> results/grasp-n<N_ACTION_STEPS>/<model>/<unit>.json
    tally -> shared {trials, hits} the main process prints progress from, or None; lock guards it
    returns {clip: unit, summary}
    """
    plant, spot = unit.split("-")[0], int(unit.split("-")[1])
    out = (RESULTS / f"grasp-n{N_ACTION_STEPS}" if grasp else RESULTS) / name
    (out / "videos").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # parallel workers: each torch on every core would fight over the CPU
    model, env = load_model(name), scene(plant)
    trials = []
    for t in range(TRIALS):
        video = out / "videos" / f"{unit}.mp4" if t < VIDEO_TRIALS else None
        r = trial(env, model, plant, spot, trial_seed(plant, spot, t), video, grasp=grasp)
        trials.append(r)
        if VERBOSE:
            print(f"  {name} {unit} #{t}: {describe(r, plant, grasp)}", flush=True)
        if tally is not None:
            with lock:
                tally["trials"] += 1
                tally["hits"] += r["lifted" if grasp else "success"]
    (out / f"{unit}.json").write_text(json.dumps(dict(
        model=name, repo="@".join(filter(None, runs.repo(name))), plant=plant, spot=spot, spot_cm=SPOTS[spot],
        trained_spot=spot in TRAINED_SPOTS, n_action_steps=N_ACTION_STEPS, max_steps=GRASP_STEPS if grasp else MAX_STEPS,
        trials=trials), indent=1))
    return dict(clip=unit, summary=f"{sum(r['lifted' if grasp else 'success'] for r in trials)}/{TRIALS}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*")
    p.add_argument("--jobs", type=int, default=2, help="units run in parallel, each with its own copy of the model")
    p.add_argument("--grasp", action="store_true", help="grasp-only trials (passed by ./run eval-grasp)")
    args = p.parse_args()
    wanted = args.models or MODELS
    from huggingface_hub import file_exists
    # fine-tuned ones (and their step-<n> branches) appear on the Hub as training finishes
    models = [m for m in wanted if file_exists(runs.repo(m)[0], "model.safetensors", revision=runs.repo(m)[1])]
    if missing := [m for m in wanted if m not in models]:
        print(f"not on the Hub yet (skipped): {', '.join(missing)}")
    out = RESULTS / f"grasp-n{N_ACTION_STEPS}" if args.grasp else RESULTS
    label = "lifted" if args.grasp else "succeeded"
    manager = multiprocessing.Manager()  # the workers' shared trial tally
    for name in models:
        todo = [u for u in UNITS if not (out / name / f"{u}.json").exists()]
        print(f"\n=== {name}: {len(todo)} of {len(UNITS)} units to run, {TRIALS} trials each", flush=True)
        tally, lock, stop = manager.dict(trials=0, hits=0), manager.Lock(), threading.Event()
        total, t0 = len(todo) * TRIALS, time.time()

        def report():
            t, h, took = tally["trials"], tally["hits"], time.time() - t0
            left = f", ~{episodes._clock(took / t * (total - t))} left" if 0 < t < total else ""
            print(f"  {name}: {t}/{total} trials, {h} {label} ({h / max(t, 1):.0%}) · {episodes._clock(took)}{left}", flush=True)

        def ticker():
            while not stop.wait(PRINT_EVERY):
                report()

        threading.Thread(target=ticker, daemon=True).start()
        for r in episodes.parallel_map(run_unit, todo, {"name": name, "grasp": args.grasp, "tally": tally, "lock": lock},
                                       args.jobs, name):
            if "error" in r:
                print(f"{r['clip']}: FAILED: {r['error']}")
        stop.set()
        report()
        print(f"{name}: {tally['hits']}/{tally['trials']} {label} (units run now)")
    print(f"\nresults: {RESULTS.relative_to(Path.cwd())}/<model>/  (./run compare for tables and plots)")


if __name__ == "__main__":
    main()
