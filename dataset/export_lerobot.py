"""Saved robot runs -> a LeRobot dataset for fine-tuning SmolVLA (the HuggingFaceVLA/smolvla_libero format).

Per control step of each episode (dataset/data/<group>/<clip>/robot_states.npz), with no physics: put the scene
in the state the action was chosen in, render the two cameras and read the robot's state, and pair them
with the action sent and the episode's command. Matches LeRobot's LIBERO env + LiberoProcessorStep, so
training sees exactly what the policy gets at evaluation:
  observation.images.image    agentview, 256x256, rotated 180 deg like LIBERO's (the raw render flipped in H and W)
  observation.images.image2   wrist camera, same
  observation.state           eef position (3), eef axis-angle (3), gripper finger positions (2)
  action                      OSC delta pose scaled to [-1, 1] (6) + gripper (-1 open, +1 close)
  task                        the scene's command, e.g. "Water the left plant"
fps = the control rate (20 Hz): one frame per action.

Only successful episodes (water in the plant, mug put back) unless ALL.
Writes dataset/data/lerobot/<NAME>/ (replaced if it exists), plus meta/clips.json: the clip name of each
episode, in episode order (training picks its demo subsets by clip). Local only: ./run upload publishes it.

Usage: ./run export [A1-0 ...]   (default: every episode with a saved robot run)
"""
import argparse
import json
import shutil

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.env_processor import LiberoProcessorStep

from dataset import episodes
from dataset.sim_runs import playback

NAME = "water_pour"  
ALL = False          # to export failed robot runs
SIZE = 256           # camera image size (px), as in SmolVLA-LIBERO
CAMERAS = {"observation.images.image": "agentview", "observation.images.image2": "robot0_eye_in_hand"}
FEATURES = {
    **{key: {"dtype": "video", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channels"]} for key in CAMERAS},
    "observation.state": {"dtype": "float32", "shape": (8,),
                          "names": ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "finger_l", "finger_r"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
}
_to_axisangle = LiberoProcessorStep()._quat2axisangle


def observe(problem):
    """Camera images and the 8-d state, as the policy will see them, for the scene's current state."""
    frame = {key: problem.sim.render(camera_name=cam, width=SIZE, height=SIZE)[::-1, ::-1].copy()
             for key, cam in CAMERAS.items()}
    obs = problem._get_observations(force_update=True)
    axisangle = _to_axisangle(torch.as_tensor(obs["robot0_eef_quat"], dtype=torch.float32)[None])[0].numpy()
    frame["observation.state"] = np.r_[obs["robot0_eef_pos"], axisangle, obs["robot0_gripper_qpos"]].astype(np.float32)
    return frame


def result(clip):
    path = episodes.clip_dir(clip) / "robot.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    args = p.parse_args()
    clips = args.clips or playback.saved()
    keep = [c for c in clips if ALL or result(c).get("success")]
    skipped = sorted(set(clips) - set(keep), key=episodes.sort_key)
    if skipped:
        print(f"skipping {len(skipped)} unsuccessful robot runs (ALL = True to keep them): {', '.join(skipped)}")
    if not keep:
        raise SystemExit(f"nothing to export: no successful robot run has saved states ({playback.FILE}). Runs made before "
                         "./run robot saved them have only robot.json; re-run ./run robot for those clips")

    root = episodes.DATA / "lerobot" / NAME
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(f"local/{NAME}", fps=playback.load(keep[0])["control_hz"],
                               features=FEATURES, root=root, robot_type="panda", use_videos=True)
    env = playback.scene()
    e = env.env
    frames = 0
    for n, clip in enumerate(keep, 1):
        s = playback.load(clip)
        task = f"Water the {s['side']} plant"  # the BDDL command for this plant (sim/watering/bddl.py)
        for k, action in enumerate(s["action"]):
            # the action was chosen in the state before this step: row k-1, or the start for k = 0
            playback.set_qpos(e, s["qpos_start"] if k == 0 else s["qpos"][k - 1])
            ds.add_frame({**observe(e), "action": action.astype(np.float32), "task": task})
        ds.save_episode()
        frames += len(s["action"])
        print(f"[{n}/{len(keep)}] {clip}: {len(s['action'])} frames, \"{task}\"", flush=True)
    ds.finalize()
    (root / "meta" / "clips.json").write_text(json.dumps(keep))
    env.close()
    print(f"\n{len(keep)} episodes, {frames} frames ({frames / ds.fps / 60:.1f} min at {ds.fps} fps) -> {root}")


if __name__ == "__main__":
    main()
