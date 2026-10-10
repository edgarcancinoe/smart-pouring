"""Replay tracked pours in the sim: the mug alone follows the recorded path (no robot).

For each clip: put the sim mug on the clip's start spot, drive it along the cleaned path at real speed and
count where the droplets land. Writes, in dataset/data/<group>/<clip>/:
  replay.json  plant, droplets in it, success
  replay.mp4   real video with the tracked mug drawn | sim from the phone's viewpoint

Usage: ./run replay [A1-0 ...] [--jobs 4]   (default: every tracked episode)
"""
import argparse
import json

import cv2
import numpy as np

from dataset import episodes
from dataset.retarget import mug_path as mp
from dataset.tracking.mug_tracker import tag_offset, tilt_deg, to_h264
from sim import watering
from sim.watering.assets import DROPLET
from sim.watering.kinematic import KinematicReplay

FPS_OUT = 30
SETTLE_S = 1.0    # keep simulating after the clip so the droplets come to rest
W, H = 640, 360   # size of each video panel
ROBOT_GROUP = 5   # geom group for the robot's visuals, hidden in the phone view
AXIS_M = 0.08     # length of the drawn mug axes


def label(img, text):
    """Write text in the top-left corner of a frame (in place); returns the frame."""
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_mug(frame, track, pos, rot, kept):
    """Draw the tracked mug's axes (red x, green y = front, blue z = up) at its tag, on a full-size real frame.

    frame    -> (h, w, 3) BGR frame of the recording, drawn on in place
    track    -> dict from mug_path.load() (camera pose, K, dist)
    pos, rot -> mug pose in this frame: (3,) floor cm, Rotation (cleaned path)
    kept     -> False if this frame's pose was filled in (tag not seen or dropped): thinner axes
    """
    R_cf, t_cf = track["cam_R"], track["cam_t"]  # camera -> floor (cm)
    R_mug = rot.as_matrix()
    tag_height, tag_radius = tag_offset()
    tag = pos + R_mug[:, 2] * tag_height + R_mug[:, 1] * tag_radius  # up the axis, out to the tag (floor cm)
    R_mc = R_cf.T @ R_mug                        # mug axes in the camera frame
    t_mc = R_cf.T @ (tag - t_cf) / 100           # tag centre in the camera frame, m
    cv2.drawFrameAxes(frame, track["K"], track["dist"], cv2.Rodrigues(R_mc)[0], t_mc, AXIS_M, 8 if kept else 3)


def phone_camera(problem, track, pos0, sim_pos0):
    """Turn the scene's shoulderview camera into the phone (same pose relative to the mug) and hide the robot
    in renders: it stands where the demonstrator stood, right in front of the phone."""
    m = problem.sim.model._model
    cid = m.camera("shoulderview").id
    m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = mp.camera_in_sim(track, pos0, sim_pos0)
    problem.sim.forward()
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name.startswith(("robot0_", "gripper0_", "mount0_")) and m.geom_group[g] == 1:
            m.geom_group[g] = ROBOT_GROUP
    problem.sim.render(camera_name="agentview", width=8, height=8)  # create the render context
    problem.sim._render_context_offscreen.vopt.geomgroup[ROBOT_GROUP] = 0


def replay(clip):
    """Drive the sim mug along the clip's tracked path and count where the droplets land.

    clip -> clip name, e.g. "A1-0"
    returns the replay.json dict: clip, plant, in_plant (droplets in the poured pot), success bool
    """
    prep = mp.prepare(clip)
    track, t, side, kept = prep["track"], prep["t"], prep["side"], prep["kept"]
    out = episodes.clip_dir(clip)

    env = watering.make_env(side, hard_reset=False, initialization_noise=None, use_camera_obs=False)
    env.reset()
    e = env.env
    times, sim_pos, sim_rot, sim_pos0 = mp.place_in_sim(prep, e)
    driver = KinematicReplay(e, times, sim_pos, sim_rot)
    phone_camera(e, track, prep["pos0"], sim_pos0)

    cap = cv2.VideoCapture(str(episodes.video(clip)))
    writer = cv2.VideoWriter(str(out / "replay_tmp.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (2 * W, H))
    real, n_read = None, 0
    every = int(round(1 / FPS_OUT / driver.dt))  # physics steps per video frame
    for i in range(int((driver.duration + SETTLE_S) / driver.dt)):
        driver.step()
        if i % every:
            continue
        ts = i * driver.dt
        # advance the real video to the same moment, by frame count (iPhone timestamps are unreliable)
        while n_read <= min(int(np.searchsorted(t, ts)), len(t) - 1):
            ok, frame = cap.read()
            if not ok:
                break
            real, n_read = frame, n_read + 1
        k = n_read - 1  # track row of the real frame shown
        frame = real.copy()
        draw_mug(frame, track, prep["pos"][k], prep["rot"][k], kept[k])
        tilt = tilt_deg(prep["rot"][k].as_matrix())
        real_label = f"{clip}  t={ts:4.1f}s  tilt {tilt:3.0f} deg" + ("" if kept[k] else "  (filled in)")
        phone = e.sim.render(camera_name="shoulderview", width=W, height=H)[::-1, :, ::-1].copy()
        in_plant = e.watering_stats()["target"]
        writer.write(np.hstack([label(cv2.resize(frame, (W, H)), real_label),
                                label(phone, f"sim, phone view  in {side}: {in_plant}/{DROPLET['count']}")]))
    writer.release()
    cap.release()
    to_h264(out / "replay_tmp.mp4", out / "replay.mp4")

    result = dict(clip=clip, plant=side, in_plant=e.watering_stats()["target"], success=bool(e._check_success()))
    env.close()
    (out / "replay.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    p.add_argument("--jobs", type=int, default=0, help="clips run in parallel (default: up to half the CPU cores)")
    args = p.parse_args()
    for r in episodes.parallel_map(replay, args.clips or episodes.tracked(), {}, args.jobs, "replay"):
        if "error" in r:  # crashed: record it as a failure
            (episodes.clip_dir(r["clip"]) / "replay.json").write_text(json.dumps(dict(r, success=False), indent=2))
            print(f"{r['clip']}: {r['error']}")
        elif not r["success"]:
            print(f"{r['clip']}: only {r['in_plant']}/{DROPLET['count']} droplets in the {r['plant']} plant")


if __name__ == "__main__":
    main()
