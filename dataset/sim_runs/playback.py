"""Saved robot executions: the full sim state at every control step of ./run robot, so an episode can be
watched (./run watch) or filmed (./run video) again without re-simulating it. Re-simulating takes
minutes per clip; playing back saved states is instant.

dataset/data/<group>/<clip>/robot_states.npz, one row per control step k:
  action[k]        the 7-d OSC action sent (delta pose scaled to [-1, 1], gripper), the format SmolVLA outputs
  qpos[k]          the sim state AFTER that action; the state it was chosen in is qpos[k-1] (qpos_start for
                   k = 0), so training pairs are (render(qpos[k-1]), action[k]); images can be rendered later
                   at any size from these states
  mocap_pos/quat   the ghosts (intended mug and hand), phase, hand error (cm, deg)
plus the plant side, playback speed and control rate.

Usage: ./run video [A1-0 ...] [--jobs 8]   robot.mp4 per clip (default: every episode with saved states)
       ./run watch [A1-0 ...] [--rate 2]  play them in the MuJoCo viewer, one after another
Viewer keys: 6 = pause / play   7 = previous episode   8 = restart   9 = next episode
             4 = show / hide the ghosts (green = intended mug, orange = intended hand)
             ]  /  [ = next / previous camera (agentview, wrist, ...);  Esc = free camera
"""
import argparse
import json
import time

import cv2
import mujoco
import numpy as np

from dataset import episodes
from dataset.tracking.mug_tracker import to_h264
from sim import watering
from sim.watering.assets import DROPLET
from sim.watering.problem import GHOST_GROUP

FILE = "robot_states.npz"
VIDEO_EVERY = 2  # film every Nth control step (the video still plays smoothly)
H = 480          # video panel size (robot camera | wrist camera)


class Recorder:
    """Collects the sim state once per control step; save() writes robot_states.npz.

    problem -> the LIBERO problem (env.env) being recorded
    """

    def __init__(self, problem):
        self.d = problem.sim.data._data
        self.qpos_start = self.d.qpos.astype(np.float32)
        self.qpos, self.mocap_pos, self.mocap_quat, self.action = [], [], [], []

    def add(self, action):
        """Store the state after one control step.

        action -> (7,) action that was sent
        """
        self.action.append(np.asarray(action, np.float32))
        self.qpos.append(self.d.qpos.astype(np.float32))
        self.mocap_pos.append(self.d.mocap_pos.astype(np.float32))
        self.mocap_quat.append(self.d.mocap_quat.astype(np.float32))

    def save(self, clip, phase, track_err_cm, rot_err_deg, side, speed, control_hz):
        """Write robot_states.npz. K = control steps recorded.

        clip         -> clip name
        phase        -> K phase labels
        track_err_cm -> K hand position errors, cm
        rot_err_deg  -> K hand orientation errors, deg
        side         -> plant name
        speed        -> fraction of real speed the pour was played at
        control_hz   -> control rate
        """
        np.savez_compressed(episodes.clip_dir(clip) / FILE, qpos=np.array(self.qpos), qpos_start=self.qpos_start,
                            action=np.array(self.action), mocap_pos=np.array(self.mocap_pos),
                            mocap_quat=np.array(self.mocap_quat), phase=np.array(phase),
                            track_err_cm=np.array(track_err_cm, np.float32), rot_err_deg=np.array(rot_err_deg, np.float32),
                            side=side, speed=speed, control_hz=control_hz)


def load(clip):
    """A saved robot run. K = control steps, nq = sim state size.

    clip -> clip name, e.g. "A1-0"
    returns dict: qpos (K, nq), qpos_start (nq,), action (K, 7), mocap_pos (K, 2, 3), mocap_quat (K, 2, 4)
            (ghosts: mug, hand), phase (K,), track_err_cm (K,), rot_err_deg (K,), side str, speed, control_hz
    """
    path = episodes.clip_dir(clip) / FILE
    if not path.exists():
        raise FileNotFoundError(f"{clip}: no saved robot run ({path.name}); run ./run robot {clip} first")
    with np.load(path) as f:
        s = {k: f[k] for k in f.files}
    for k in ("side", "speed", "control_hz"):
        s[k] = s[k].item()
    return s


def saved():
    """Clips with a saved robot run, in episode order.

    returns list of clip names
    """
    return episodes.having(FILE)


def scene(offscreen=True):
    """The watering scene, ready to show saved states (all three plant commands share one model).

    offscreen -> True: render to images; False: no renderer (for the MuJoCo viewer)
    returns the LIBERO env (the problem is env.env)
    """
    env = watering.make_env("left", offscreen=offscreen, hard_reset=False, initialization_noise=None,
                            use_camera_obs=False, **({"camera_heights": H, "camera_widths": H} if offscreen else {}))
    env.reset()
    return env


def set_qpos(problem, qpos, mocap_pos=None, mocap_quat=None):
    """Put the scene in a saved state (and the ghosts at saved poses, if given).

    problem               -> the LIBERO problem (env.env)
    qpos                  -> (nq,) sim state
    mocap_pos, mocap_quat -> ghost poses (2, 3) and (2, 4) wxyz, or None to leave them
    """
    m, d = problem.sim.model._model, problem.sim.data._data
    if len(qpos) != m.nq:
        raise ValueError("the scene changed since this run was saved; re-run ./run robot for it")
    d.qpos[:] = qpos
    if mocap_pos is not None:
        d.mocap_pos[:], d.mocap_quat[:] = mocap_pos, mocap_quat
    mujoco.mj_forward(m, d)


def apply(problem, s, k):
    """Put the scene in the state after control step k, ghosts included.

    problem -> the LIBERO problem (env.env)
    s       -> dict from load()
    k       -> control step index
    """
    set_qpos(problem, s["qpos"][k], s["mocap_pos"][k], s["mocap_quat"][k])


def render(clip):
    """robot.mp4 from the saved states: robot camera with ghosts | wrist camera (each H x H),
    played so the pour runs at the demonstrator's real speed.

    clip -> clip name, e.g. "A1-0"
    returns {clip}
    """
    s = load(clip)
    env = scene()
    e = env.env
    e.sim.render(camera_name="agentview", width=8, height=8)  # create the render context
    e.sim._render_context_offscreen.vopt.geomgroup[GHOST_GROUP] = 1  # show the intended mug and hand
    out = episodes.clip_dir(clip)
    tmp = out / "robot_tmp.mp4"
    fps = s["control_hz"] / VIDEO_EVERY / s["speed"]
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * H, H))
    white = (255, 255, 255)
    for k in range(0, len(s["qpos"]), VIDEO_EVERY):
        apply(e, s, k)
        imgs = [e.sim.render(camera_name=c, width=H, height=H)[::-1, :, ::-1].copy()
                for c in ("agentview", "robot0_eye_in_hand")]
        in_plant = e.watering_stats()[f"plant_pot_{s['side']}"]
        cv2.putText(imgs[0], f"{clip}  {s['phase'][k]}  {in_plant}/{DROPLET['count']} in plant", (10, 25), 0, 0.6, white, 2)
        cv2.putText(imgs[0], f"hand off target: {s['track_err_cm'][k]:4.1f} cm {s['rot_err_deg'][k]:3.0f} deg",
                    (10, 50), 0, 0.6, white, 2)
        cv2.putText(imgs[0], "green = intended mug, orange = intended hand", (10, H - 12), 0, 0.45, white, 1)
        cv2.putText(imgs[1], "wrist camera", (10, 25), 0, 0.6, white, 2)
        wr.write(np.hstack(imgs))
    wr.release()
    env.close()
    to_h264(tmp, out / "robot.mp4")
    return dict(clip=clip)


def watch(clips, rate=1.0):
    """Play saved runs in the MuJoCo viewer (needs mjpython on macOS): the pour at the demonstrator's
    real speed x rate; the terminal shows each episode's result and the phase as it plays.

    clips -> clip names to play, in order
    rate  -> playback speed (1 = the demonstrator's speed)
    """
    import mujoco.viewer

    # offscreen=False: no robosuite renderer (its GLFW window crashes off the main thread under mjpython)
    env = scene(offscreen=False)
    e = env.env
    m, d = e.sim.model._model, e.sim.data._data
    state = {"i": 0, "k": 0, "playing": True, "pending": None, "s": None, "phase": None}

    def summary(clip):
        path = episodes.clip_dir(clip) / "robot.json"
        if not path.exists():
            return ""
        r = json.loads(path.read_text())
        if "error" in r:
            return f"FAILED: {r['error']}"
        yn = lambda b: "?" if b is None else ("yes" if b else "NO")
        return (f"{'SUCCESS' if r['success'] else 'FAILED'} | water {r['in_plant']}/30 "
                f"| placed {yn(r.get('placed'))} | grip {r.get('tilt', '?')} deg")

    def show(i):
        """Switch to episode i; skips (and reports) one that can't be shown."""
        clip = clips[i]
        try:
            s = load(clip)
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[{i + 1}/{len(clips)}] {clip}: {err}")
            return False
        state.update(i=i, k=0, s=s, phase=None)
        print(f"\n[{i + 1}/{len(clips)}] {clip} ({s['side']} plant): {summary(clip)}")
        return True

    def on_key(key):  # runs on the viewer thread: only record the request
        state["pending"] = {ord("6"): "pause", ord("7"): "prev", ord("8"): "restart", ord("9"): "next"}.get(key)

    def step_to(direction):
        i = state["i"]
        for _ in clips:
            i = (i + direction) % len(clips)
            if show(i):
                return

    if not show(0):
        step_to(+1)
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        with viewer.lock():  # the viewer draws on its own thread: id before type, or it sees a fixed camera -1
            viewer.opt.geomgroup[GHOST_GROUP] = 1
            viewer.cam.lookat[:] = [-0.05, 0.0, 0.95]  # free-camera start point, used after Esc
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, 180, -35
            viewer.cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        while viewer.is_running():
            tic = time.time()
            action, state["pending"] = state["pending"], None
            if action == "pause":
                state["playing"] = not state["playing"]
                print("paused" if not state["playing"] else "playing")
            elif action == "restart":
                state["k"], state["playing"] = 0, True
            elif action in ("next", "prev"):
                step_to(+1 if action == "next" else -1)
                state["playing"] = True
            s, k = state["s"], state["k"]
            with viewer.lock():
                apply(e, s, k)
            viewer.sync()
            if s["phase"][k] != state["phase"]:
                state["phase"] = s["phase"][k]
                print(f"   {state['phase']}")
            if state["playing"]:
                if k + 1 < len(s["qpos"]):
                    state["k"] = k + 1
                else:
                    state["playing"] = False
                    in_plant = e.watering_stats()["plant_pot_" + s["side"]]
                    print(f"   end: {in_plant}/30 droplets in the plant. 9 = next, 8 = replay")
            # one control step per frame, timed so the pour plays at the demonstrator's speed (x rate)
            frame_s = s["speed"] / s["control_hz"] / rate
            time.sleep(max(0.002, frame_s - (time.time() - tic)))
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    p.add_argument("--watch", action="store_true", help="play in the viewer instead of rendering videos")
    p.add_argument("--rate", type=float, default=1.0, help="viewer playback speed (default %(default)s = the demonstrator's speed)")
    p.add_argument("--jobs", type=int, default=0, help="videos rendered in parallel (default: up to half the CPU cores)")
    args = p.parse_args()
    clips = args.clips or saved()
    if not clips:
        raise SystemExit("no saved robot runs yet: ./run robot first")
    if args.watch:
        return watch(clips, args.rate)
    for r in episodes.parallel_map(render, clips, {}, args.jobs, "video"):
        if "error" in r:
            print(f"{r['clip']}: FAILED: {r['error']}")
    print("videos: dataset/data/<group>/<clip>/robot.mp4")


if __name__ == "__main__":
    main()
