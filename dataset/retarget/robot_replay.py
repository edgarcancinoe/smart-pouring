"""The robot reproduces a tracked pour: the demonstrations SmolVLA is trained on.

Grip: chosen per clip. The hand points at the upright mug from the robot's shoulder on the mug's side,
fingers closing on its left/right, tilted down by the flattest of TILTS that leaves the arm joint room for
the whole pour. During the pour the hand target is the mug's pose times that fixed grip offset, as if the
mug were glued to the gripper.

Phases (CONTROL_HZ, OSC delta-pose actions: the format SmolVLA outputs):
  reach    scripted: up, across to behind the mug, down, slide in, close the gripper
  follow   the clip's mug path -> hand targets
  release  scripted: open, slide back out, lift, return to the start pose

Success = water in the plant and the mug put back where the demonstration ended.
Writes, in dataset/data/<group>/<clip>/: robot.json (result), robot_states.npz (sim state and action per
control step: for ./run watch, ./run video and ./run export) and, if VIDEO, robot.mp4.

Each clip also runs SHIFTS more times as <clip>~1, <clip>~2, ...: the mug (and the whole pour) shifted by up to
SHIFT_CM, so the policy must find the mug in the image instead of memorizing four grasp spots.

Usage: ./run robot [A1-0 ...] [--jobs 4]   (default: every tracked episode)
"""
import argparse
import json
import zlib

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from dataset import episodes
from dataset.retarget import mug_path as mp
from dataset.sim_runs import playback
from sim import watering
from sim.watering import CONTROL_HZ
from sim.watering.assets import DROPLET
from sim.watering.robot import Arm, best_family, grab_point, heading, radial_grip, START_Q

VIDEO = True            # also render robot.mp4 (./run video can render it later from the saved states)
TILTS = (35, 40, 50)    # deg the hand may point down at the grab, flattest first
MIN_ROOM = 5            # deg every joint must keep from its limits over the grab and pour to take a tilt
PLAN_STEP_S = 0.2       # s of clip time between the pour poses checked when choosing the tilt
SPEED = 0.6             # play the demonstrator's path at this fraction of real speed
LEAD_S = 0.10           # s to aim ahead along the path (the controller trails its target)
APPROACH = 0.12         # m behind the mug where the hand descends before sliding in
SAFE_HEIGHT = 0.20      # m above the mug for moves across the scene
REACH_STEP = 0.012      # m per control step along the scripted lines
SHOULDER_OFFSET = 0.08  # m beside the base, on the mug's side: the grip points at the mug from there
MAX_STEP = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])  # controller's range per step (m, rad): action +-1
SHIFTS = 3              # extra runs per clip with the mug shifted (0: only the clip as recorded)
SHIFT_CM = 3.0          # uniform +- per axis on the table
PLACED_CM, PLACED_DEG = 3.0, 15.0  # the mug counts as put back within this of the demonstration's end pose


def eef_pose(obs):
    """The gripper pose from an observation.

    obs -> LIBERO observation dict
    returns pos (3,) m, Rotation
    """
    return obs["robot0_eef_pos"].copy(), R.from_quat(obs["robot0_eef_quat"])


def perturb(action, phase, steps):
    """The action the arm executes; the recorded one is always the clean action (smolvla-recoveries/ may override this).

    action -> (7,) clean action from action_to
    phase  -> label of the step ("reach" / "grasp" / ...)
    steps  -> step budget of the drive() call it comes from (1: a step along a scripted line)
    """
    return action


def action_to(obs, p_target, r_target, grip):
    """Delta action toward a target pose (clipped to the controller's per-step range).

    obs                -> LIBERO observation dict (current gripper pose)
    p_target, r_target -> target pose: (3,) m, Rotation
    grip               -> -1 open, 1 closed
    returns action (7,) in [-1, 1] (dx dy dz, rotation vector, grip: the SmolVLA action format),
            delta (6,) the raw pose difference (m, rad)
    """
    p, r = eef_pose(obs)
    delta = np.r_[p_target - p, (r_target * r.inv()).as_rotvec()]
    return np.r_[np.clip(delta / MAX_STEP, -1, 1), grip], delta


def choose_tilt(arm, prep, e, base_xy, shift):
    """The flattest of TILTS from which an arm posture grabs this clip's mug and follows the whole pour with
    MIN_ROOM deg of joint room (geometry only, ~5 s per tilt); else the one with the most room.

    arm     -> Arm (kinematics of this scene's robot)
    prep    -> dict from mug_path.prepare()
    e       -> the LIBERO problem (env.env); the mug is moved to the clip's start spot
    base_xy -> (2,) robot base position, m
    shift   -> (2,) m the mug is moved from its spot
    returns tilt deg, q_grab (7,) joint angles rad at the grab
    """
    times, mug_pos, mug_rot, mug0 = mp.place_in_sim(prep, e, shift=shift)  # the pour in sim coordinates; mug on its spot
    
    # mug poses along the pour, every PLAN_STEP_S of clip time
    samples     = np.arange(0, times[-1], PLAN_STEP_S)
    sampled_pos = np.column_stack([np.interp(samples, times, mug_pos[:, k]) for k in range(3)])
    sampled_rot = Slerp(times, mug_rot)(samples) # We use it to bring mug-relative rotation to sim rotation

    fallback = None
    
    for tilt in TILTS:
        # this tilt's grip, and the same grip in the mug's frame (fixed for the whole pour)
        R_grip  = radial_grip(mug0[:2], base_xy, tilt, SHOULDER_OFFSET)
        grab    = grab_point(mug0, R_grip)
        off_p   = mug_rot[0].inv().apply(grab - mug_pos[0])
        off_r   = mug_rot[0].inv() * R.from_matrix(R_grip)

        # hand targets along the pour: mug pose x grip offset
        targets = [ (sampled_pos[i] + sampled_rot[i].apply(off_p),  # Position
                    (sampled_rot[i] * off_r).as_matrix())           # Orientation
                    for i in range(len(samples))]

        # IK along the path. room is the minimum joint room over the whole pour
        q, room = best_family(arm, grab, R_grip, targets)  # room = -1: no posture follows the whole pour

        if room >= MIN_ROOM:
            return tilt, q  # the flattest tilt with enough room
        
        if room >= 0 and (fallback is None or room > fallback[2]):
            fallback = (tilt, q, room)

    if fallback is None:
        raise SystemExit(f"{prep['clip']}: no arm posture can grab and pour at any tilt")
    
    return fallback[:2]  # no tilt had MIN_ROOM: the one with the most room

def run(clip):
    """Robot: Grab mug, follow path, pour, put mug back; save

    clip -> clip name, e.g. "A1-0"
    returns robot.json dict: clip, plant, tilt deg, in_plant (%), watered / placed / success bools
    """
    episodes.clip_dir(clip).mkdir(parents=True, exist_ok=True)  # shifted runs get their own folder
    prep = mp.prepare(clip.split("~")[0])
    # shifted runs: a fixed offset per name, so re-running gives the same demo
    shift = np.random.default_rng(zlib.crc32(clip.encode())).uniform(-SHIFT_CM, SHIFT_CM, 2) / 100 if "~" in clip else (0, 0)
    
    # no camera observations in simulation. video is rendered afterwards from the saved states
    env = watering.make_env(prep["side"], hard_reset=False, initialization_noise=None, horizon=100000, ignore_done=True, use_camera_obs=False)
    env.reset()
    e = env.env
    arm = Arm(e)
    base_xy = e.sim.data.body_xpos[e.sim.model._model.body("robot0_base").id][:2].copy()
    
    
    # Pick the grip before moving: Robot may be limited to perform the pour depending on how the mug is grasped initially.
    # We test samples along the planned path assuming different hand tilts. We start by lower inclination (flattest hand tilt)
    # and choose the first one that grabs the mug and performs whole pour.
    
    tilt, _ = choose_tilt(arm, prep, e, base_xy, shift)
    
    # start pose: the same for every demo, as at evaluation
    q_ready = START_Q
    R_start = arm.fk(q_ready)[1]  # hand orientation at the start

    # restart the episode from it (reset puts the robot at init_qpos, so set that first), and make it the
    # posture the controller relaxes toward (otherwise it drifts back to LIBERO's home pose)
    e.robots[0].init_qpos = q_ready
    env.reset()
    e.robots[0].controller.update_initial_joints(q_ready)

    # mug on the clip's start spot; the pour in sim coordinates
    times, sim_pos, sim_rot, mug0 = mp.place_in_sim(prep, e, shift=shift)
    obs, *_ = env.step(np.r_[np.zeros(6), -1])  # one step with the gripper open, for a fresh observation

    # grab pointing at the mug from the shoulder on the mug's side, fingers left/right
    Rg = radial_grip(mug0[:2], base_xy, tilt, SHOULDER_OFFSET)

    # fixed grip offset in the mug's frame, from the upright start pose
    mug_r0  = sim_rot[0]
    grip_p0 = grab_point(mug0, Rg)
    off_p   = mug_r0.inv().apply(grip_p0 - sim_pos[0])
    r_grip  = R.from_matrix(Rg)
    off_r   = mug_r0.inv() * r_grip
    slerp   = Slerp(times, sim_rot)

    def grip_target(t):
        """Where the hand should be at clip time t: the mug's pose then, times the fixed grip offset.

        t -> s along the (sim) path
        returns pos (3,) m, Rotation
        """
        t = float(np.clip(t, times[0], times[-1]))
        p = np.array([np.interp(t, times, sim_pos[:, k]) for k in range(3)])
        r = slerp(t)
        return p + r.apply(off_p), r * off_r

    rec = playback.Recorder(e)  # sim state, action and phase per control step -> robot_states.npz
    wxyz = lambda r: r.as_quat()[[3, 0, 1, 2]]
    e.set_ghost("ghost_mug", sim_pos[0], wxyz(sim_rot[0]))  # intended mug: its start pose until the follow

    def drive(p_t, r_t, grip, phase, steps, tol=(0.005, 3.0)):
        """Up to `steps` control steps toward a pose; stops once within tol (m, deg), or once the hand
        has stopped getting closer (< 0.5 mm and < 0.3 deg in 10 steps: e.g. a joint at its limit).

        p_t, r_t -> target pose: (3,) m, Rotation
        grip     -> -1 open, 1 closed
        phase    -> label saved with each step
        steps    -> max control steps
        tol      -> (m, deg) close enough
        """
        nonlocal obs
        errs = []
        for _ in range(steps):
            a, delta = action_to(obs, p_t, r_t, grip)
            e.set_ghost("ghost_grip", p_t, wxyz(r_t))
            obs, *_ = env.step(perturb(a, phase, steps))
            rec.add(a, phase)
            if np.linalg.norm(delta[:3]) < tol[0] and np.degrees(np.linalg.norm(delta[3:])) < tol[1]:
                break
            errs.append((np.linalg.norm(delta[:3]), np.degrees(np.linalg.norm(delta[3:]))))
            if len(errs) > 10 and errs[-11][0] - errs[-1][0] < 0.0005 and errs[-11][1] - errs[-1][1] < 0.3:
                break

    # --- reach: up to a safe height, across to behind the mug, down, slide in horizontally

    def move(a, b, ra, rb, phase, step=REACH_STEP):
        """Gripper open, straight line a -> b in small steps while turning ra -> rb, then line up
        within 2 mm / 1 deg before going on.

        a, b   -> start and end positions (3,) m
        ra, rb -> start and end orientations, Rotation
        phase  -> label saved with each step
        step   -> m per control step along the line
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        turn = Slerp([0, 1], R.concatenate([ra, rb]))
        n = max(2, int(np.linalg.norm(b - a) / step) + 1, int(np.degrees((rb * ra.inv()).magnitude()) / 3))
        for s_ in np.linspace(0, 1, n)[1:]:
            drive(a + s_ * (b - a), turn(s_), -1, phase, 1)
        drive(b, rb, -1, phase, 60, tol=(0.002, 1.0))

    behind = grip_p0 - APPROACH * heading(Rg)
    p_start, r_start = obs["robot0_eef_pos"].copy(), R.from_matrix(R_start)
    z_safe = max(mug0[2] + SAFE_HEIGHT, p_start[2])
    move(p_start, [*p_start[:2], z_safe], r_start, r_start, "reach")
    move([*p_start[:2], z_safe], [*behind[:2], z_safe], r_start, r_grip, "reach")
    move([*behind[:2], z_safe], behind, r_grip, r_grip, "reach")
    move(behind, grip_p0, r_grip, r_grip, "reach", REACH_STEP / 2)  # slide in at half speed
    drive(grip_p0, r_grip, -1, "reach", 20, tol=(0.002, 1.0))
    for _ in range(12):
        drive(grip_p0, r_grip, 1, "grasp", 1)

    # --- follow: aim LEAD_S ahead; errors are measured against where the hand should be now
    n_follow = int(times[-1] / SPEED * CONTROL_HZ) + 1
    for k in range(n_follow):
        t_k = float(np.clip(k / CONTROL_HZ * SPEED, times[0], times[-1]))  # clip time now
        p_t, r_t = grip_target(t_k)                       # intended now (orange ghost)
        p_aim, r_aim = grip_target(t_k + LEAD_S * SPEED)  # what we command
        e.set_ghost("ghost_mug", [np.interp(t_k, times, sim_pos[:, j]) for j in range(3)], wxyz(slerp(t_k)))
        e.set_ghost("ghost_grip", p_t, wxyz(r_t))
        a, _ = action_to(obs, p_aim, r_aim, 1)
        obs, *_ = env.step(a)
        rec.add(a, "follow")

    # --- release: open, slide back out, lift, return to the start pose
    p_end, r_end = eef_pose(obs)
    mug_set = e.sim.data.get_joint_qpos(e.get_object(e.mug_name).joints[-1]).copy()  # end of follow, still held
    for _ in range(40):
        drive(p_end, r_end, -1, "release", 1)
        if np.abs(obs["robot0_gripper_qpos"]).min() > 0.038:  # fully open is 0.04 per finger
            break
    out = p_end - APPROACH * heading(r_end.as_matrix())
    move(p_end, out, r_end, r_end, "release", REACH_STEP / 2)
    move(out, [*out[:2], z_safe], r_end, r_end, "release")
    move([*out[:2], z_safe], [*p_start[:2], z_safe], r_end, r_start, "release")
    move([*p_start[:2], z_safe], p_start, r_start, r_start, "release")
    for _ in range(20):  # let droplets settle
        obs, *_ = env.step(np.r_[np.zeros(6), -1])

    in_plant = e.watering_stats()["target"]
    watered = bool(e._check_success())
    placed_cm = float(np.linalg.norm(mug_set[:3] - sim_pos[-1]) * 100)
    placed_deg = float(np.degrees((R.from_quat(mug_set[[4, 5, 6, 3]]) * sim_rot[-1].inv()).magnitude()))
    placed = placed_cm <= PLACED_CM and placed_deg <= PLACED_DEG
    env.close()

    res = dict(clip=clip, plant=prep["side"], tilt=tilt, in_plant=in_plant, watered=watered, placed=placed,
               success=watered and placed)
    (episodes.clip_dir(clip) / "robot.json").write_text(json.dumps(res, indent=2))
    rec.save(clip, prep["side"], SPEED, CONTROL_HZ)
    
    if VIDEO:
        playback.render(clip)
    
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    p.add_argument("--jobs", type=int, default=0, help="clips run in parallel (default: up to half the CPU cores)")
    args = p.parse_args()
    clips = [f"{c}~{i}" if i else c for c in args.clips or episodes.tracked() for i in range(SHIFTS + 1)]
    for r in episodes.parallel_map(run, clips, {}, args.jobs, "robot"):
        if "error" in r:  # crashed: record it as a failure
            (episodes.clip_dir(r["clip"]) / "robot.json").write_text(json.dumps(dict(r, success=False), indent=2))
            print(f"{r['clip']}: {r['error']}")
        elif not r["success"]:
            print(f"{r['clip']}: water {r['in_plant']}/{DROPLET['count']} in the {r['plant']} plant, mug put back: {'yes' if r['placed'] else 'no'}")


if __name__ == "__main__":
    main()
