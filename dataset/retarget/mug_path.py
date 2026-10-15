"""Tracked mug path (dataset/data/<group>/<clip>/track.npz) -> cleaned path -> sim coordinates.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation as R

from dataset.episodes import SETUP

from dataset.episodes import clip_dir
FLOOR_TO_SIM = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
POUR_DEG        = 30.0      # the mug tilted this far is pouring: a shifted replay pours where the human did, unshifted
LIFT_CM         = 1.0       # the mug this far above its start is off the table: the shift fades out / in while carried
FLIP_DEG        = 25.0      # an orientation this far from its smoothed neighbours is a tracking glitch
SMOOTH_FRAMES   = 2.0       # gaussian sigma, in video frames (60 fps)


def load(clip):
    """The clip's tracked path, as saved by ./run track. N = video frames (60 fps).

    clip -> clip name: "A1-0"
    returns dict: t (N,) s, ok (N,) bool (tag seen), mug_pos (N, 3) floor cm, mug_R (N, 3, 3),
            cam_R (3, 3) and cam_t (3,) cm (camera -> floor), K (3, 3), dist (5,), image_size (2,) px
    """
    d = np.load(clip_dir(clip) / "track.npz")
    return {k: d[k] for k in d.files}


def clean(track):
    """Fill gaps, drop orientation glitches, smooth.

    track -> dict from load()
    returns t (N,) s, pos (N, 3) floor cm, rot Rotation (N poses), kept (N,) bool (frames used, not filled in)
    """
    t = track["t"]
    seen = track["ok"]                                   # (N,) the tag was found in this frame
    quat = to_quaternions(track["mug_R"], seen)          # (N, 4), zeros where not seen

    # 1. Drop glitches: frames where mug orientation jumps away from their neighbours (the tag's pose flipping).
    neighbours  = median_filter(quat, size=(9, 1), mode="nearest")
    neighbours  /= np.linalg.norm(neighbours, axis=1, keepdims=True) + 1e-9
    kept        = seen & (angle_between(quat, neighbours) < FLIP_DEG)

    # 2. Fill the dropped and unseen frames by interpolating between the kept ones.
    pos     = fill_gaps(track["mug_pos"], kept)
    quat    = fill_gaps(quat, kept)

    # 3. Smooth out the remaining jitter.
    pos     = gaussian_filter1d(pos, SMOOTH_FRAMES, axis=0, mode="nearest")
    quat    = gaussian_filter1d(quat, SMOOTH_FRAMES, axis=0, mode="nearest")
    rot     = R.from_quat(quat / np.linalg.norm(quat, axis=1, keepdims=True))
    return t, pos, rot, kept


def to_quaternions(mats, seen):
    """Rotation matrices -> quaternions with continuous signs (q and -q are the same rotation; flipping
    signs to match the previous frame lets quaternions be filtered and interpolated like numbers).

    mats -> (N, 3, 3) rotation matrices
    seen -> (N,) bool, which rows are valid
    returns (N, 4) quaternions xyzw, zeros where not seen
    """
    quat = np.zeros((len(mats), 4))
    quat[seen] = R.from_matrix(mats[seen]).as_quat()
    for i in range(1, len(quat)):
        if seen[i] and np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1
    return quat


def angle_between(q1, q2):
    """Angle between two rotations, row by row.

    q1, q2 -> (N, 4) unit quaternions
    returns (N,) deg
    """
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=1)), 0, 1)))


def fill_gaps(x, kept):
    """Replace the rows that are not kept by linear interpolation between the kept ones.

    x    -> (N, D) values
    kept -> (N,) bool
    returns (N, D) copy
    """
    idx = np.arange(len(x))
    return np.column_stack([np.interp(idx, idx[kept], x[kept, k]) for k in range(x.shape[1])])


def rest_pose(t, pos, rot, seconds=0.3):
    """Mug pose while it still stands at its start spot (median over the first moments).

    t, pos, rot -> from clean(): (N,) s, (N, 3) floor cm, Rotation (N)
    seconds     -> how long counts as "the first moments"
    returns pos0 (3,) floor cm, rot0 Rotation (one pose)
    """
    n = max(1, int(np.searchsorted(t, t[0] + seconds)))
    return np.median(pos[:n], axis=0), rot[:n].mean()


def nearest_start(pos0):
    """The measured start spot (dataset/setup.json mug_starts) closest to where the mug stood.

    pos0 -> (3,) floor cm, from rest_pose()
    returns spot index 0-3, distance to it in cm
    """
    spots = np.array(SETUP["mug_starts"])
    i = int(np.argmin(np.linalg.norm(spots - pos0[:2], axis=1)))
    return i, float(np.linalg.norm(spots[i] - pos0[:2]))


def watered_plant(pos, rot):
    """Which pot the mug pours into: the one under the mouth at the strongest tilt.

    pos, rot -> from clean(): (N, 3) floor cm, Rotation (N)
    returns side "left" / "middle" / "right", max tilt deg, mouth-to-pot distance cm
    """
    tilt = np.degrees(np.arccos(np.clip(rot.as_matrix()[:, 2, 2], -1, 1)))
    i = int(np.argmax(tilt))
    mouth = pos[i] + rot[i].as_matrix()[:, 2] * SETUP["mug"]["height"]
    pots = {s: np.array(SETUP["pots"][s]) for s in ("left", "middle", "right")}
    side = min(pots, key=lambda s: np.linalg.norm(pots[s] - mouth[:2]))
    return side, float(tilt[i]), float(np.linalg.norm(pots[side] - mouth[:2]))


def to_sim(pos, rot, pos0, rot0, sim_pos0, sim_rot0):
    """Motion relative to the rest pose (floor, cm) -> sim world, anchored at the sim mug's start.

    pos, rot           -> clip path: (N, 3) floor cm, Rotation (N)
    pos0, rot0         -> clip rest pose: (3,) floor cm, Rotation
    sim_pos0, sim_rot0 -> sim mug's start: (3,) m, Rotation
    returns sim_pos (N, 3) m, sim_rot Rotation (N)
    """
    rel = (pos - pos0) / 100.0 @ FLOOR_TO_SIM.T
    rel[:, 2] = np.maximum(rel[:, 2], 0.0)  # never push the mug below its start height (the table)
    sim_pos = sim_pos0 + rel
    M = R.from_matrix(FLOOR_TO_SIM)
    sim_rot = M * (rot * rot0.inv()) * M.inv() * sim_rot0
    return sim_pos, sim_rot


def camera_in_sim(track, pos0, sim_pos0):
    """The phone's pose as a MuJoCo camera, same placement relative to the mug as in the video.

    track    -> dict from load() (uses cam_R, cam_t, K, image_size)
    pos0     -> clip rest pose (3,) floor cm
    sim_pos0 -> sim mug's start (3,) m
    returns cam_pos (3,) m, quat (4,) wxyz, fovy deg
    """
    cam_pos = sim_pos0 + (track["cam_t"] - pos0) / 100.0 @ FLOOR_TO_SIM.T
    # OpenCV camera axes (x right, y down, z forward) -> MuJoCo (x right, y up, z backward)
    R_mj = FLOOR_TO_SIM @ track["cam_R"] @ np.diag([1.0, -1.0, -1.0])
    quat = R.from_matrix(R_mj).as_quat()[[3, 0, 1, 2]]
    fovy = float(np.degrees(2 * np.arctan(track["image_size"][1] / 2 / track["K"][1, 1])))
    return cam_pos, quat, fovy


def prepare(clip):
    """Everything about a clip that doesn't need the sim: cleaned path, rest pose, start spot, plant.

    clip -> clip name, e.g. "A1-0"
    returns prep dict: clip, track (load()), t (N,) s, pos (N, 3) floor cm, rot Rotation (N), kept (N,) bool,
            pos0 (3,) cm, rot0 Rotation, spot 0-3, spot_err cm, side plant name, max_tilt deg, mouth_err cm
    """
    track = load(clip)
    t, pos, rot, kept = clean(track)
    pos0, rot0 = rest_pose(t, pos, rot)
    spot, spot_err = nearest_start(pos0)
    side, max_tilt, mouth_err = watered_plant(pos, rot)
    return dict(clip=clip, track=track, t=t, pos=pos, rot=rot, kept=kept, pos0=pos0, rot0=rot0, spot=spot,
                spot_err=spot_err, side=side, max_tilt=max_tilt, mouth_err=mouth_err)


def shift_weight(t, pos, rot):
    """How much of the start shift each pose of the path keeps: all of it on the table (pick-up and put-back), none
    while pouring (the pour stays over the pot)

    t        -> (N,) s
    pos, rot -> the path: (N, 3) m, z up; Rotation (N)
    returns (N,) weights 0-1
    """
    up = np.flatnonzero(pos[:, 2] - pos[0, 2] > LIFT_CM / 100)
    pour = np.flatnonzero(np.degrees(np.arccos(np.clip(rot.as_matrix()[:, 2, 2], -1, 1))) > POUR_DEG)
    w = np.ones(len(t))
    if not len(up) or not len(pour):
        return w  # never lifted or never poured: the whole path moves, as before
    lift, a, b, down = min(up[0], pour[0]), pour[0], pour[-1], max(up[-1], pour[-1])
    ease = lambda x: x * x * (3 - 2 * x)  # smoothstep: no jump in speed where the blend starts or ends
    w[lift:a] = 1 - ease((t[lift:a] - t[lift]) / max(t[a] - t[lift], 1e-6))
    w[a:b + 1] = 0.0
    w[b + 1:down + 1] = ease((t[b + 1:down + 1] - t[b]) / max(t[down] - t[b], 1e-6))
    return w


def place_in_sim(prep, problem, settle_steps=200, shift=(0.0, 0.0)):
    """Put the sim mug on the clip's start spot (pots on their measured spots), let the droplets
    settle, and return the clip's path in sim world coordinates.

    prep        -> dict from prepare()
    problem     -> the LIBERO problem (env.env) of a watering scene
    settle_steps -> physics steps for the droplets to come to rest
    shift        -> (2,) m added to the start spot; the pick-up and put-back move with it, the pour doesn't (shift_weight)
    returns times (N,) s from 0, sim_pos (N, 3) m, sim_rot Rotation (N), sim_pos0 (3,) m (sim mug's start)
    """
    from sim.watering.bddl import MUG_STARTS, POTS
    problem.set_layout(np.add(MUG_STARTS[prep["spot"]], shift), POTS)
    robot = problem.robots[0]
    q_arm = problem.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
    for _ in range(settle_steps):
        # no controller runs here, so hold the arm still or it sags under gravity
        problem.sim.data.qpos[robot._ref_joint_pos_indexes] = q_arm
        problem.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
        problem.sim.step()
    q = problem.sim.data.get_joint_qpos(problem.get_object(problem.mug_name).joints[-1])
    sim_pos0, sim_rot0 = q[:3].copy(), R.from_quat(q[[4, 5, 6, 3]])
    return (*sim_path(prep, sim_pos0, sim_rot0, shift), sim_pos0)


def sim_path(prep, sim_pos0, sim_rot0, shift=(0.0, 0.0)):
    """The clip's path in sim world coordinates for a mug starting at sim_pos0, the pour kept where the human
    poured (shift_weight)

    prep               -> dict from prepare()
    sim_pos0, sim_rot0 -> sim mug's start: (3,) m, Rotation
    shift              -> (2,) m the start is away from the clip's spot
    returns times (N,) s from 0, sim_pos (N, 3) m, sim_rot Rotation (N)
    """
    sim_pos, sim_rot = to_sim(prep["pos"], prep["rot"], prep["pos0"], prep["rot0"], sim_pos0, sim_rot0)
    t = prep["t"] - prep["t"][0]
    sim_pos -= np.outer(1 - shift_weight(t, sim_pos, sim_rot), [*shift, 0.0])  # to_sim moved every pose by the shift
    return t, sim_pos, sim_rot
