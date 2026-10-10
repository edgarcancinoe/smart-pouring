"""Arm postures for the watering task: inverse kinematics and grip geometry.

The gripper holds the mug from the robot's side, tilted down, fingers closing left/right (radial_grip).
Which arm posture holds that grip decides whether joint 7 (the pouring roll, +-166 deg) has room for
the pour, so postures are searched by IK (best_family).
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

SITE = "gripper0_grip_site"   # position: the point between the fingertips (= obs robot0_eef_pos)
HAND = "robot0_right_hand"    # orientation: the frame the controller steers (= obs robot0_eef_quat)
# In HAND: fingers close along y, the wrist camera sits on the +x side, z points out of the fingers.
GRIP_HEIGHT = 0.03   # m above the mug's bottom
GRIP_BACK = 0.0175   # m toward the robot for tilted grips, so the palm clears the mug's wider rim
LEVEL_DEG = 5        # a hand pointing down less than this counts as level
# Where the hand starts every episode: 20 cm above the table, in front of the robot.
READY_POS = np.array([-0.20, 0.0, 1.10])
# Level grip pointing +x (world), as gripper axes in columns.
LEVEL_GRIP = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float)


class Arm:
    """Kinematics of the robot in a LIBERO problem (on a private MjData copy; the sim is untouched)."""

    def __init__(self, problem):
        self.m = problem.sim.model._model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = problem.sim.data._data.qpos
        robot = problem.robots[0]
        self.idx = np.array(robot._ref_joint_pos_indexes)
        self.dof = np.array(robot._ref_joint_vel_indexes)
        self.lo, self.hi = self.m.jnt_range[[self.m.joint(f"robot0_joint{i}").id for i in range(1, 8)]].T
        self.site = self.m.site(SITE).id
        self.hand = self.m.body(HAND).id
        self._jp, self._jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))

    def fk(self, q):
        """Grip point position and hand orientation (the controller's frames) for joint angles q."""
        self.d.qpos[self.idx] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.xmat[self.hand].reshape(3, 3).copy()

    def ik(self, pos, rot, q0, iters=200, max_step=0.1, rest=None, null_gain=0.02):
        """Damped least squares from q0 (small steps so it stays in q0's posture family).
        Returns q, position error (m), orientation error (deg)."""
        q = np.array(q0, float)
        rest = q.copy() if rest is None else rest
        for _ in range(iters):
            p, Rm = self.fk(q)
            err = np.r_[pos - p, (R.from_matrix(rot) * R.from_matrix(Rm).inv()).as_rotvec()]
            if np.linalg.norm(err[:3]) < 2e-4 and np.linalg.norm(err[3:]) < 2e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            J = np.vstack([self._jp[:, self.dof], self._jr[:, self.dof]])
            Jinv = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
            dq = Jinv @ err + (np.eye(7) - Jinv @ J) @ (null_gain * (rest - q))
            n = np.abs(dq).max()
            q = np.clip(q + dq * min(1.0, max_step / (n + 1e-12)), self.lo + 1e-3, self.hi - 1e-3)
        p, Rm = self.fk(q)
        return q, float(np.linalg.norm(pos - p)), float(np.degrees((R.from_matrix(rot) * R.from_matrix(Rm).inv()).magnitude()))

    def margin_deg(self, q):
        return np.degrees(np.minimum(q - self.lo, self.hi - q))


def follow_path(arm, targets, q_start):
    """IK along a sequence of (pos, rot matrix) targets, each step starting from the previous answer.
    Returns per-step q, position error (cm), rotation error (deg)."""
    qs, pe, re = [], [], []
    q = q_start
    for pos, rot in targets:
        q, e_p, e_r = arm.ik(pos, rot, q, iters=60)
        qs.append(q)
        pe.append(e_p * 100)
        re.append(e_r)
    return np.array(qs), np.array(pe), np.array(re)


def grip_rotation(tilt_deg=0.0, turn_deg=0.0):
    """Grip orientation: LEVEL_GRIP tilted to point tilt_deg downward, then turned turn_deg about
    the vertical (positive = pointing toward the robot's left). Fingers still close horizontally."""
    r = R.from_euler("z", turn_deg, degrees=True) * R.from_euler("y", tilt_deg, degrees=True)
    return (r * R.from_matrix(LEVEL_GRIP)).as_matrix()


def random_postures(arm, n, rng):
    return [rng.uniform(arm.lo + 0.1, arm.hi - 0.1) for _ in range(n)]


def ready_posture(arm, rot, q_near, n_random=40, seed=0, min_margin_deg=5.0):
    """Joint angles putting the hand exactly at READY_POS with orientation rot, clear of joint
    limits, and as close as possible to q_near (e.g. the grab posture, so the reach stays short).
    Returns q (7,) rad; raises if no clean solution exists."""
    rng = np.random.default_rng(seed)
    p_near, _ = arm.fk(q_near)
    walk = follow_path(arm, [(p_near + s * (READY_POS - p_near), rot) for s in np.linspace(0, 1, 30)], q_near)[0][-1]
    seeds = [walk, q_near] + random_postures(arm, n_random, rng)
    good = []
    for s in seeds:
        q, e_p, e_r = arm.ik(READY_POS, rot, s, iters=400)
        if e_p < 0.002 and e_r < 0.5 and arm.margin_deg(q).min() > min_margin_deg:
            good.append(q)
    if not good:
        raise RuntimeError("no ready pose found within joint limits; try another --tilt")
    return min(good, key=lambda q: np.abs(q - q_near).max())


def radial_grip(mug_xy, base_xy, tilt_deg, shoulder_offset=0.08):
    """Hand orientation for grabbing a mug: tilted tilt_deg down, pointing at the mug along the line
    from a point shoulder_offset beside the base centre, on the mug's side."""
    ref = np.asarray(base_xy[:2], float) + [0.0, shoulder_offset * np.sign(mug_xy[1] - base_xy[1])]
    return grip_rotation(tilt_deg, float(np.degrees(np.arctan2(mug_xy[1] - ref[1], mug_xy[0] - ref[0]))))


def best_family(arm, grab_pos, grab_rot, targets, n_random=40, keep=4, seed=0):
    """Arm posture at the grab pose from which the whole path `targets` [(pos, rot), ...] can be followed
    with the most joint room. Returns (q_grab, min joint room deg); room -1 if nothing follows the path."""
    rng = np.random.default_rng(seed)
    sols = []
    for q_seed in random_postures(arm, n_random, rng):
        q, e_p, e_r = arm.ik(grab_pos, grab_rot, q_seed, iters=250)
        if e_p < 0.003 and e_r < 1:
            sols.append((q, arm.margin_deg(q).min()))
    best_q, best_room = (sols[0][0] if sols else None), -1.0
    for q, _ in sorted(sols, key=lambda s: -s[1])[:keep]:
        qs, pe, re = follow_path(arm, targets, q)
        if pe.max() < 1.0 and re.max() < 3:
            room = min(arm.margin_deg(q).min(), min(arm.margin_deg(x).min() for x in qs))
            if room > best_room:
                best_q, best_room = q, room
    return best_q, best_room


def heading(rot):
    """Horizontal unit vector of the hand's pointing direction."""
    h = np.asarray(rot)[:, 2] * [1, 1, 0]
    return h / np.linalg.norm(h)


def grab_point(mug_pos, rot):
    """Where the grip point goes for a mug standing at mug_pos (bottom centre) and hand orientation rot."""
    tilted = -np.asarray(rot)[2, 2] > np.sin(np.radians(LEVEL_DEG))  # pointing axis has a downward part
    return np.asarray(mug_pos, float) + [0, 0, GRIP_HEIGHT] - (GRIP_BACK if tilted else 0.0) * heading(rot)
