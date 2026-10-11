"""Arm postures for the watering task: grip geometry and inverse kinematics.

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
# Level grip pointing +x (world, away from the robot), fingers horizontal, camera on top: hand axes as columns.
LEVEL_GRIP = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float)
GRIP_HEIGHT = 0.03   # m above the mug's bottom
GRIP_BACK = 0.0175   # m toward the robot for tilted grips, so the palm clears the mug's wider rim
LEVEL_DEG = 5        # a hand pointing down less than this counts as level
# Arm joints (rad) every episode starts from, demos and evaluation alike: hand 20 cm above the table.
START_Q = np.array([-1.9605, 1.3877, 1.8972, -2.7726, -2.0429, 2.1932, 1.9253])
GRAB_TOL_M, GRAB_TOL_DEG = 0.003, 1.0     # an IK answer counts as reaching the grab pose within this
PATH_TOL_CM, PATH_TOL_DEG = 1.0, 3.0      # ... and as staying on the path within this, at every target


# --- grip geometry ---

def grip_rotation(tilt_deg, turn_deg):
    """Hand orientation (3, 3): LEVEL_GRIP pitched tilt_deg down, then turned turn_deg about the vertical
    (positive = toward the robot's left). The fingers stay horizontal."""
    r = R.from_euler("z", turn_deg, degrees=True) * R.from_euler("y", tilt_deg, degrees=True)
    return (r * R.from_matrix(LEVEL_GRIP)).as_matrix()


def radial_grip(mug_xy, base_xy, tilt_deg, shoulder_offset):
    """Hand orientation (3, 3) for grabbing a mug: tilted tilt_deg down, pointing at the mug along the line
    from the "shoulder", a point shoulder_offset (m) beside the base centre on the mug's side."""
    shoulder = np.asarray(base_xy[:2], float) + [0.0, shoulder_offset * np.sign(mug_xy[1] - base_xy[1])]
    return grip_rotation(tilt_deg, float(np.degrees(np.arctan2(mug_xy[1] - shoulder[1], mug_xy[0] - shoulder[0]))))


def heading(rot):
    """Horizontal unit vector (3,) of the hand's pointing direction."""
    h = np.asarray(rot)[:, 2] * [1, 1, 0]
    return h / np.linalg.norm(h)


def grab_point(mug_pos, rot):
    """Where the grip point goes (3,) m, for a mug standing at mug_pos (bottom centre) and hand orientation rot."""
    tilted = -np.asarray(rot)[2, 2] > np.sin(np.radians(LEVEL_DEG))  # pointing axis has a downward part
    return np.asarray(mug_pos, float) + [0, 0, GRIP_HEIGHT] - (GRIP_BACK if tilted else 0.0) * heading(rot)


# --- kinematics ---

def rotation_error(rot, Rm):
    """Rotation (scipy) taking orientation Rm to rot, both (3, 3)."""
    return R.from_matrix(rot) * R.from_matrix(Rm).inv()


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
        """Grip point position (3,) m and hand orientation (3, 3) for joint angles q (7,) rad."""
        self.d.qpos[self.idx] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.xmat[self.hand].reshape(3, 3).copy()

    def ik(self, pos, rot, q0, iters=200, max_step=0.1, null_gain=0.02):
        """Joint angles reaching a hand pose, by damped least squares from q0. Small steps and a pull back
        toward q0 keep the answer in q0's posture family.

        pos, rot  -> target: (3,) m, (3, 3)
        q0        -> (7,) rad starting posture
        iters     -> max iterations
        max_step  -> rad: largest joint change per iteration
        null_gain -> strength of the pull toward q0 (in the joint motions that don't move the hand)
        returns q (7,) rad, position error m, orientation error deg
        """
        q = np.array(q0, float)
        rest = q.copy()
        for _ in range(iters):
            p, Rm = self.fk(q)
            err = np.r_[pos - p, rotation_error(rot, Rm).as_rotvec()]
            if np.linalg.norm(err[:3]) < 2e-4 and np.linalg.norm(err[3:]) < 2e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            J = np.vstack([self._jp[:, self.dof], self._jr[:, self.dof]])  # (6, 7): joint speeds -> hand motion
            Jinv = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))         # damped pseudo-inverse
            dq = Jinv @ err + (np.eye(7) - Jinv @ J) @ (null_gain * (rest - q))
            q = np.clip(q + dq * min(1.0, max_step / (np.abs(dq).max() + 1e-12)), self.lo + 1e-3, self.hi - 1e-3)
        p, Rm = self.fk(q)
        return q, float(np.linalg.norm(pos - p)), float(np.degrees(rotation_error(rot, Rm).magnitude()))

    def margin_deg(self, q):
        """How far each joint (7,) is from its nearest limit, deg."""
        return np.degrees(np.minimum(q - self.lo, self.hi - q))


def random_postures(arm, n, rng):
    """n random joint configurations (7,) rad, 0.1 rad inside the limits."""
    return [rng.uniform(arm.lo + 0.1, arm.hi - 0.1) for _ in range(n)]


def follow_path(arm, targets, q_start):
    """IK along hand poses in order, each step starting from the previous answer (stays in one posture family).

    targets -> [(pos (3,), rot (3, 3)), ...]
    q_start -> (7,) rad
    returns qs (n, 7) rad, position errors (n,) cm, orientation errors (n,) deg
    """
    qs, err_cm, err_deg = [], [], []
    q = q_start
    for pos, rot in targets:
        q, e_m, e_deg = arm.ik(pos, rot, q, iters=60)
        qs.append(q)
        err_cm.append(e_m * 100)
        err_deg.append(e_deg)
    return np.array(qs), np.array(err_cm), np.array(err_deg)


def best_family(arm, grab_pos, grab_rot, targets, n_random=40, keep=4, seed=0):
    """The arm posture at the grab from which the whole path can be followed with the most joint room.

    arm                -> Arm
    grab_pos, grab_rot -> grab pose: (3,) m, (3, 3) rotation matrix
    targets            -> [(pos (3,), rot (3, 3)), ...] hand poses along the path, in order
    n_random           -> random seed postures tried for the grab
    keep               -> how many of the grab solutions (most joint room first) are tried on the path
    returns q_grab (7,) rad, room deg: the smallest joint-limit margin over the grab and the whole path
            (room -1 if no grab solution follows the path)
    """
    rng = np.random.default_rng(seed)

    # 1. Ways to grab: IK to the grab pose from many random postures (each lands in some posture family).
    grabs = []  # (q, joint room at the grab)
    for q_seed in random_postures(arm, n_random, rng):
        q, err_m, err_deg = arm.ik(grab_pos, grab_rot, q_seed, iters=250)
        if err_m < GRAB_TOL_M and err_deg < GRAB_TOL_DEG:
            grabs.append((q, arm.margin_deg(q).min()))

    # 2. Follow the path from the roomiest grabs; keep the one whose worst moment has the most room.
    best_q, best_room = (grabs[0][0] if grabs else None), -1.0
    for q_grab, _ in sorted(grabs, key=lambda g: -g[1])[:keep]:
        qs, err_cm, err_deg = follow_path(arm, targets, q_grab)
        if err_cm.max() >= PATH_TOL_CM or err_deg.max() >= PATH_TOL_DEG:
            continue  # this posture can't stay on the path
        room = min(arm.margin_deg(q_grab).min(), min(arm.margin_deg(q).min() for q in qs))
        if room > best_room:
            best_q, best_room = q_grab, room
    return best_q, best_room
