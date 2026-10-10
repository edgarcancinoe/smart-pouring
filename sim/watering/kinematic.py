"""Move the mug directly (no robot) along a pour tracked from video.

The mug is driven like a puppet: every physics step we set its pose and matching velocity,
so droplets feel a smooth motion. The robot is frozen in place so it doesn't sag.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp



class _KinematicMug:
    """Drives the mug along self.pose(t) -> (x y z, w x y z) in world coordinates."""

    def __init__(self, problem):
        self.e, self.sim = problem, problem.sim
        self.joint = problem.get_object(problem.mug_name).joints[-1]
        self.dt = self.sim.model.opt.timestep
        self.robot_q = self.sim.data.qpos[problem.robots[0]._ref_joint_pos_indexes].copy()
        self.t, self.prev = 0.0, self.sim.data.get_joint_qpos(self.joint).copy()
        self.duration = 0.0

    def pose(self, t):
        raise NotImplementedError

    def freeze_robot(self):
        idx = self.e.robots[0]._ref_joint_pos_indexes
        self.sim.data.qpos[idx] = self.robot_q
        self.sim.data.qvel[self.e.robots[0]._ref_joint_vel_indexes] = 0

    def step(self):
        """Advance one physics step along the path (holds the last pose when finished)."""
        qpos = self.pose(self.t)
        prev_rot = R.from_quat(self.prev[[4, 5, 6, 3]])
        rot = R.from_quat(qpos[[4, 5, 6, 3]])
        vel = np.r_[(qpos[:3] - self.prev[:3]) / self.dt, (rot * prev_rot.inv()).as_rotvec() / self.dt]
        self.sim.data.set_joint_qpos(self.joint, qpos)
        self.sim.data.set_joint_qvel(self.joint, vel)
        self.freeze_robot()
        self.sim.step()
        self.prev, self.t = qpos, self.t + self.dt


class KinematicReplay(_KinematicMug):
    """Follow a recorded path: times (s), world positions (N,3) and orientations (scipy Rotation, N).

    ghost_robot turns off the frozen robot's collisions so a human path can't hit it.
    """

    def __init__(self, problem, times, positions, rotations, ghost_robot=True):
        super().__init__(problem)
        self.times, self.positions = np.asarray(times) - times[0], np.asarray(positions)
        self.slerp = Slerp(self.times, rotations)
        self.duration = float(self.times[-1])
        self.prev = self.pose(0.0)
        if ghost_robot:
            m = self.sim.model._model
            for g in range(m.ngeom):
                body = m.body(m.geom_bodyid[g]).name
                if body.startswith(("robot0_", "gripper0_", "mount0_")):
                    m.geom_contype[g] = m.geom_conaffinity[g] = 0

    def pose(self, t):
        t = float(np.clip(t, 0.0, self.duration))
        pos = np.array([np.interp(t, self.times, self.positions[:, k]) for k in range(3)])
        return np.r_[pos, self.slerp(t).as_quat()[[3, 0, 1, 2]]]
