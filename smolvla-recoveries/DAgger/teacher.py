"""The demo as a teacher that can take over from any state: follows the demo's hand path (reach -> carry -> release),
aiming a few poses ahead of the nearest one. See README.md.

Usage: python teacher.py   (self-check of the path search; the teacher in the sim: ./run dagger-check)
"""
import json
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from dataset import episodes
from dataset.retarget import mug_path as mp
from dataset.retarget.robot_replay import LEAD_S, SHOULDER_OFFSET, SPEED, action_to, eef_pose
from dataset.sim_runs import playback
from evaluate import mug_pose
from sim.watering import CONTROL_HZ
from sim.watering.bddl import MUG_STARTS
from sim.watering.robot import Arm, grab_point, radial_grip

DEG_PER_CM = 10.0        # an angle error this big counts as 1 cm off the path
ON_PATH_CM = 1.0         # within this the teacher moves on along the path; further off it first goes back to it
NEAR_MUG_CM = 6.0        # reach: off the path this close to the grip point (beside the mug): back out, not sideways
LINED_UP = (0.002, 1.0)  # (m, deg) close enough to the grip point to close the gripper, as robot_replay
SETTLE_CM = 0.5          # ... or after SETTLE_STEPS this close to it anyway, as robot_replay
SETTLE_STEPS = 20
CLOSE_STEPS = 12         # control steps the gripper closes for, as robot_replay
CLOSING_SPEED = 0.005    # m/s the fingers move together at, at least, to count as closing (they close at ~0.04)
EMPTY_CM = 2.0           # fingers closer than this, not opening, hold nothing (open: 7.9 cm, mug: 5.4, start pose: 1.6)
HELD_CM = (4.0, 7.0)     # fingers stopped this far apart, at the grip point, hold the mug
WINDOW_S = (0.5, 2.0)    # seconds of path behind / ahead of the last progress searched for the nearest pose
PUT_BACK_CM = 5.0        # carry: the mug this close to where it is set down is the put-back (dagger.GATE: tighter)
OPENED_CM = 0.3          # carry: fingers this much wider than at the grasp are letting go
DOWN_CM = 1.0            # carry: the mug this close above where it is set down is down: only then let go
OPEN_CM = 7.8            # release: fingers this far apart are open, as robot_replay
OPEN_STEPS = 40          # ... or after this many control steps anyway

DT = SPEED / CONTROL_HZ                    # clip seconds per control step, as robot_replay plays the pour
LEAD = 1 + round(LEAD_S * SPEED / DT)      # path steps the teacher aims ahead: one step on, plus robot_replay's LEAD_S
WINDOW = (round(WINDOW_S[0] * CONTROL_HZ), round(WINDOW_S[1] * CONTROL_HZ))  # in path steps


def demo_tilt(plant, spot):
    """Most used grip tilt for this plant + spot, deg."""
    tilts = [json.loads((episodes.clip_dir(c) / "robot.json").read_text()).get("tilt") for c in episodes.having("robot.json")
             if (p := episodes.parse(c)) and (p["plant"], p["position"]) == (plant, spot)]
    tilts = [t for t in tilts if t is not None]
    if not tilts:
        raise SystemExit(f"no robot run of {plant} spot {spot} to take the grip tilt from: ./run robot first")
    return Counter(tilts).most_common(1)[0][0]


def demo_clip(plant, spot, tilt):
    """First successful recording of this plant + spot at this tilt: the demo the teacher follows."""
    for c in episodes.having(playback.FILE):
        p = episodes.parse(c)
        if p and "~" not in c and (p["plant"], p["position"]) == (plant, spot):
            r = json.loads((episodes.clip_dir(c) / "robot.json").read_text())
            if r.get("success") and r.get("tilt") == tilt:
                return c
    raise SystemExit(f"no saved successful robot run of {plant} spot {spot} at {tilt} deg: ./run robot first")


def nearest_sample(p, r, pos, rot, k, window=WINDOW):
    """Nearest path pose to the hand, searched only around k (the carry path crosses itself).

    p, r     -> hand pose
    pos, rot -> path
    k        -> last nearest index
    window   -> (behind, ahead) poses searched
    returns index, cm off
    """
    i = np.arange(max(0, k - window[0]), min(len(pos), k + window[1] + 1))
    cm = np.maximum(100 * np.linalg.norm(pos[i] - p, axis=1),
                    np.degrees((rot[i] * r.inv()).magnitude()) / DEG_PER_CM)
    j = int(np.argmin(cm))
    return int(i[j]), float(cm[j])


def bend(pos, rot, dp, dr, w):
    """Path shifted by dp and turned by dr, each pose by its share w (0-1)."""
    return pos + w[:, None] * dp, R.from_rotvec(w[:, None] * dr) * rot


def ramp(pos):
    """0 -> 1 along the path (smoothstep by distance): bend share per pose."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    x = d / max(d[-1], 1e-9)
    return x * x * (3 - 2 * x)


def degrees(ra, rb):
    """Angle between two rotations, deg."""
    return float(np.degrees((ra * rb.inv()).magnitude()))


def width_cm(obs):
    """Finger gap, cm."""
    q = obs["robot0_gripper_qpos"]
    return float(q[0] - q[1]) * 100


class Teacher:
    """The demo for one scene. observe(obs) every step, whoever drives; act(obs) when the teacher drives.

    e     -> env.env, mug on its start spot
    plant -> "left" / "middle" / "right"
    spot  -> start spot 1-4
    """

    def __init__(self, e, plant, spot):
        self.e = e
        self.base_xy = e.sim.data.body_xpos[e.sim.model._model.body("robot0_base").id][:2].copy()
        self.tilt = demo_tilt(plant, spot)
        clip = demo_clip(plant, spot, self.tilt)

        # the demo robot run's hand, every control step: its reach and release
        s = playback.load(clip)
        arm, idx = Arm(e), e.robots[0]._ref_joint_pos_indexes
        hand = [arm.fk(q[idx]) for q in s["qpos"]]
        pos, rot = np.array([h[0] for h in hand]), R.from_matrix([h[1] for h in hand])
        reach, release = np.flatnonzero(s["phase"] == "reach"), np.flatnonzero(s["phase"] == "release")
        self.reach_pos, self.reach_rot = pos[reach], rot[reach]
        self.release_pos, self.release_rot = pos[release], rot[release]

        # the carry's mug path: the recording's pour, the mug starting where it stands now, the pour where the human poured
        prep = mp.prepare(clip)
        mug0, rot0, _ = mug_pose(e)
        t, mpos, mrot = mp.sim_path(prep, mug0, rot0, mug0[:2] - np.asarray(MUG_STARTS[prep["spot"]])[:2])
        ts = np.arange(0.0, t[-1], DT)  # one pose per control step, as robot_replay plays it
        self.mug_pos = np.column_stack([np.interp(ts, t, mpos[:, j]) for j in range(3)])
        self.mug_rot = Slerp(t, mrot)(ts)
        far = np.flatnonzero(np.linalg.norm(self.mug_pos - self.mug_pos[-1], axis=1) > PUT_BACK_CM / 100)
        self.put_back = far[-1] + 1 if len(far) else 0  # carry poses from here on: the put-back

        self.stage = "reach"
        self.ref = self.reach_ref()  # the current stage's path: (positions (M, 3) m, Rotation (M))
        self.k = 0                   # index of its pose nearest the hand (on the path: at least the demo's pace)
        self.off = 0.0               # cm the hand is off that pose (nearest_sample)
        self.closed = 0              # reach: control steps the teacher has been closing the gripper for
        self.settle = 0              # reach: control steps spent near the grip point without lining up
        self.opening = 0             # release: control steps spent opening

    @property
    def gate(self):
        """Stage for dagger.GATE: the carry's put-back gets its own."""
        return "put-back" if self.stage == "carry" and self.k >= self.put_back else self.stage

    @property
    def busy(self):
        """Closing or opening the gripper: don't hand back yet."""
        return 0 < self.closed < CLOSE_STEPS or (self.stage == "release" and 0 < self.opening < OPEN_STEPS)

    def grip(self):
        """Grip pose for the mug where it is now."""
        mug, *_ = mug_pose(self.e)
        Rg = radial_grip(mug[:2], self.base_xy, self.tilt, SHOULDER_OFFSET)
        return grab_point(mug, Rg), R.from_matrix(Rg)

    def reach_ref(self):
        """Demo reach, bent to end on this mug's grip."""
        grip, rg = self.grip()
        dp, dr = grip - self.reach_pos[-1], (rg * self.reach_rot[-1].inv()).as_rotvec()
        return bend(self.reach_pos, self.reach_rot, dp, dr, ramp(self.reach_pos))

    def observe(self, obs):
        """Update stage and progress along the path."""
        p, r = eef_pose(obs)
        w = width_cm(obs)
        if self.stage == "reach":
            self.ref = self.reach_ref()  # the mug may have been nudged
            stopped = abs(obs["robot0_gripper_qvel"][0] - obs["robot0_gripper_qvel"][1]) < CLOSING_SPEED
            if stopped and np.linalg.norm(p - self.grip()[0]) < 0.02 and HELD_CM[0] < w < HELD_CM[1]:
                # the mug is in the hand (whoever grasped it): carry it with the real grip, as robot_replay's offset
                mug, mug_r, _ = mug_pose(self.e)
                off_p, off_r = mug_r.inv().apply(p - mug), mug_r.inv() * r
                self.ref = (self.mug_pos + self.mug_rot.apply(off_p), self.mug_rot * off_r)
                self.stage, self.k, self.held_cm = "carry", 0, w
        elif self.stage == "carry":
            down = mug_pose(self.e)[0][2] - self.mug_pos[-1][2] < DOWN_CM / 100
            if self.k >= len(self.ref[0]) - 1 and down:  # the mug is back on its spot: the demo's release, from where the hand is
                dp, dr = p - self.release_pos[0], (r * self.release_rot[0].inv()).as_rotvec()
                self.ref = bend(self.release_pos, self.release_rot, dp, dr, 1 - ramp(self.release_pos))
                self.stage, self.k = "release", 0
            elif w < EMPTY_CM:  # the mug was dropped: grasp it again where it lies
                self.stage, self.k, self.closed, self.settle = "reach", 0, 0, 0
                self.ref = self.reach_ref()
        k, self.off = nearest_sample(p, r, *self.ref, self.k)
        on = self.off <= ON_PATH_CM
        self.k = min(max(k, self.k + 1), len(self.ref[0]) - 1) if on else k  # on it: at least the demo's pace

    def off_track_cm(self, obs):
        """cm off the path. Reach while closing / empty: cm off the grip point (closing elsewhere is a mistake).
        Carry while opening: off (letting go before the mug is down is a mistake)."""
        p, r = eef_pose(obs)
        v = obs["robot0_gripper_qvel"][0] - obs["robot0_gripper_qvel"][1]  # > 0: opening
        if self.stage == "carry" and width_cm(obs) > self.held_cm + OPENED_CM:
            return np.inf
        if self.stage == "reach" and (v < -CLOSING_SPEED or (width_cm(obs) < EMPTY_CM and v < CLOSING_SPEED)):
            grip, rg = self.grip()
            return max(100 * np.linalg.norm(p - grip), degrees(r, rg) / DEG_PER_CM)
        return self.off

    def act(self, obs):
        """Next action from here: action (7,), phase label."""
        p, r = eef_pose(obs)
        pos, rot = self.ref
        k, off = self.k, self.off
        ahead = min(k + LEAD, len(pos) - 1)
        if self.stage == "carry":
            i = ahead if off <= ON_PATH_CM else k  # on the pour: on along it; off: back to it
            return action_to(obs, pos[i], rot[i], 1)[0], "follow"
        if self.stage == "release":
            if width_cm(obs) < OPEN_CM and self.opening < OPEN_STEPS:  # let go first, where the hand is
                self.opening += 1
                return action_to(obs, pos[0], rot[0], -1)[0], "release"
            self.opening = OPEN_STEPS
            i = ahead if off <= ON_PATH_CM else k
            return action_to(obs, pos[i], rot[i], -1)[0], "release"

        # reach: at the grip point close (and keep closing once started); else along the demo's reach
        grip, rg = self.grip()
        if self.closed >= CLOSE_STEPS and width_cm(obs) < EMPTY_CM:  # closed on nothing: approach again
            self.closed = self.settle = 0
        near = np.linalg.norm(p - grip)
        self.settle += near < SETTLE_CM / 100
        if self.closed or (near < LINED_UP[0] and degrees(r, rg) < LINED_UP[1]) or self.settle >= SETTLE_STEPS:
            self.closed += 1
            return action_to(obs, grip, rg, 1)[0], "grasp"
        if off <= ON_PATH_CM:
            i = ahead
        elif near < NEAR_MUG_CM / 100:  # off the path beside the mug: back out along the slide-in
            i = max(k - LEAD, 0)
        else:
            i = k
        return action_to(obs, pos[i], rot[i], -1)[0], "reach"


if __name__ == "__main__":
    # out and back over the same line (like a carry to the pot and back): the window keeps the right leg
    pos = np.r_[np.linspace([0, 0, 0], [0.2, 0, 0], 21), np.linspace([0.2, 0, 0], [0, 0, 0], 21)]
    rot = R.identity(len(pos))
    hand = np.array([0.05, 0, 0])
    assert nearest_sample(hand, R.identity(), pos, rot, 4, (2, 4))[0] == 5     # going out
    assert nearest_sample(hand, R.identity(), pos, rot, 36, (2, 4))[0] == 36   # coming back: not the way-out pose 5
    # a bend moves the end fully and the start not at all
    p2, r2 = bend(pos, rot, np.array([0, 0.1, 0]), np.array([0, 0, 0.5]), ramp(pos))
    assert np.allclose(p2[0], pos[0]) and np.allclose(p2[-1], pos[-1] + [0, 0.1, 0]) and abs(r2[-1].magnitude() - 0.5) < 1e-9
    print("teacher path search: ok")
