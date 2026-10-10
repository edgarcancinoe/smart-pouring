"""LIBERO problem for the watering task: a mug full of droplet spheres and plant pots.

BDDL files name this problem as `Watering_Tabletop_Manipulation` and state the goal as
`(Watered <pot>)`. Success = at least SUCCESS_FRAC of the droplets resting in that pot.
"""
import xml.etree.ElementTree as ET

import numpy as np

from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
from libero.libero.envs.problems import libero_tabletop_manipulation  # noqa: F401  (fills TASK_MAPPING)
from libero.libero.envs.regions import REGION_SAMPLERS

from .assets import ASSET_DIR, DROPLET, MUG, POTS_DIMS

SUCCESS_FRAC = 0.5
ROBOT_DROP = 0.15  # m below LIBERO's default mounting height
# Ghosts: see-through mocap copies of the mug and gripper at an intended pose, in geom group 4
# (hidden unless a render turns it on).
GHOST_GROUP = 4
# No-slip solver passes: without them a held mug creeps through the fingers (soft contacts).
NOSLIP_ITERATIONS = 10

# "shoulderview": the phone's spot when recording, for comparison videos (the policy uses agentview +
# wrist). World frame, metres; table top z = 0.90.
SHOULDER_CAM = dict(pos=(-0.70, 0.55, 1.75), lookat=(-0.05, 0.0, 0.92), fovy=50)

# register_problem returns None, so the module-level name is None; take the class from the registry.
Libero_Tabletop_Manipulation = TASK_MAPPING["libero_tabletop_manipulation"]


def _lookat_quat(pos, target):
    """(w, x, y, z) for a MuJoCo camera at pos looking at target (cameras look down their -z)."""
    fwd = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = np.stack([right, up, -fwd], axis=1)
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)]


class Watering_Tabletop_Manipulation(Libero_Tabletop_Manipulation):
    mug_name = "mug_1"

    @property
    def target_pot(self):
        (pred, pot), = self.parsed_problem["goal_state"]
        assert pred.lower() == "watered", f"expected (Watered <pot>), got {pred}"
        return pot

    @property
    def pot_names(self):
        return [n for n in self.objects_dict if n.startswith("plant_pot")]

    def _setup_camera(self, mujoco_arena):
        super()._setup_camera(mujoco_arena)
        c = SHOULDER_CAM
        mujoco_arena.set_camera(camera_name="shoulderview", pos=list(c["pos"]),
                                quat=_lookat_quat(c["pos"], c["lookat"]), camera_attribs={"fovy": str(c["fovy"])})

    def _load_model(self):
        super()._load_model()
        # mount the robot lower, so a tilted grab of the low mug leaves joint room for the pour
        rm = self.robots[0].robot_model
        rm.set_base_xpos(np.array(rm.base_xpos_offset["table"](self.table_full_size[0])) - [0, 0, ROBOT_DROP])
        r = DROPLET["radius"]
        for i in range(DROPLET["count"]):
            body = ET.SubElement(self.model.worldbody, "body", name=f"droplet_{i}", pos=f"0 0 {2 + 0.02 * i}")
            ET.SubElement(body, "freejoint", name=f"droplet_{i}_joint")
            # group 0: collisions (hidden), group 1: visual. Stiff contacts so piles don't sink into walls;
            # priority 1 so droplet contacts use the droplet's friction, not the mug's.
            ET.SubElement(body, "geom", type="sphere", size=f"{r}", density="1000", rgba="0 0 0 0",
                          friction="0.3 0.005 0.0001", condim="3", group="0", contype="2", conaffinity="3",
                          solref="0.004 1", solimp="0.95 0.99 0.001", solmix="10", priority="1")
            ET.SubElement(body, "geom", type="sphere", size=f"{r}", rgba=DROPLET["rgba"],
                          contype="0", conaffinity="0", group="1")

        self._add_ghosts()

    def _add_ghosts(self):
        wb = self.model.worldbody
        mug = ET.SubElement(wb, "body", name="ghost_mug", mocap="true", pos="0 0 -1")
        src = ET.parse(ASSET_DIR / "watering_mug" / "watering_mug.xml")
        for g in src.iter("geom"):
            if g.get("group") == "1":  # the mug's visual geoms, recoloured
                attrs = {k: v for k, v in g.attrib.items() if k not in ("material", "rgba", "group")}
                ET.SubElement(mug, "geom", **attrs, group=str(GHOST_GROUP), rgba="0.2 0.95 0.3 0.35")
        hand = ET.SubElement(wb, "body", name="ghost_grip", mocap="true", pos="0 0 -1")
        # gripper frame: z = pointing direction, y = finger axis; origin at the fingertip centre
        for pos, size in [("0 0 -0.075", "0.03 0.09 0.025"),       # palm
                          ("0 0.042 -0.03", "0.01 0.008 0.03"),     # fingers (open-ish)
                          ("0 -0.042 -0.03", "0.01 0.008 0.03"),
                          ("0 0 0.04", "0.003 0.003 0.04")]:        # pointer
            ET.SubElement(hand, "geom", type="box", pos=pos, size=size, group=str(GHOST_GROUP),
                          rgba="1.0 0.55 0.1 0.4", contype="0", conaffinity="0", mass="0")

    def set_ghost(self, name, pos, quat_wxyz):
        """Place 'ghost_mug' or 'ghost_grip' (world pose). Visible only where group 4 is rendered."""
        mid = self.sim.model._model.body_mocapid[self.sim.model._model.body(name).id]
        self.sim.data._data.mocap_pos[mid] = pos
        self.sim.data._data.mocap_quat[mid] = quat_wxyz

    def _setup_references(self):
        super()._setup_references()
        self.sim.model._model.opt.noslip_iterations = NOSLIP_ITERATIONS
        self.droplet_body_ids = [self.sim.model.body_name2id(f"droplet_{i}") for i in range(DROPLET["count"])]

    def _reset_internal(self):
        super()._reset_internal()
        self._fill_mug()

    def set_layout(self, mug_xy, pot_xys):
        """Put the mug and pots exactly at these table-frame xy spots (metres) and refill the mug.

        Resets randomize positions by up to +-JITTER; call this after reset for a fixed scene.
        """
        for name, xy in [(self.mug_name, mug_xy)] + [(f"plant_pot_{s}", xy) for s, xy in pot_xys.items()]:
            joint = self.get_object(name).joints[-1]
            qpos = self.sim.data.get_joint_qpos(joint).copy()
            qpos[:2] = np.add(xy, self.workspace_offset[:2])
            self.sim.data.set_joint_qpos(joint, qpos)
            self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self._fill_mug()

    def _fill_mug(self):
        """Stack the droplets inside the mug in layers of seven (centre + hexagon)."""
        mug_pos = self.sim.data.get_joint_qpos(self.get_object(self.mug_name).joints[-1])[:3]
        r = DROPLET["radius"]
        ring = 2.05 * r  # hexagon spacing so no two droplets start overlapping
        assert ring + r < MUG["r_in"], "droplets too big for the bottom of the mug"
        layer = [(0.0, 0.0)] + [(ring * np.cos(a), ring * np.sin(a))
                                 for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        for i in range(DROPLET["count"]):
            dx, dy = layer[i % 7]
            z = MUG["bottom"] + r + 0.001 + (i // 7) * 2.05 * r
            self.sim.data.set_joint_qpos(f"droplet_{i}_joint",
                                         np.r_[mug_pos + [dx + 0.001 * (i // 7), dy, z], 1, 0, 0, 0])
            self.sim.data.set_joint_qvel(f"droplet_{i}_joint", np.zeros(6))
        self.sim.forward()

    def droplet_positions(self):
        return np.array([self.sim.data.body_xpos[b] for b in self.droplet_body_ids])

    def watering_stats(self):
        """Droplet counts per pot, in the mug, and spilled."""
        drops = self.droplet_positions()
        counts, used = {}, np.zeros(len(drops), bool)
        for name in self.pot_names + [self.mug_name]:
            body_id = self.obj_body_id[name]
            pos = self.sim.data.body_xpos[body_id]
            up = self.sim.data.body_xmat[body_id].reshape(3, 3)[:, 2]
            rel = drops - pos
            h = rel @ up
            radial = np.linalg.norm(rel - np.outer(h, up), axis=1)
            dims = POTS_DIMS[name.split("_")[-1]] if name.startswith("plant_pot") else MUG
            floor = dims["soil"] if name.startswith("plant_pot") else dims["bottom"]
            inside = (radial < dims.get("r_in_top", dims["r_in"])) & (h > floor - 0.005) & (h < dims["height"] + 0.01)
            counts[name] = int(inside.sum())
            used |= inside
        counts["spilled"] = int((~used).sum())
        counts["target"] = counts[self.target_pot]
        return counts

    def _check_success(self):
        return self.watering_stats()["target"] >= SUCCESS_FRAC * DROPLET["count"]


register_problem(Watering_Tabletop_Manipulation)
REGION_SAMPLERS["watering_tabletop_manipulation"] = REGION_SAMPLERS["libero_tabletop_manipulation"]
