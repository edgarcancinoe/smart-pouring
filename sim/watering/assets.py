"""Procedural MJCF for the mug and the plant pot.

Both are open shells built from a ring of thin boxes plus a solid bottom, so they are hollow
for the droplet spheres. The mug can taper (wider at the top). Real sizes come from dataset/setup.json.
"""
import json
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).parent / "assets"
# The measured real setup (cm, floor frame), shared with the dataset pipeline.
SETUP = json.loads((Path(__file__).resolve().parents[2] / "dataset" / "setup.json").read_text())

# Metres. The real mug scaled to fit the Panda gripper's ~8 cm opening.
# r_in / r_in_top: inner radii at the bottom / top.
MUG_SCALE = 0.8
_WALL = 0.004
_mug = {k: v / 100 * MUG_SCALE for k, v in SETUP["mug"].items()}  # m
MUG = dict(r_in=_mug["diameter_bottom"] / 2 - _WALL, r_in_top=_mug["diameter_top"] / 2 - _WALL, wall=_WALL,
           height=_mug["height"], bottom=0.006, n_sides=24,
           rgba="0.95 0.72 0.18 1")
_POT_STYLE = dict(wall=0.006, n_sides=24, rgba="0.72 0.38 0.24 1", soil_rgba="0.30 0.20 0.13 1",
                  leaf_rgba="0.25 0.55 0.22 1")


def _pot(diameter, height):
    """Pot from measured outer diameter and height (m); soil sits 3 cm below the rim."""
    return dict(_POT_STYLE, r_in=diameter / 2 - _POT_STYLE["wall"], height=height, soil=height - 0.03)


POTS_DIMS = {s: _pot(SETUP["pots"]["diameter"][s] / 100, SETUP["pots"]["height"][s] / 100)
             for s in ("left", "middle", "right")}
DROPLET = dict(radius=0.005, count=30, rgba="0.30 0.60 0.95 1")  # 30 fit below the rim of the scaled mug

COL = 'group="0" rgba="0 0 0 0" friction="1.0 0.005 0.0001"'
# The mug's outer surface, where the fingers hold it: grippy, with torsional friction (condim 4) so the
# mug doesn't pivot between the two fingertips.
MUG_COL = 'group="0" rgba="0 0 0 0" friction="1.5 0.03 0.0001" condim="4"'
VIS = 'group="1" contype="0" conaffinity="0"'
# Collision bits: ordinary geoms 1/1, droplets 2/3 (hit ordinary geoms and each other). DROPLET_ONLY
# geoms (2/2) touch only droplets: the mug's invisible thick shell, so droplets can't pass the thin wall.
DROPLET_ONLY = 'group="0" rgba="0 0 0 0" contype="2" conaffinity="2" friction="0.3 0.005 0.0001"'
SHELL = 0.015  # m, droplet-only thickness behind the mug's inner surface


def _ring(r_in, wall, z0, height, n, rgba, density, r_in_top=None, kinds=None):
    """Boxes approximating a (possibly tapered) wall: one collision and one visual geom per facet."""
    r_in_top = r_in if r_in_top is None else r_in_top
    r_mid = (r_in + r_in_top) / 2 + wall / 2
    tilt = np.arctan2(r_in_top - r_in, height)  # lean outward for a taper
    half_len = np.hypot(height, r_in_top - r_in) / 2
    half_tan = (r_in_top + wall / 2) * np.tan(np.pi / n) * 1.08  # sized for the wide end; overlap closes seams
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        pos = f"{r_mid * np.cos(a):.5f} {r_mid * np.sin(a):.5f} {z0 + height / 2:.5f}"
        # yaw to face outward, then pitch about the facet's tangent axis by the taper angle
        q_yaw = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])
        q_tilt = np.array([np.cos(tilt / 2), 0, np.sin(tilt / 2), 0])
        q = _qmul(q_yaw, q_tilt)
        quat = " ".join(f"{v:.5f}" for v in q)
        size = f"{wall / 2:.5f} {half_tan:.5f} {half_len:.5f}"
        for kind in kinds or (f'{COL} density="{density}"', f'{VIS} rgba="{rgba}"'):
            out.append(f'<geom type="box" pos="{pos}" quat="{quat}" size="{size}" {kind}/>')
    return out


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def outer_radius(dims):
    """Widest outer radius (the top for a tapered mug); what LIBERO uses for placement."""
    return dims.get("r_in_top", dims["r_in"]) + dims["wall"]


def _cylinder(r, z0, height, rgba, density, collide=True):
    pos, size = f"0 0 {z0 + height / 2:.5f}", f"{r:.5f} {height / 2:.5f}"
    geoms = [f'<geom type="cylinder" pos="{pos}" size="{size}" {VIS} rgba="{rgba}"/>']
    if collide:
        geoms.append(f'<geom type="cylinder" pos="{pos}" size="{size}" {COL} density="{density}"/>')
    return geoms


def _wrap(name, geoms, top, radius):
    body = "\n        ".join(geoms)
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        {body}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {top:.5f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{radius:.5f} 0 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""


def mug_xml(m=MUG):
    r_bot = m["r_in"] + m["wall"]
    r_at = lambda z: r_bot + (m["r_in_top"] - m["r_in"]) * z / m["height"]  # outer radius at height z
    g = _cylinder(r_bot, 0, m["bottom"], m["rgba"], 1000)
    g += _ring(m["r_in"], m["wall"], m["bottom"], m["height"] - m["bottom"], m["n_sides"], m["rgba"], 1000,
               r_in_top=m["r_in_top"])
    # droplet-only shell: same inner surface, SHELL thick, stopping 1 cm below the rim so pours
    # still roll over the real thin lip; plus a thick droplet-only floor under the bottom
    shell_h = m["height"] - m["bottom"] - 0.01
    shell_top = m["r_in"] + (m["r_in_top"] - m["r_in"]) * shell_h / (m["height"] - m["bottom"])
    g += _ring(m["r_in"], SHELL, m["bottom"], shell_h, m["n_sides"], m["rgba"], 1000,
               r_in_top=shell_top, kinds=[f'{DROPLET_ONLY} density="0"'])
    g.append(f'<geom type="cylinder" pos="0 0 {m["bottom"] - SHELL / 2:.5f}" size="{m["r_in"] + SHELL:.5f} {SHELL / 2:.5f}" '
             f'{DROPLET_ONLY} density="0"/>')
    # Handle on -x in the model; LIBERO's placement yaws objects 180 deg, so it faces away from the robot.
    z_lo, z_hi = m["height"] * 0.30, m["height"] * 0.80
    hx = -(r_at((z_lo + z_hi) / 2) + 0.014)
    for pos, size in [(f"{hx:.5f} 0 {(z_lo + z_hi) / 2:.5f}", f"0.004 0.006 {(z_hi - z_lo) / 2:.5f}"),
                      (f"{-(r_at(z_hi) + 0.006):.5f} 0 {z_hi:.5f}", "0.009 0.006 0.004"),
                      (f"{-(r_at(z_lo) + 0.008):.5f} 0 {z_lo:.5f}", "0.010 0.006 0.004")]:
        g.append(f'<geom type="box" pos="{pos}" size="{size}" {COL} density="1000"/>')
        g.append(f'<geom type="box" pos="{pos}" size="{size}" {VIS} rgba="{m["rgba"]}"/>')
    g = [x.replace(COL, MUG_COL) for x in g]  # the gripper holds these surfaces
    return _wrap("watering_mug", g, m["height"], outer_radius(m))


def pot_xml(p):
    r_out = p["r_in"] + p["wall"]
    # Soil fills the pot up to `soil`; droplets land on it. Heavy so bumps don't move the pot.
    g = _cylinder(r_out, 0, p["soil"], p["soil_rgba"], 3000)
    g += _ring(p["r_in"], p["wall"], 0, p["height"], p["n_sides"], p["rgba"], 3000)
    # A small visual-only sprout in the middle, kept low so the mug can pour over it.
    z = p["soil"]
    g.append(f'<geom type="capsule" fromto="0 0 {z:.4f} 0 0 {z + 0.05:.4f}" size="0.003" {VIS} rgba="{p["leaf_rgba"]}"/>')
    for i in range(5):
        a = 2 * np.pi * i / 5
        c, s = np.cos(a), np.sin(a)
        g.append(f'<geom type="ellipsoid" pos="{0.3 * p['r_in'] * c:.4f} {0.3 * p['r_in'] * s:.4f} {z + 0.035 + 0.006 * (i % 2):.4f}" '
                 f'quat="{np.cos(a / 2):.4f} 0 0 {np.sin(a / 2):.4f}" size="0.018 0.007 0.002" {VIS} rgba="{p["leaf_rgba"]}"/>')
    return _wrap("plant_pot", g, p["height"], r_out)


def write_assets():
    models = [("watering_mug", mug_xml())] + [(f"plant_pot_{s}", pot_xml(p)) for s, p in POTS_DIMS.items()]
    for name, xml in models:
        d = ASSET_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.xml").write_text(xml)
    return ASSET_DIR
