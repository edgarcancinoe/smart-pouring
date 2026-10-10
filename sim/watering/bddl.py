"""Write one BDDL file per command ("water the left/middle/right plant").

The layout is measured on the real setup (REAL_* below, cm). Real frame: +x to the
demonstrator's right, +y away from them. Sim table frame: origin at the table centre,
+x away from the robot, +y the robot's left. The demonstrator stands where the robot is.
Regions are (x_min, y_min, x_max, y_max); each start spot gets +-JITTER.
"""
from pathlib import Path

from .assets import MUG, POTS_DIMS, outer_radius

BDDL_DIR = Path(__file__).parent / "bddl"
JITTER = 0.005  # metres: how far the sim may shift each object from its measured spot

# Measured on the real floor setup, cm.
REAL_POTS = {"left": (-33.0, 46.1), "middle": (-7.5, 46.1), "right": (13.5, 46.1)}
REAL_MUG_STARTS = [(-14.0, 20.0), (12.0, 20.0), (-3.5, 24.0), (18.2, 24.0)]

# The middle pot lands at SIM_ANCHOR (table frame, m): 10 cm past the table centre, so every mug
# spot is comfortably reachable.
SIM_ANCHOR = (0.10, 0.0)


def real_to_sim(xy_cm):
    ax, ay = REAL_POTS["middle"]
    x, y = xy_cm
    return (SIM_ANCHOR[0] + (y - ay) / 100, SIM_ANCHOR[1] - (x - ax) / 100)


POTS = {side: real_to_sim(xy) for side, xy in REAL_POTS.items()}
MUG_STARTS = [real_to_sim(xy) for xy in REAL_MUG_STARTS]

TEMPLATE = """(define (problem Watering_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Water the {side} plant)
    (:regions
{regions}
    )

  (:fixtures
    main_table - table
  )

  (:objects
    mug_1 - watering_mug
{pot_objects}
  )

  (:obj_of_interest
    mug_1
    plant_pot_{side}
  )

  (:init
    (On mug_1 main_table_mug_region)
{pot_init}
  )

  (:goal
    (And (Watered plant_pot_{side}))
  )

)
"""


def _region(name, spots, radius):
    """One region; with several spots LIBERO picks one at random on each reset.

    LIBERO keeps the whole object inside the region by shrinking it by the object's radius
    before sampling, so pad by that radius to leave exactly +-JITTER for the centre.
    """
    j = radius + JITTER
    rects = "\n".join(f"              ({x - j:.3f} {y - j:.3f} {x + j:.3f} {y + j:.3f})" for x, y in spots)
    return f"""      ({name}
          (:target main_table)
          (:ranges (
{rects}
            )
          )
      )"""


def write_bddl():
    BDDL_DIR.mkdir(exist_ok=True)
    # pad by each object's horizontal_radius_site (see _region)
    regions = [_region("mug_region", MUG_STARTS, outer_radius(MUG))]
    regions += [_region(f"pot_{s}_region", [xy], outer_radius(POTS_DIMS[s])) for s, xy in POTS.items()]
    paths = {}
    for side in POTS:
        text = TEMPLATE.format(
            side=side,
            regions="\n".join(regions),
            # each pot has its own size, so its own type (the parser wants one line per type)
            pot_objects="\n".join(f"    plant_pot_{s} - plant_pot_{s}" for s in POTS),
            pot_init="\n".join(f"    (On plant_pot_{s} main_table_pot_{s}_region)" for s in POTS),
        )
        paths[side] = BDDL_DIR / f"water_the_{side}_plant.bddl"
        paths[side].write_text(text)
    return paths

