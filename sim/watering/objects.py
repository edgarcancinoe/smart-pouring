"""Register the generated mug and plant pot as LIBERO objects (usable by name in BDDL)."""
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object

from .assets import write_assets

ASSET_DIR = write_assets()


class _GeneratedObject(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            str(ASSET_DIR / obj_name / f"{obj_name}.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = obj_name
        self.rotation = (0, 0)  # models are z-up; fixed yaw (see the handle note in assets.py)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


@register_object
class WateringMug(_GeneratedObject):
    def __init__(self, name="watering_mug", obj_name="watering_mug"):
        super().__init__(name, obj_name)


# One class per pot because each real pot has its own size (see POTS_DIMS in assets.py).
@register_object
class PlantPotLeft(_GeneratedObject):
    def __init__(self, name="plant_pot_left", obj_name="plant_pot_left"):
        super().__init__(name, obj_name)


@register_object
class PlantPotMiddle(_GeneratedObject):
    def __init__(self, name="plant_pot_middle", obj_name="plant_pot_middle"):
        super().__init__(name, obj_name)


@register_object
class PlantPotRight(_GeneratedObject):
    def __init__(self, name="plant_pot_right", obj_name="plant_pot_right"):
        super().__init__(name, obj_name)
