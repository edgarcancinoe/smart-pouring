"""Plant-watering task for LIBERO. Importing this registers the objects and the problem."""
import os

import libero.libero as _libero

# Use the cached LIBERO assets directly (otherwise hf-libero prints a misleading download message).
_CACHED_ASSETS = os.path.expanduser("~/.cache/libero/assets")
if os.path.isdir(_CACHED_ASSETS):
    _libero._assets_path_cache = _CACHED_ASSETS

from . import objects, problem  # noqa: E402,F401
from .bddl import POTS, write_bddl


def make_env(side="middle", offscreen=True, **kwargs):
    """LIBERO env for "water the <side> plant".

    offscreen=False builds it without robosuite's renderer (no camera observations). Use that
    under the interactive viewer: on macOS the renderer opens a GLFW window, which crashes
    off the main thread, and mjpython runs the script off the main thread.
    """
    from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv

    bddl = str(write_bddl()[side])
    if not offscreen:
        kwargs = {"has_renderer": False, "has_offscreen_renderer": False, "use_camera_obs": False, **kwargs}
        return ControlEnv(bddl_file_name=bddl, **kwargs)
    kwargs = {"camera_heights": 256, "camera_widths": 256, **kwargs}
    return OffScreenRenderEnv(bddl_file_name=bddl, **kwargs)
