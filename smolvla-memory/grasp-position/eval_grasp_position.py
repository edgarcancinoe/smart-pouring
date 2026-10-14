"""./run eval for models trained with grasp_position.py: the same trials, with the grasp slots filled as in training
(the gripper's x, y when its fingers last stopped on something while closing: grasp_position.held). Results in smolvla-finetune/results/<model>/.

Usage: ./run grasp-pos-eval [libero-4s-dart-xl-sd-gp ...] [--jobs 8]
"""
import torch
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import evaluate as ev
import grasp_position as gp
import runs

SCRIPTS = ["smolvla-memory/grasp-position/grasp_position.py"]  # training scripts whose models fill the grasp slots
BLANK = False  # ablation: keep the grasp slots at zero all trial (does the policy use them?)

_reset, _scene, _load_model = ev.reset, ev.scene, ev.load_model
_last = {"held": False, "closing": False, "width": 0.0, "start": 0.0, "obs": None, "stats": None}  # this process's running trial


def load_model(name):
    """evaluate.load_model, after checking the model was trained with the grasp slots; keeps its dataset's state stats."""
    model = runs.MODELS[name.partition("@")[0]]
    if model.get("script") not in SCRIPTS:
        raise SystemExit(f"{name} isn't trained with the grasp slots ({SCRIPTS}): use ./run eval")
    _last["stats"] = LeRobotDatasetMetadata(model.get("data", runs.DATASET)).stats["observation.state"]
    return _load_model(name)


def step(env, action):
    """env.step, noting the gripper's x, y when its fingers stop on something while closing (the same rule as in training)."""
    q = _last["obs"]["robot0_gripper_qpos"]
    width, closing = float(q[0] - q[1]), action[6] > 0
    if closing and not _last["closing"]:
        _last["start"] = width
    now = gp.held(closing, width, _last["width"], _last["start"])
    if now and not _last["held"] and not BLANK:
        xy = gp.normalize(_last["obs"]["robot0_eef_pos"][:2], _last["stats"])
        gp.CURRENT = torch.tensor([1, *xy], dtype=torch.float32)
    _last.update(held=now, closing=closing, width=width)
    out = env._unwrapped_step(action)
    _last["obs"] = out[0]
    return out


def scene(plant):
    """evaluate.scene, with its env.step wrapped (once) by step()."""
    env = _scene(plant)
    if not hasattr(env, "_unwrapped_step"):
        env._unwrapped_step = env.step
        env.step = lambda action: step(env, action)
    return env


def reset(env, spot, rng, jitter=ev.MUG_JITTER):
    """evaluate.reset, then empty grasp slots: a new trial."""
    obs = _reset(env, spot, rng, jitter)
    gp.CURRENT = torch.zeros(3)
    q = obs["robot0_gripper_qpos"]
    _last.update(held=False, closing=False, width=float(q[0] - q[1]), start=float(q[0] - q[1]), obs=obs)
    return obs


# at import, so the spawned eval workers get them too
SmolVLAPolicy.prepare_state = gp.prepare_state
ev.load_model, ev.scene, ev.reset = load_model, scene, reset
ev.MODELS = [m for m, c in runs.MODELS.items() if c.get("script") in SCRIPTS]  # default: every grasp-slot model

if __name__ == "__main__":
    ev.main()
