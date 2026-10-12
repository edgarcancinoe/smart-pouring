"""LeRobot's training script, with the robot state zeroed in a share of the training samples, so the policy
can't predict the actions from the arm's pose alone and has to look at the cameras. Called by train.py.

Usage: python state_dropout.py <share 0-1> <lerobot_train args ...>
"""
import sys

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_train
from wandb.sdk.wandb_run import Run

if not hasattr(Run, "get_url"):
    Run.get_url = lambda self: self.url

P = float(sys.argv.pop(1))
_prepare_state = SmolVLAPolicy.prepare_state


def prepare_state(self, batch):
    """SmolVLA's state input; while training, zeroed (the normalized mean pose) for a share P of the batch."""
    state = _prepare_state(self, batch)
    if self.training and P:
        state = state * (torch.rand(len(state), 1, device=state.device) >= P)
    return state


SmolVLAPolicy.prepare_state = prepare_state  # evaluation (policy.eval()) sees the full state

if __name__ == "__main__":
    lerobot_train.main()
