"""LeRobot's training script, with the robot state zeroed in a share of the training samples, so the policy
can't predict the actions from the arm's pose alone and has to look at the cameras. Called by train.py.

Usage: python state_dropout.py <share 0-1> <lerobot_train args ...>
"""
import sys

import av
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_train
from wandb.sdk.wandb_run import Run

if not hasattr(Run, "get_url"):
    Run.get_url = lambda self: self.url

_av_open = av.open


def av_open(*args, **kwargs):
    """av.open, decoding video on one thread: LeRobot opens the file for every frame, and FFmpeg sizes its thread
    pool by the cores it sees (256 on a pod allowed 27) -> a pause every num_workers steps. At import: the
    spawned data-loader workers re-import this file."""
    container = _av_open(*args, **kwargs)
    for s in getattr(container.streams, "video", ()):
        s.codec_context.thread_count = 1
    return container


av.open = av_open

P = 0.0  # set in __main__: the spawned data-loader workers re-run this file with the share already popped from argv
_prepare_state = SmolVLAPolicy.prepare_state


def prepare_state(self, batch):
    """SmolVLA's state input; while training, zeroed (the normalized mean pose) for a share P of the batch."""
    state = _prepare_state(self, batch)
    if self.training and P:
        state = state * (torch.rand(len(state), 1, device=state.device) >= P)
    return state


if __name__ == "__main__":
    P = float(sys.argv.pop(1))
    SmolVLAPolicy.prepare_state = prepare_state  
    lerobot_train.main()
