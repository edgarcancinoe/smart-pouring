"""LeRobot's training script (state_dropout.py) with a memory of the grasp in the robot state: [grasped, x, y] of the
gripper when it last started closing, in free slots of SmolVLA's padded state. Called by train.py. See README.md.

Usage: python grasp_position.py <share> <lerobot_train args ...>   (./run train <model with this script>)
"""
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts import lerobot_train

import state_dropout

SLOTS = slice(8, 11)  # padded state slots filled: grasped (0/1), x, y (the real state is the first 8)

TABLE = None              # training: (frames, 3) grasp slots per global frame index (the batch's "index")
CURRENT = torch.zeros(3)  # eval: the running trial's grasp slots (eval_grasp_position.py updates it)


### Pre-TrainingL: Obtain grasp slots from dataset ##########################
def grasp_positions(closing, xy):
    """Grasp slots per step of one episode: [1, x, y] where the gripper was when it last started closing, zeros before.

    closing -> (T,) bool, the step's gripper action closes (action[6] > 0)
    xy      -> (T, 2) gripper position the step's action was chosen from (normalized)
    returns (T, 3) float32; a step carries the grasps of earlier steps only (its own close is decided without it)
    """
    out, last = np.zeros((len(xy), 3), np.float32), None
    for k in range(len(xy)):
        if last is not None:
            out[k] = [1, *xy[last]]
        if closing[k] and (k == 0 or not closing[k - 1]):
            last = k
    return out


def normalize(xy, stats):
    """xy (..., 2) m -> as the policy's normalizer scales the state's x, y (mean/std of the dataset)."""
    return (xy - stats["mean"][:2]) / stats["std"][:2]


def table(repo):
    """Grasp slots of every frame of a LeRobot dataset, indexed by its global frame index to use during training
    """
    ds = LeRobotDataset(repo, download_videos=False)
    t = pq.read_table(ds.root / "data", columns=["index", "episode_index", "observation.state", "action"]).to_pandas()
    t = t.sort_values("index")
    out = torch.zeros(t["index"].max() + 1, 3)
    for _, ep in t.groupby("episode_index"):
        xy = normalize(np.stack(ep["observation.state"].to_numpy())[:, :2], ds.meta.stats["observation.state"])
        closing = np.stack(ep["action"].to_numpy())[:, 6] > 0
        out[ep["index"].to_numpy()] = torch.from_numpy(grasp_positions(closing, xy))
    print(f"grasp slots: {len(t)} frames, {(out[:, 0] > 0).float().mean():.0%} after a grasp", flush=True)
    return out


def prepare_state(self, batch):
    """state_dropout's state, plus the grasp slots: from TABLE while training, else CURRENT. Written after the
    dropout so the policy can't learn to ignore them."""
    state = state_dropout.prepare_state(self, batch)
    grasp = TABLE[batch["index"].long().cpu()] if self.training else CURRENT.expand(len(state), 3)
    state[:, SLOTS] = grasp.to(state)
    return state

###############################################################################

def main(prepare=prepare_state):    
    global TABLE
    
    state_dropout.P = float(sys.argv.pop(1))
    repo = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--dataset.repo_id="))
    TABLE = table(repo)

    # Override the policy's prepare_state to include grasp slots in the state
    SmolVLAPolicy.prepare_state = prepare

    lerobot_train.main()


if __name__ == "__main__":
    main()
