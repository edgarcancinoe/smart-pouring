"""memory.py (../VLM-cache) with the memory isolated inside the VLM: the observation (cameras, command, state) doesn't
read the start frame, so it is processed exactly as without memory; only the action expert reads the memory.
Called by train.py. See README.md.

Usage: python isolated.py <share> <lerobot_train args ...>   (./run train <model with this script>)
"""
import torch
from lerobot.policies.smolvla import modeling_smolvla
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

import memory

_make_att_2d_masks, _embed_prefix = modeling_smolvla.make_att_2d_masks, VLAFlowMatching.embed_prefix
_n = [0]  # memory tokens in front of the prefix last embedded


def embed_prefix(self, *args, **kwargs):
    """memory.embed_prefix, noting how many memory tokens it put in front (the tokens before the first flag)."""
    embs, pad, att = _embed_prefix(self, *args, **kwargs)
    _n[0] = int(att[0].float().argmax()) if self.memory is not None else 0
    return embs, pad, att


def make_att_2d_masks(pad, att):
    """SmolVLA's mask without the present's and the state's reads of the memory (blocks 1-2 -> block 0);
    the actions (blocks 3+) keep them. Training and uncached eval; the cached eval uses memory.ISOLATE."""
    masks = _make_att_2d_masks(pad, att)
    n = _n[0]
    if n and not att[:, :n].any():  # a sequence starting with the memory, not denoise_step's actions alone
        block = torch.cumsum(att, 1)
        masks[:, :, :n] &= ~((block >= 1) & (block <= 2))[:, :, None]
    return masks


# at import, so eval's spawned workers get them too (after memory.py's own patches)
memory.ISOLATE = True
VLAFlowMatching.embed_prefix = embed_prefix
modeling_smolvla.make_att_2d_masks = make_att_2d_masks

if __name__ == "__main__":
    memory.main()
