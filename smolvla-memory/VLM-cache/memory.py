"""LeRobot's training script (state_dropout.py) with a memory of the episode's start: the VLM tokens of its first
agentview frame go in front of every prefix, in their own attention block (the present and the actions see them,
they never see the present). So their keys and values are the same in every chunk: eval computes them once per
trial and reuses them (CACHE), like the earlier turns of a chat. Called by train.py. See README.md.

Usage: python memory.py <share> <lerobot_train args ...>   (./run train <model with this script>)
"""
import copy
import sys

import pyarrow.parquet as pq
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.scripts import lerobot_train

import state_dropout

CAMERA = "observation.images.image"  # the camera remembered: the agentview shows where the mug stands
CACHE = True   # eval: the memory's keys and values computed once per trial (False: again with every chunk, as in training)
BLANK = False  # eval ablation: an empty memory (pad mask 0), the same as no memory
ISOLATE = False  # the current obs doesn't read the memory, only the expert does (set by VLM-cache-isolated/isolated.py)

TABLE = None  # training: (episodes, 3, H, W) uint8, each episode's first CAMERA frame, by episode_index

_reset, _forward, _get_action_chunk = SmolVLAPolicy.reset, SmolVLAPolicy.forward, SmolVLAPolicy._get_action_chunk
_embed_prefix, _sample_actions = VLAFlowMatching.embed_prefix, VLAFlowMatching.sample_actions


def memory_table(repo):
    """First CAMERA frame of every episode of a LeRobot dataset.

    repo -> Hub dataset id (the one training reads)
    returns (episodes, 3, H, W) uint8 tensor, indexed by episode_index
    """
    ds = LeRobotDataset(repo, video_backend="pyav")
    t = pq.read_table(ds.root / "data", columns=["index", "episode_index", "frame_index"]).to_pandas()
    first = t[t["frame_index"] == 0]
    out = None
    for ep, i in zip(first["episode_index"], first["index"]):
        img = (ds[int(i)][CAMERA] * 255).round().to(torch.uint8)
        if out is None:
            out = torch.zeros(first["episode_index"].max() + 1, *img.shape, dtype=torch.uint8)
        out[ep] = img
    print(f"memory: first {CAMERA} frame of {len(first)} episodes", flush=True)
    return out


def remember(policy, img):
    """Make img (B, 3, H, W) in [0, 1] the memory, processed like the policy's cameras."""
    (img,), (mask,) = policy.prepare_images({CAMERA: img})
    policy.model.memory, policy.model.memory_kv = (img, mask & (not BLANK)), None


def reset(self):
    """SmolVLAPolicy.reset, and forget: a new trial."""
    _reset(self)
    self.model.memory = self.model.memory_kv = None


def forward(self, batch, *args, **kwargs):
    """SmolVLAPolicy.forward (training), with each sample's episode start as the memory."""
    remember(self, TABLE[batch["episode_index"].long().cpu()].to(batch[CAMERA].device).float() / 255)
    return _forward(self, batch, *args, **kwargs)


def get_action_chunk(self, batch, *args, **kwargs):
    """SmolVLAPolicy._get_action_chunk (eval); the trial's first chunk remembers its own frame: the start."""
    if self.model.memory is None:
        remember(self, batch[CAMERA])
    return _get_action_chunk(self, batch, *args, **kwargs)


def memory_tokens(self):
    """The memory's embeddings, pad and attention masks: embed_prefix with the image alone (no language, and
    the state token dropped)."""
    img, mask = self.memory
    none = torch.zeros(len(img), 0, dtype=torch.long, device=img.device)
    embs, pad, att = _embed_prefix(self, [img], [mask], none, none.bool(),
                                   torch.zeros(len(img), self.config.max_state_dim, device=img.device))
    return embs[:, :-1], pad[:, :-1], att[:, :-1]


def embed_prefix(self, *args, **kwargs):
    """VLAFlowMatching.embed_prefix with the memory in front, as block 0: the present starts block 1."""
    embs, pad, att = _embed_prefix(self, *args, **kwargs)
    if self.memory is None:
        return embs, pad, att
    m_embs, m_pad, m_att = memory_tokens(self)
    att = att.clone()
    att[:, 0] = True
    return torch.cat([m_embs, embs], 1), torch.cat([m_pad, pad], 1), torch.cat([m_att, att], 1)


def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
    """VLAFlowMatching.sample_actions with the memory's keys and values cached: computed on the trial's first chunk,
    then each chunk runs only the present on top of a copy of them. Same actions as without the cache."""
    if not CACHE or self.memory is None:
        return _sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise, **kwargs)
    vlm = self.vlm_with_expert
    if self.memory_kv is None:
        m_embs, m_pad, m_att = memory_tokens(self)
        _, kv = vlm.forward(attention_mask=make_att_2d_masks(m_pad, m_att), position_ids=torch.cumsum(m_pad, 1) - 1,
                            past_key_values=None, inputs_embeds=[m_embs, None], use_cache=True)
        self.memory_kv = kv, m_pad
    kv, m_pad = self.memory_kv

    embs, pad, att = _embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=state)
    mask = torch.cat([m_pad[:, None, :] & pad[:, :, None] & (not ISOLATE), make_att_2d_masks(pad, att)], 2)
    cache = copy.deepcopy(kv)  # the present is appended to it: keep the stored memory as it is
    mode, vlm.attention_mode = vlm.attention_mode, "self_attn"  # else the cross-attention layers skip the present
    _, cache = vlm.forward(attention_mask=mask, position_ids=m_pad.sum(1, keepdim=True) + torch.cumsum(pad, 1) - 1,
                           past_key_values=cache, inputs_embeds=[embs, None], use_cache=True)
    vlm.attention_mode = mode

    if noise is None:
        noise = self.sample_noise((len(state), self.config.chunk_size, self.config.max_action_dim), state.device)
    prefix_pad = torch.cat([m_pad, pad], 1)
    return euler_integrate(lambda x, t: self.denoise_step(x_t=x, prefix_pad_masks=prefix_pad, past_key_values=cache,
                                                          timestep=t), noise, self.config.num_steps)


SmolVLAPolicy.reset, SmolVLAPolicy.forward, SmolVLAPolicy._get_action_chunk = reset, forward, get_action_chunk
VLAFlowMatching.embed_prefix, VLAFlowMatching.sample_actions = embed_prefix, sample_actions

def main():
    """Train (isolated.py calls it too)."""
    global TABLE
    state_dropout.P = float(sys.argv.pop(1))
    TABLE = memory_table(next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--dataset.repo_id=")))
    SmolVLAPolicy.prepare_state = state_dropout.prepare_state
    lerobot_train.main()


if __name__ == "__main__":
    main()
