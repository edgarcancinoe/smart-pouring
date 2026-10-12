"""The models compared (<init>-<k>, see README.md) and which phone demos each one trains on."""
from dataset import episodes

DATASET = "edgarcancinoe/water_pour"  # the demos with the mug exactly on its 4 measured spots
SHIFTED = episodes.HUB_DATASET       # the same + each replayed with the mug shifted (robot_replay.py SHIFTS)

# The two pretrained checkpoints every model starts from
LIBERO  = "HuggingFaceVLA/smolvla_libero"   # pretrained on LIBERO: same Panda
BASE    = "lerobot/smolvla_base"             # pretrained on SO arms

# Every model compared.
#   start -> checkpoint it starts from
#   k     -> phone demos per plant x start spot it is fine-tuned on (0 = not fine-tuned)
#   repo  -> Hugging Face repo holding it (./run train pushes there, ./run eval loads from there)
#   data  -> dataset it trains on (default DATASET)
#   batch -> samples per step (default train.BATCH)
#   dropout -> share of training samples with the robot state zeroed (default train.STATE_DROPOUT)
MODELS = {
    "libero-0": dict(start=LIBERO,  k=0, repo=LIBERO),  # baseline: smolvla_libero as downloaded
    "libero-1": dict(start=LIBERO,  k=1, repo="edgarcancinoe/smolvla_water_libero-1"),
    "libero-2": dict(start=LIBERO,  k=2, repo="edgarcancinoe/smolvla_water_libero-2"),
    "libero-4": dict(start=LIBERO,  k=4, repo="edgarcancinoe/smolvla_water_libero-4"),
    "base-1":   dict(start=BASE,    k=1, repo="edgarcancinoe/smolvla_water_base-1"),
    "base-2":   dict(start=BASE,    k=2, repo="edgarcancinoe/smolvla_water_base-2"),
    "base-4":   dict(start=BASE,    k=4, repo="edgarcancinoe/smolvla_water_base-4"),
    "libero-4s": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s", data=SHIFTED),
    "base-4s":   dict(start=BASE,   k=4, repo="edgarcancinoe/smolvla_water_base-4s",   data=SHIFTED),
    "libero-4s-b64": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-b64", data=SHIFTED, batch=64),
    "base-4s-b64":   dict(start=BASE,   k=4, repo="edgarcancinoe/smolvla_water_base-4s-b64",   data=SHIFTED, batch=64),
    "libero-4s-sd":  dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-sd", data=SHIFTED, dropout=0.3),
    "base-4s-sd":    dict(start=BASE,   k=4, repo="edgarcancinoe/smolvla_water_base-4s-sd",   data=SHIFTED, dropout=0.3),
}


CAMERA_NAMES = {
    LIBERO: {},
    BASE: {"observation.images.image": "observation.images.camera1",
           "observation.images.image2": "observation.images.camera2"},
}


def repo(name):
    """Where a model loads from: Hub repo and revision. <model>@<step> (e.g. libero-4@10000) is the
    checkpoint saved at that step, kept on the repo's branch step-<step>; a plain name is the final model (main)."""
    base, _, step = name.partition("@")
    return MODELS[base]["repo"], (f"step-{int(step)}" if step else None)


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def pick(clips, k):
    """
    clips -> clip name of each dataset episode, in episode order (meta/clips.json)
    k     -> recordings per pair; each comes with its shifted replays (A1-0~1, ...)
    """
    pairs, chosen = {}, []
    for i, clip in sorted(enumerate(clips), key=lambda ic: episodes.sort_key(ic[1])):
        p = episodes.parse(clip)
        recs = pairs.setdefault((p["plant"], p["position"]), [])
        if p["rep"] not in recs and len(recs) < k:
            recs.append(p["rep"])
        if p["rep"] in recs:
            chosen.append((i, clip))
    chosen.sort()
    return [i for i, _ in chosen], [c for _, c in chosen]
