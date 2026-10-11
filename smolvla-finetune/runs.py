"""The models compared (<init>-<k>, see README.md) and which phone demos each one trains on."""
from dataset import episodes

DATASET = episodes.HUB_DATASET

# The two pretrained checkpoints every model starts from
LIBERO  = "HuggingFaceVLA/smolvla_libero"   # pretrained on LIBERO: same Panda
BASE    = "lerobot/smolvla_base"             # pretrained on SO arms

# Every model compared.
#   start -> checkpoint it starts from
#   k     -> phone demos per plant x start spot it is fine-tuned on (0 = not fine-tuned)
#   repo  -> Hugging Face repo holding it (./run train pushes there, ./run eval loads from there)
MODELS = {
    "libero-0": dict(start=LIBERO,  k=0, repo=LIBERO),  # baseline: smolvla_libero as downloaded
    "libero-1": dict(start=LIBERO,  k=1, repo="edgarcancinoe/smolvla_water_libero-1"),
    "libero-2": dict(start=LIBERO,  k=2, repo="edgarcancinoe/smolvla_water_libero-2"),
    "libero-4": dict(start=LIBERO,  k=4, repo="edgarcancinoe/smolvla_water_libero-4"),
    "base-1":   dict(start=BASE,    k=1, repo="edgarcancinoe/smolvla_water_base-1"),
    "base-2":   dict(start=BASE,    k=2, repo="edgarcancinoe/smolvla_water_base-2"),
    "base-4":   dict(start=BASE,    k=4, repo="edgarcancinoe/smolvla_water_base-4"),
}


CAMERA_NAMES = {
    LIBERO: {},
    BASE: {"observation.images.image": "observation.images.camera1",
           "observation.images.image2": "observation.images.camera2"},
}


def repo(name):
    """Where a model loads from: its Hub repo, or for <model>@<step> (e.g. libero-4@10000) the local
    checkpoint ./run train saved at that step."""
    base, _, step = name.partition("@")
    return f"outputs/train/{base}/checkpoints/{int(step):06d}/pretrained_model" if step else MODELS[base]["repo"]


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def pick(clips, k):
    """
    clips -> clip name of each dataset episode, in episode order (meta/clips.json)
    k     -> demos per pair
    """
    pairs = {}
    for i, clip in sorted(enumerate(clips), key=lambda ic: episodes.sort_key(ic[1])):
        p = episodes.parse(clip)
        pairs.setdefault((p["plant"], p["position"]), []).append((i, clip))
    chosen = sorted(ic for demos in pairs.values() for ic in demos[:k])
    return [i for i, _ in chosen], [c for _, c in chosen]
