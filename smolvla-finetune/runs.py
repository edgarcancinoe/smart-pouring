"""The models compared (<init>-<k>, see README.md) and which phone demos each one trains on."""
from dataset import episodes

DATASET = "edgarcancinoe/water_pour"  # the demos with the mug exactly on its 4 measured spots
SHIFTED = episodes.HUB_DATASET       # the same + each replayed with the mug shifted (robot_replay.py SHIFTS)
DART    = "edgarcancinoe/water_pour_dart"  # SHIFTED + each replayed with noise pushes on the way to the mug (smolvla-recoveries/DART)

# The two pretrained checkpoints every model starts from
LIBERO  = "HuggingFaceVLA/smolvla_libero"   # pretrained on LIBERO: same Panda
BASE    = "lerobot/smolvla_base"             # pretrained on SO arms

# Every model compared.
#   start -> checkpoint it starts from, or another model's name: continue fine-tuning that model (DAgger)
#   k     -> phone demos per plant x start spot it is fine-tuned on (0 = not fine-tuned)
#   repo  -> Hugging Face repo holding it (./run train pushes there, ./run eval loads from there)
#   data  -> dataset it trains on (default DATASET)
#   batch -> samples per step (default train.BATCH)
#   dropout -> share of training samples with the robot state zeroed (default train.STATE_DROPOUT)
#   steps -> optimizer steps (default train.STEPS)
#   save  -> steps between checkpoints, each also pushed to the branch step-<step> (default train.SAVE_EVERY)
#   script -> training script, called like state_dropout.py: <share> <lerobot args> (default state_dropout.py)
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
    "libero-4s-dart": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart", data=DART),
    "libero-4s-dart-xl": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl",
                              data="edgarcancinoe/water_pour_dart_xl"),  # 16 DART runs per recording
    "base-4s-dart-xl":   dict(start=BASE, k=4, repo="edgarcancinoe/smolvla_water_base-4s-dart-xl",
                              data="edgarcancinoe/water_pour_dart_xl"),
    "libero-4s-dart-xl-b64": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-b64",
                                  data="edgarcancinoe/water_pour_dart_xl", batch=64),
    # DART xl + state dropout: does dropout's gain at new spots add up with the best data?
    "libero-4s-dart-xl-sd": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-sd",
                                 data="edgarcancinoe/water_pour_dart_xl", dropout=0.3),
    "libero-4s-dart-xl-sd-b64": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-sd-b64",
                                     data="edgarcancinoe/water_pour_dart_xl", dropout=0.3, batch=64, steps=20_000, save=15_000),
    "base-4s-dart-xl-sd":  dict(start=BASE, k=4, repo="edgarcancinoe/smolvla_water_base-4s-dart-xl-sd",
                                 data="edgarcancinoe/water_pour_dart_xl", dropout=0.3),
    # + the grasp position in the state: where the mug was picked up (smolvla-memory/grasp-position)
    "libero-4s-dart-xl-sd-gp": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-sd-gp",
                                    data="edgarcancinoe/water_pour_dart_xl", dropout=0.3,
                                    script="smolvla-memory/grasp-position/grasp_position.py"),
    # the same, retrained once the memory waited for the fingers to stop on the mug (gp: set when closing began)
    "libero-4s-dart-xl-sd-gp2": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-sd-gp2",
                                    data="edgarcancinoe/water_pour_dart_xl", dropout=0.3,
                                    script="smolvla-memory/grasp-position/grasp_position.py"),
    # + where the gripper is relative to that grasp position (smolvla-memory/relative-actions)
    "libero-4s-dart-xl-sd-rel": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-sd-rel",
                                    data="edgarcancinoe/water_pour_dart_xl", dropout=0.3,
                                    script="smolvla-memory/relative-actions/relative.py"),
    "libero-4s-dart-xl-sd-rel-b64": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-sd-rel-b64",
                                        data="edgarcancinoe/water_pour_dart_xl", dropout=0.3, batch=64,
                                        script="smolvla-memory/relative-actions/relative.py"),
    # rel on DART full: pushes on the carry, pour and put-back 
    "libero-4s-dart-full-xl-sd-rel":    dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xl-sd-rel",
                                            data="edgarcancinoe/water_pour_dart_full-xl", dropout=0.3,
                                            script="smolvla-memory/relative-actions/relative.py"),
    "libero-4s-dart-full-xl-sd-rel-b64":    dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xl-sd-rel-b64",
                                                data="edgarcancinoe/water_pour_dart_full-xl", dropout=0.3, batch=64,
                                                script="smolvla-memory/relative-actions/relative.py"),
    "libero-4s-dart-full-xxl-sd-rel":   dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xxl-sd-rel",
                                            data="edgarcancinoe/water_pour_dart_full-xxl", dropout=0.3,
                                            script="smolvla-memory/relative-actions/relative.py"),
    "libero-4s-dart-full-xxl-sd-rel-b64":   dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xxl-sd-rel-b64",
                                               data="edgarcancinoe/water_pour_dart_full-xxl", dropout=0.3, batch=64,
                                               script="smolvla-memory/relative-actions/relative.py"),
    # the same on anchored-pour replays
    "libero-4s-dart-full-anchor-xl-sd-rel-b64": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-anchor-xl-sd-rel-b64",
                                                     data="edgarcancinoe/water_pour_dart_full_anchor-xl", dropout=0.3, batch=64,
                                                     script="smolvla-memory/relative-actions/relative.py"),
    # + the VLM's keys and values of the first agentview frame, attended to all episode (smolvla-memory/VLM-cache)
    "libero-4s-dart-xl-sd-mem": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-sd-mem",
                                     data="edgarcancinoe/water_pour_dart_xl", dropout=0.3,
                                     script="smolvla-memory/VLM-cache/memory.py"),
    # + the same, but only the expert reads the memory: the present doesn't (smolvla-memory/VLM-cache-isolated)
    "libero-4s-dart-xl-sd-memiso": dict(start=LIBERO, k=4, repo="edgarcancinoe/smolvla_water_libero-4s-dart-xl-sd-memiso",
                                        data="edgarcancinoe/water_pour_dart_xl", dropout=0.3,
                                        script="smolvla-memory/VLM-cache-isolated/isolated.py"),
    # full-xxl-rel-b64 dagger
    "libero-4s-dart-full-xxl-sd-rel-b64-dagger1": dict(start="libero-4s-dart-full-xxl-sd-rel-b64", k=4,
                                                       repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xxl-sd-rel-b64-dagger1",
                                                       data="edgarcancinoe/water_pour_dagger1", dropout=0.3, batch=64, steps=8_000,
                                                       script="smolvla-memory/relative-actions/relative.py"),
    # full-xxl-rel-b64 dagger, round 2: continues round 1 on its own rollouts' takeovers
    "libero-4s-dart-full-xxl-sd-rel-b64-dagger2": dict(start="libero-4s-dart-full-xxl-sd-rel-b64-dagger1", k=4,
                                                       repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xxl-sd-rel-b64-dagger2",
                                                       data="edgarcancinoe/water_pour_dagger2", dropout=0.3, batch=64, steps=8_000,
                                                       script="smolvla-memory/relative-actions/relative.py"),
    # full-xxl-rel-b64 dagger, rounds 1 + 2 pooled: from the pre-DAgger model, as dagger1 (smolvla-recoveries/DAgger: ./run dagger-merge)
    "libero-4s-dart-full-xxl-sd-rel-b64-dagger12": dict(start="libero-4s-dart-full-xxl-sd-rel-b64", k=4,
                                                        repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xxl-sd-rel-b64-dagger12",
                                                        data="edgarcancinoe/water_pour_dagger12", dropout=0.3, batch=64, steps=8_000,
                                                        script="smolvla-memory/relative-actions/relative.py"),
    # dagger12 + 4k steps on the same data: 12k in all, dagger1's passes over the data (1228 episodes vs 844)
    "libero-4s-dart-full-xxl-sd-rel-b64-dagger12-12k": dict(start="libero-4s-dart-full-xxl-sd-rel-b64-dagger12", k=4,
                                                            repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xxl-sd-rel-b64-dagger12-12k",
                                                            data="edgarcancinoe/water_pour_dagger12", dropout=0.3, batch=64, steps=4_000,
                                                            script="smolvla-memory/relative-actions/relative.py"),
    # dagger1 trained on with the cheat's watered pours at spots 1-4 (smolvla-worldmodel/distill.py: control B)
    "libero-4s-dart-full-xxl-sd-rel-b64-dagger1-cheat": dict(start="libero-4s-dart-full-xxl-sd-rel-b64-dagger1", k=4,
                                                             repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-xxl-sd-rel-b64-dagger1-cheat",
                                                             data="edgarcancinoe/water_pour_dagger1_cheat", dropout=0.3, batch=64, steps=8_000,
                                                             script="smolvla-memory/relative-actions/relative.py"),
    # anchor-xl-rel-b64 dagger
    "libero-4s-dart-full-anchor-xl-sd-rel-b64-dagger1": dict(start="libero-4s-dart-full-anchor-xl-sd-rel-b64", k=4,
                                                             repo="edgarcancinoe/smolvla_water_libero-4s-dart-full-anchor-xl-sd-rel-b64-dagger1",
                                                             data="edgarcancinoe/water_pour_anchor_dagger1", dropout=0.3, batch=64, steps=8_000,
                                                             script="smolvla-memory/relative-actions/relative.py"),
}


CAMERA_NAMES = {
    LIBERO: {},
    BASE: {"observation.images.image": "observation.images.camera1",
           "observation.images.image2": "observation.images.camera2"},
}


def pretrained(name):
    """The pretrained checkpoint (LIBERO or BASE) a model comes from, through the models it continues."""
    start = MODELS[name]["start"]
    return pretrained(start) if start in MODELS else start


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
    k     -> recordings per pair; each comes with its shifted replays (A1-0~1, ...); DAgger takeovers are always kept
    """
    pairs, chosen, seen = {}, [], set()
    for i, clip in sorted(enumerate(clips), key=lambda ic: episodes.sort_key(ic[1])):
        if clip in seen:  # merged DAgger datasets carry the same demos twice: keep the first
            continue
        seen.add(clip)
        p = episodes.parse(clip)
        if p is None:  # a DAgger takeover (r1-left-2-0-t0): not from one recording
            chosen.append((i, clip))
            continue
        recs = pairs.setdefault((p["plant"], p["position"]), [])
        if p["rep"] not in recs and len(recs) < k:
            recs.append(p["rep"])
        if p["rep"] in recs:
            chosen.append((i, clip))
    chosen.sort()
    return [i for i, _ in chosen], [c for _, c in chosen]
