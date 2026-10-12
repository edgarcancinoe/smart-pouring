# DART: demos that show how to recover

**Problem.** The fine-tuned policies reach the mug but miss the grasp by about 1 cm. Every demo follows a perfect
scripted line to the mug, so the policy never sees the hand *off* that line. Once it is a few mm off, it is in a
state it has never seen, and the error grows instead of shrinking (compounding error).

**Idea** ([DART](https://arxiv.org/abs/1703.09327), Laskey et al. 2017). Replay each demo again, but push the hand
off its path with small random noise while the script drives. Record what the script *meant* to do at each
pushed state, not the noisy action that was executed. The script aims at the target from wherever the hand is,
so every recorded action is a correction. The policy learns a tube around the path instead of a single line.

## How

| Step | What |
|---|---|
| Noise | slowly drifting push (std `NOISE_CM` 1 cm, `NOISE_DEG` 2 deg, correlation `RHO` 0.9 per step ≈ 0.5 s), added to the executed action only |
| Where | reach phase only, on the scripted lines (up, across, down, slide in) |
| Clean | the final line-up before closing, the grasp, the pour and the release: no demo closes the gripper off-centre |
| Runs | `RUNS` 2 per recording (`A1-0~d1`, `A1-0~d2`), each with its own ±3 cm mug shift, in `dataset/data/dart/` |
| Kept | successful runs only (water in the plant, mug put back), as for every other demo |
| Dataset | the recordings + shifted replays + DART replays → `edgarcancinoe/water_pour_dart` |
| Model | `libero-4s-dart`: the same settings as `libero-4s` (`smolvla-finetune/runs.py`), on the DART data |

The robot replay itself (`dataset/retarget/robot_replay.py`) is unchanged except for one hook: `perturb()`,
which `dart.py` overrides.

**Starting point: the pretrained checkpoint, not the fine-tuned one.** The DART data includes every demo the
plain model saw, so nothing is lost. The comparison changes one thing only (the data), and we don't start
from a model that already learned to memorize four grasp spots. Continuing from a fine-tuned model is what
DAgger rounds do (`../DAgger`).

## Run

```sh
./run dart --jobs 8     # noisy replays (~5 min each; check a few robot.mp4 in dataset/data/dart/)
./run dart-export       # LeRobot dataset -> Hub
./run dart-train        # GPU machine
./run eval-grasp libero-4s-dart --jobs 8
./run eval libero-4s-dart --jobs 8
```

## Results

| model | lifted (grasp eval) | success (full eval) |
|---|---|---|
| `libero-4s` | | |
| `libero-4s-dart` | | |
