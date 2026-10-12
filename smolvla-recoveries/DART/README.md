# DART: demos that show how to recover

**Problem.** The policy reaches the mug but misses the grasp: the demos only show the perfect path. No recoery, no variations.

**Fix** ([DART](https://arxiv.org/abs/1703.09327)). Simple noise during replay, randomly pushing the hand off its path on the way to the mug (~1 cm). We record the scripted clean action, which steers back. Every frame teaches recovery.

| | |
|---|---|
| Runs | 2 noisy replays per demonstration (`A1-0~d1`, `~d2`)|
| Data | demos + shifted replays + DART replays → `edgarcancinoe/water_pour_dart` |
| Model | `libero-4s-dart`: from the pretrained libero checkpoint |

## Run

```sh
./run dart --jobs 8
./run dart-export
./run dart-train        # GPU machine
./run eval-grasp libero-4s-dart --jobs 8
```

## Results (todo)

| model | lifted (grasp eval) | success (full eval) |
|---|---|---|
| `libero-4s` | | |
| `libero-4s-dart` | | |
