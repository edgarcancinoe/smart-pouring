# Takeover DAgger: corrections where the policy actually goes wrong

**Problem.** Same as [DART](../DART/README.md): the policy identifies mug but misses the grasp and can't how to get back.

**Fix** ([HG-DAgger](https://arxiv.org/abs/1810.02890), human replaced by a state-identiying script). The policy drives; when robot hand is >1 cm off expected path, a scripted teacher steers back. We add these takeovers to our demos, making corrections come from the policy's real mistakes.

| | |
|---|---|
| Teacher | `teacher.py`: the scripted grasp, but it picks its step from where the hand is, so it can take over anywhere (**TODO**) |
| Rollouts | trained spots 1–4, mug ±3 cm, new seeds: the eval stays unseen → `dataset/data/dagger/` |
| Model | `libero-4s-dagger1`: **continues** `libero-4s` or `libero-4s-dart` (to be defined) for (8k?) steps on demos + takeovers |

## Run

```sh
./run dagger-check --jobs 8   # the teacher alone must lift the mug ~every time
./run dagger --jobs 8
./run dagger-export
./run train libero-4s-dagger1           # GPU machine
./run eval-grasp libero-4s-dagger1 --jobs 8
```

## Results

| model | takeovers saved | lifted (grasp eval) | success (full eval) |
|---|---|---|---|
| `libero-4s` | – | | |
| `libero-4s-dagger1` | | | |
