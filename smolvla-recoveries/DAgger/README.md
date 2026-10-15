# Takeover DAgger ([rel](../../smolvla-memory/relative-actions/README.md) strategy)

The policy drives; when its hand is >1 cm off the demo's path, a (scripted) teacher takes over and steers it back. Takeovers are saved and added to the demos ([HG-DAgger](https://arxiv.org/abs/1810.02890)).

- **Teacher:** follows the whole demo from any state (reach → carry → release).
- **Memory:** each takeover keeps the policy's grasp memory as a `grasp_memory` column, so short clips still know the mug is held (for the memory enabling '[rel](../../smolvla-memory/relative-actions/README.md)' strategy to put cup back)

## Run

```sh
./run dagger-check --jobs 8   # teacher alone: must water + put back ~every time
./run dagger
./run dagger-export
./run train libero-4s-dart-full-xxl-sd-rel-b64-dagger1
./run relative-eval libero-4s-dart-full-xxl-sd-rel-b64-dagger1
```

## Results

| model | takeovers | success, trained spots | success, new spots |
|---|---|---|---|
| `libero-4s-dart-full-xxl-sd-rel-b64` | – | 25/60 | 7/60 |
| `…-dagger1` | | | |
