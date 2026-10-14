# VLM-cache memory

The VLM tokens of the episode's first agentview frame go in front of the multimodal prefix.

- Prefix can read these memories in every step 
- Action expert has access to the memory-enriched prefix and also the memory itself to attend to
- Memory tokens are fixed in every chunk and step. Eval computes them once per trial and caches them.

The VLM is frozen (`train_expert_only`), so the expert learns to read the memory.


Model `libero-4s-dart-xl-sd-mem` (= `-sd` + memory).

```sh
./run train libero-4s-dart-xl-sd-mem
./run memory-eval libero-4s-dart-xl-sd-mem --jobs 8
./run compare
```

| model | success | placed | mug_off_cm 1–4 | mug_off_cm 5–8 |
|---|---|---|---|---|
| `-sd-gp` | | | | |
| `-sd-mem` | | | | |
| `-sd-mem`, blanked | | | | |
