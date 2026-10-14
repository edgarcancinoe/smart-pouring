# VLM-cache, isolated

[VLM-cache](../VLM-cache/README.md) with one change: inside the VLM, the present (cameras, command, state) doesn't
read the start frame. The present is processed exactly as without memory; only the action expert reads the memory,
next to the present. Same cost, same cache. Everything else is imported from `../VLM-cache/memory.py`.

Model `libero-4s-dart-xl-sd-memiso` (= `-sd-mem` with the memory isolated).

```sh
./run train libero-4s-dart-xl-sd-memiso
./run isolated-eval libero-4s-dart-xl-sd-memiso --jobs 8
./run compare
```
