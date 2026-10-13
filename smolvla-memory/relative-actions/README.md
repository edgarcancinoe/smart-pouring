# Relative grasp position

SmolVLA can't remember where the mug started, so it can't put it back.

Fix: similar to [grasp-position](../grasp-position/README.md), but slots 9–10 hold `x − gx, y − gy`: where the gripper is relative to the grasp. 

Putting the mug back becomes a task to make this value zero, potentially assisting on generalization.

Model `libero-4s-dart-xl-sd-rel` (= `-sd` + relative grasp position).

```sh
./run train libero-4s-dart-xl-sd-rel
./run relative-eval libero-4s-dart-xl-sd-rel --jobs 8
./run compare
```

| model | success | placed | mug_off_cm 1–4 | mug_off_cm 5–8 |
|---|---|---|---|---|
| `-sd-gp` | | | | |
| `-sd-rel` | | | | |
| `-sd-rel`, blanked | | | | |
