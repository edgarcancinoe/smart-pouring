# Grasp position

SmolVLA can't remember where the mug started, so it can't put it back.
Fix 1: when the gripper starts closing, its `x, y` go into free state slots 8–10 as `[grasped, x, y]` (zeros before).

Model `libero-4s-dart-xl-sd-gp` (= `-sd` + grasp position).

```sh
./run train libero-4s-dart-xl-sd-gp
./run grasp-pos-eval libero-4s-dart-xl-sd-gp --jobs 8
./run compare
```

Ablation: `BLANK = True` in `eval_grasp_position.py` keeps the slots at zero.

| model | success | placed | mug_off_cm 1–4 | mug_off_cm 5–8 |
|---|---|---|---|---|
| `-sd` | | | | |
| `-sd-gp` | | | | |
| `-sd-gp`, blanked | | | | |
