# World model: can a judge fix the pour?

The best policy (dagger1) lifts the mug almost every time, then fails the pour from unseen spots: it pours beside the pot or never tips the mug. A world model would act as a judge: sample several chunks from SmolVLA, predict where each leads, run the best one. Before building one, the cheat test measures the most such a judge could add.

## Cheat test (`cheat.py`)

After the lift, at each replan (every 10 actions):

1. Save the sim state (MuJoCo data, controller, grasp memory).
2. Sample `K` = 8 chunks from SmolVLA in one batch (different noise).
3. For each: run its first 20 actions, score it, rewind.
4. Run the best chunk's first 10 actions.

Score: 10 per drop over the target pot (in it or falling into it), −10 per drop out of the mug elsewhere, minus the mug mouth's xy distance to the pot centre in cm. The cheat stops once the mug is empty (the put-back is the plain policy).

No learned judge can choose better than this, so the cheat's gain over dagger1 is the ceiling.

Trials: 3 plants × spots 5–8 × 3 = 36, the first 3 seeds of the main eval. `./run cheat-plain` runs the same trials without the cheat, with the same log.

Each trial in `results/<arm>/<model>/<plant>-<spot>-<trial>.json` (one job per trial) adds:
- `log`: one entry per replan: mug mouth vs target pot (cm), tilt, drops in the mug / over the pot / elsewhere; why a mug drops: `grip_cmd` (lowest gripper command over the last chunk, < 0 = open), `gap_mm` (fingers), `hand_mug_cm`, `mug_on_pot`; for the cheat, the 8 scores, the pick and the spread of the 8 end positions (cm).
- `rewind_mm`: the same chunk run twice from the same saved state; 0 means the rewind is exact (checked on the Mac: 0.0).

## Reading the result (successes of 36; dagger1: 13)

| Cheat | Meaning | Next |
|---|---|---|
| ≥ 21 | Good pours exist among the policy's own options | learned judge: simple scorer, then world model |
| 17–20 | Unclear | 24 more trials (60 in all), go if ≥ 31 |
| ≤ 16 | All options are bad from unseen spots | more spread if the 8 were alike; else wider data, no world model |

Result (36 paired trials): watered 22 → 24, success 13 → 14. The 8 chunks are nearly identical (median 0.5 cm apart after 1 s), so there is little to choose; it helped only where they differed (right-5).

## Pour-only benchmark (`pour.py`)

Each trial replays the plain policy until the mug is carried (5 cm up, under 30° tilt; same seeds as the main eval, so this part is exact), then the arm under test pours. It ends 1 s after the mug is empty, or when the mug is back on the table, or after 30 s. Trials where the mug is knocked over or never held end early and don't count.

Score: drops in the target pot (`in_target`, of 10) over the carried trials (`carried` is not null). `ended` says why each stopped. The put-back never runs, so `placed`/`success` are always false here.

`./run pour-plain` is the baseline; `./run pour` the cheat; `./run pour-distance` the cheat scored by mouth-to-pot distance only (no water), to see whether a judge must predict the water or only where the mug goes.

Result (60 seeds): plain 30/40 held mugs watered (7.4 drops), cheat 40/41 (9.4); paired on the 36 trials whose pour started the same, 28 → 35. Distance only: 31/41 (7.4), no better than plain: it fixes the tip-untip pours but every middle-5 pour tips near the pot and misses. A judge has to predict the water.

## Controls

- **A, random of 8** (`./run pour-random`): the cheat's 8 look-aheads, then a random one runs. Should equal plain: the cheat's gain is the choice, not the resampling or the rewinds. Result: 31/46 held mugs watered (6.9 drops); paired on the 34 trials both carried, plain 26, random 25. It does: the gain is the choice.
- **B, the cheat distilled** (`distill.py`): the cheat pours at the trained spots 1-4 (DAgger's seeds and jitter), recorded from the carry on like a DAgger takeover; the watered ones join dagger1's data and dagger1 trains 8k steps more (`…-dagger1-cheat`). Pour-tested on spots 5-8 (`./run pour-distilled`). Near the cheat's 40/41: no judge needed at run time.

Plan: https://claude.ai/artifact/CXUSRic5UALTvam4SzRcDN
