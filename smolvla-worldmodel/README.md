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
- `log`: one entry per replan: mug mouth vs target pot (cm), tilt, drops in the mug / over the pot / elsewhere; for the cheat, the 8 scores, the pick and the spread of the 8 end positions (cm).
- `rewind_mm`: the same chunk run twice from the same saved state; 0 means the rewind is exact (checked on the Mac: 0.0).

## Reading the result (successes of 36; dagger1: 13)

| Cheat | Meaning | Next |
|---|---|---|
| ≥ 21 | Good pours exist among the policy's own options | learned judge: simple scorer, then world model |
| 17–20 | Unclear | 24 more trials (60 in all), go if ≥ 31 |
| ≤ 16 | All options are bad from unseen spots | more spread if the 8 were alike; else wider data, no world model |

Plan: https://claude.ai/artifact/CXUSRic5UALTvam4SzRcDN
