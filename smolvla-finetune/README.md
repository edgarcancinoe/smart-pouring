# SmolVLA fine-tuning

Fine-tune SmolVLA on the phone demos (the LeRobot dataset from `./run export`), then test every model in the watering scene.

```sh
./run train             # GPU machine
./run eval              # Mac (for graphics!)
./run eval-watch        # eval_watch.py
./run compare           # results/ -> table + plots
```

## Models

A model is named `<init>-<k>` (`runs.py`):

| Part | Meaning |
| --- | --- |
| `init` | `libero`: HuggingFaceVLA/smolvla_libero (Panda) <br>`base`: lerobot/smolvla_base (SO-100/101) |
| `k` | demos per plant (1, 2 or 4 → 12, 24 or 47 episodes); 0 = not fine-tuned |

Models: `libero-0` (baseline), `libero-1/2/4`, `base-1/2/4`. Every run is configured with the same learning parameters for fair comparison.

## Training (Colab)

GPU runtime, one cell each:

```text
!git clone https://github.com/edgarcancinoe/smart-pouring.git
%cd smart-pouring
!pip install -q "lerobot[smolvla,dataset]==0.6.1" && hf auth login
!./run train
```

## Evaluation

3 plants × 8 mug start spots × 7 trials per model = 168

- Spots 1–4 are where the phone demos started
- Spots 5–8 are new, 7–10 cm from every trained spot

##### Each trial: 

- Mug on its spot (±1 cm, seeded)
- Robot in the start posture
- Command `Water the <plant> plant`
- Policy runs 10 actions of each predicted chunk before looking again.

##### Termination conditions:

- Mug has been touched, it rests, and it's been released (we opened the gripper)
- Episode has ran for 60s 
- Mug has been lifted

##### On a GPU (Colab)

Same code, the policy runs on CUDA; `./run` renders headless (`MUJOCO_GL=egl`) on Linux. After the clone cell above:

```text
!pip install -q "lerobot[smolvla,libero]==0.6.1" "hf-libero==0.1.4" "mujoco>=3.0,<3.9"
!./run eval --jobs 4             # every model in evaluate.MODELS
!./run eval libero-4 --jobs 4    # (libero-4 repetitions policy)
```

Then copy `smolvla-finetune/results/` back to the Mac for `./run compare`.

##### Results
In `results/<model>/<plant>-<spot>.json` plus `results/<model>/videos/<plant>-<spot>.mp4`.

| Field | Meaning |
| --- | --- |
| `in_target` / `in_wrong` / `spilled` / `in_mug` | where the 30 droplets ended |
| `watered` | at least half of spheres are inside the commanded plant |
| `lifted` | the mug rose 3 cm off the table |
| `placed` | the mug stands upright, released, within 3 cm of where it started |
| `success` | watered and placed |
