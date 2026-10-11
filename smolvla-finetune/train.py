"""Fine-tune SmolVLA on the phone demos (runs.pick) and push each model to the Hub

Usage: ./run train
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

import runs

MODELS  = ["libero-4", "libero-2", "libero-1", "base-4", "base-2", "base-1"]  # to train, in order (README.md)
STEPS   = 10_000              # optimizer steps per model
BATCH   = 32                  # samples per step
LOG_EVERY = 100             # steps between the loss lines in the log
OUT = Path("outputs/train")


def train(name, clips):
    """Fine-tune one model on its demos, push it with its log and demo list.

    name  -> model name, e.g. "libero-2"
    clips -> clip name of each dataset episode, in episode order
    """
    model = runs.MODELS[name]
    start, k, repo = model["start"], model["k"], model["repo"]

    if HfApi().file_exists(repo, "demos.json"):  # uploaded last: the model is done
        print(f"{name}: already on the Hub")
        return
    
    episodes, demos = runs.pick(clips, k)
    print(f"\n=== {name}: {start} + {len(episodes)} demos -> {repo}")

    out = OUT / name
    shutil.rmtree(out, ignore_errors=True)  # LeRobot refuses an existing output folder
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
           f"--policy.path={start}",
           f"--policy.repo_id={repo}",
           "--policy.push_to_hub=true",  # the checkpoint's own config may say otherwise
           f"--policy.device={runs.device()}",
           f"--policy.scheduler_decay_steps={STEPS}",  # decay over this run's length (default: 30k steps)
           f"--dataset.repo_id={runs.DATASET}",
           f"--dataset.episodes={episodes}",
           f"--rename_map={json.dumps(runs.CAMERA_NAMES[start])}",
           f"--batch_size={BATCH}",
           f"--steps={STEPS}",
           f"--save_freq={STEPS}",
           f"--log_freq={LOG_EVERY}",
           f"--output_dir={out}",
           "--wandb.enable=false"]

    # show the training output and keep it in a log
    log = OUT / f"{name}.log"
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
    if proc.wait():
        raise SystemExit(f"{name}: training failed, see {log}")

    hub = HfApi()
    hub.upload_file(path_or_fileobj=log, path_in_repo="train.log", repo_id=repo)
    hub.upload_file(path_or_fileobj=json.dumps(demos, indent=1).encode(), path_in_repo="demos.json", repo_id=repo)
    print(f"{name}: done -> https://huggingface.co/{repo}")


def main():
    clips = json.loads(Path(hf_hub_download(runs.DATASET, "meta/clips.json", repo_type="dataset")).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        train(name, clips)


if __name__ == "__main__":
    main()
