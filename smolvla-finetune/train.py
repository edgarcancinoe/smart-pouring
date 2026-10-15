"""Fine-tune SmolVLA on the phone demos (runs.pick) and push each model to the Hub

Usage: ./run train [libero-4 base-4 ...]   (default: MODELS)
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

import runs

MODELS  = ["libero-4", "libero-2", "libero-1", "base-4", "base-2", "base-1"]  # to train, in order (README.md)
STEPS   = 20_000              # optimizer steps per model
BATCH   = 32                  # samples per step
STATE_DROPOUT = 0.0           # share of training samples with the robot state zeroed (state_dropout.py)
LOG_EVERY = 100             # steps between the loss lines in the log
SAVE_EVERY = 15_000          # steps between local checkpoints (outputs/train/<model>/checkpoints)
WANDB_PROJECT = "smart-pouring"  # live loss curves at wandb.ai/<you>/smart-pouring
OUT = Path("outputs/train")


def cores():
    """CPU cores this machine may use: a pod's cgroup quota (it shows the host's 256 but may use ~27), else all."""
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            quota, period = (Path(path).read_text().split() if path.endswith("max") else
                             (Path(path).read_text(), Path(path.replace("quota", "period")).read_text()))
            if quota.strip() not in ("max", "-1"):
                return max(1, int(int(quota) / int(period)))
        except (OSError, ValueError):
            pass
    return os.cpu_count()


def train(name):
    """Fine-tune one model on its demos, push it with its log and demo list.

    name -> model name, e.g. "libero-2"
    """
    model = runs.MODELS[name]
    start, k, repo = model["start"], model["k"], model["repo"]
    path = runs.MODELS[start]["repo"] if start in runs.MODELS else start  # a model name: continue from that fine-tuned model
    steps = model.get("steps", STEPS)
    data = model.get("data", runs.DATASET)

    if HfApi().file_exists(repo, "demos.json"):  # uploaded last: the model is done
        print(f"{name}: already on the Hub")
        return
    
    clips = json.loads(Path(hf_hub_download(data, "meta/clips.json", repo_type="dataset")).read_text())
    episodes, demos = runs.pick(clips, k)
    print(f"\n=== {name}: {start} + {len(episodes)} demos -> {repo}")

    hub = HfApi()
    hub.create_repo(repo, exist_ok=True)  # no write access: fail now, not after hours of training
    out = OUT / name
    final = out / "checkpoints" / f"{steps:06d}" / "pretrained_model"  # LeRobot saves it before its push
    script = model.get("script", Path(__file__).parent / "state_dropout.py")  # e.g. a smolvla-memory/ variant
    cmd =[sys.executable, str(script), str(model.get("dropout", STATE_DROPOUT)),
           f"--policy.path={path}",
           f"--policy.repo_id={repo}",
           "--policy.push_to_hub=true",  # the checkpoint's own config may say otherwise
           f"--policy.device={runs.device()}",
           f"--policy.scheduler_decay_steps={steps}",  # decay over this run's length (default: 30k steps)
           f"--dataset.repo_id={data}",
           f"--dataset.episodes={episodes}",
           "--dataset.video_backend=pyav",  # AV1 videos: PyAV bundles a decoder; torchcodec uses whatever FFmpeg is installed
           f"--rename_map={json.dumps(runs.CAMERA_NAMES[runs.pretrained(name)])}",
           f"--batch_size={model.get('batch', BATCH)}",
           f"--num_workers={max(4, cores() - 4)}",  # video decoding feeds the GPU: one worker per core the pod may use
           f"--steps={steps}",
           f"--save_freq={model.get('save', SAVE_EVERY)}",
           f"--log_freq={LOG_EVERY}",
           f"--output_dir={out}",
           f"--job_name={name}",  # run name on wandb
           "--wandb.enable=true",
           f"--wandb.project={WANDB_PROJECT}",
           "--wandb.disable_artifact=true"]  # checkpoints go to the Hub, not wandb

    log = OUT / f"{name}.log"
    if final.exists():  # trained before but the push failed: upload it instead of training again
        print(f"{name}: uploading the finished local checkpoint {final}")
        hub.upload_folder(folder_path=final, repo_id=repo)
    else:
        if out.exists():
            shutil.rmtree(out)  # LeRobot refuses an existing output folder
        # show the training output and keep it in a log
        with open(log, "w") as f:
            one_thread = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}  # per worker: threads sized for 256 cores on a 27-core quota stalled every 16 steps
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, **one_thread})
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
        if proc.wait():
            raise SystemExit(f"{name}: training failed, see {log}")

    for ck in sorted((out / "checkpoints").glob("[0-9]*")):  # earlier checkpoints -> branch step-<n>, for ./run eval <model>@<n>
        if int(ck.name) != steps:
            hub.create_branch(repo, branch=f"step-{int(ck.name)}", exist_ok=True)
            hub.upload_folder(folder_path=ck / "pretrained_model", repo_id=repo, revision=f"step-{int(ck.name)}")
    hub.upload_file(path_or_fileobj=log, path_in_repo="train.log", repo_id=repo)
    hub.upload_file(path_or_fileobj=json.dumps(demos, indent=1).encode(), path_in_repo="demos.json", repo_id=repo)
    print(f"{name}: done -> https://huggingface.co/{repo}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or MODELS:
        train(name)


if __name__ == "__main__":
    main()
