"""The exported LeRobot dataset -> the Hugging Face Hub
Usage: ./run upload
"""
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dataset import episodes
from dataset.export_lerobot import NAME

def upload(name=NAME, repo=episodes.HUB_DATASET):
    """dataset/data/lerobot/<name>/ -> the Hub dataset repo (public).

    name -> local dataset folder name
    repo -> Hub dataset repo id
    """
    root = episodes.DATA / "lerobot" / name
    if not (root / "meta" / "clips.json").exists():
        raise SystemExit(f"{root}/meta/clips.json missing (training picks demos by clip): re-run the export")
    ds = LeRobotDataset(repo, root=root)
    print(f"{ds.num_episodes} episodes, {ds.num_frames} frames -> {repo}")
    ds.push_to_hub(private=False)
    print(f"uploaded: https://huggingface.co/datasets/{repo}")


def main():
    upload()


if __name__ == "__main__":
    main()
