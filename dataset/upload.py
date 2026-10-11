"""The exported LeRobot dataset -> the Hugging Face Hub
Usage: ./run upload
"""
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dataset import episodes
from dataset.export_lerobot import NAME

def main():
    root = episodes.DATA / "lerobot" / NAME
    if not (root / "meta" / "clips.json").exists():
        raise SystemExit(f"{root}/meta/clips.json missing (training picks demos by clip): re-run ./run export")
    ds = LeRobotDataset(episodes.HUB_DATASET, root=root)
    print(f"{ds.num_episodes} episodes, {ds.num_frames} frames -> {episodes.HUB_DATASET}")
    ds.push_to_hub(private=False)
    print(f"uploaded: https://huggingface.co/datasets/{episodes.HUB_DATASET}")


if __name__ == "__main__":
    main()
