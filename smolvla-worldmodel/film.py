"""Film the pours that failed: re-run them with the plain policy (same seeds, so the same trial again) and save
a video of each -> results/pour-plain/<model>/videos/<plant>-<spot>-<trial>.mp4. Failed: carried, not watered.

Usage: ./run pour-film [clips] [--jobs 30]   clips: e.g. right-5-4 (default: every failed pour in results/pour-plain/)
"""
import argparse
import json

import torch

import cheat
import pour  # noqa: F401 (at import: the pour-only trial)
import evaluate as ev
from dataset import episodes

OUT = cheat.RESULTS / cheat.MODEL


def failed():
    """Units of the carried trials that didn't water, from pour-plain's results."""
    return [f.stem for f in sorted(OUT.glob("*.json"))
            if (r := json.loads(f.read_text())["trials"][0])["carried"] and not r["watered"]]


def film(unit):
    """Re-run one trial with video.

    unit -> "<plant>-<spot>-<t>", e.g. "right-5-4"
    """
    plant, spot, t = unit.split("-")
    torch.set_num_threads(1)
    r = cheat.trial(ev.scene(plant), ev.load_model(cheat.MODEL), plant, int(spot),
                    ev.trial_seed(plant, int(spot), int(t)), OUT / "videos" / f"{unit}.mp4")
    return dict(clip=unit, summary=f"{r['in_target']}/10 in the pot, ended: {r['ended']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    p.add_argument("--jobs", type=int, default=30)
    args = p.parse_args()
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    for r in episodes.parallel_map(film, args.clips or failed(), {}, args.jobs, "failed pours"):
        print(f"  {r['clip']}: {r.get('summary', r.get('error'))}", flush=True)
