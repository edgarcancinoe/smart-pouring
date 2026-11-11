"""Outcome judge, step 2: is there a choice to learn, and where? Both recordings (judge.py: pour only; judge_full.py:
whole episodes), per replan: the K chunks' scores after their look-ahead, by phase. How often they differ, how much the
best beats the chunk that ran (a random one), and when in the episode that happens -> results/judge/<model>/check.png
+ check.json.

Scores by phase (how a chunk is labelled; the judge never sees the phase), all - KNOCK if the mug ends knocked over
and out of the hand:
  grasp             + LIFTED if the mug ends held and lifted (milestones only: the pilot's distance terms were noise)
  pour start, tip   cheat.score: (drops over the pot - drops elsewhere) x DROP_WEIGHT - the mouth's xy distance (cm)
  put back          - HOME_WEIGHT x the mug's distance from where it stood (cm) + PLACED if placed as the eval counts it (released,
                    upright, near its start, on the table), - drops elsewhere x DROP_WEIGHT

The recordings come from results/<recording>/<model>/data/, or from the Hub if they aren't here.

Usage: ./run judge-check
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download

import evaluate as ev
from dataset.export_lerobot import CAMERAS
from sim.watering import CONTROL_HZ

MODEL = "libero-4s-dart-full-xxl-sd-rel-b64-dagger1"  # the policy the recordings ran
HUB = "edgarcancinoe/smolvla_water_results"  # results/ uploaded as worldmodel/
RESULTS = Path(__file__).parent / "results"
RECORDINGS = ("judge-data", "judge-full")  # judge.py's, judge_full.py's
OUT = RESULTS / "judge" / MODEL
PHASES = ("grasp", "pour start", "tip", "put back")
TIP_DEG = 20.0          # carried mug tilted at least this: the tip; less: the start of the pour (judge_full.py too)
DROP_WEIGHT = 10.0      # cheat.py's score: per drop over the pot (minus per drop elsewhere), vs cm of mouth distance
KNOCK = 100.0           # score lost if the mug ends knocked over
LIFTED = 50.0           # grasp: the mug ends held and lifted (evaluate's LIFT_CM)
PLACED = 50.0           # put back: placed as the eval counts it
HOME_WEIGHT = 0.2       # put back: per cm from where the mug stood (small: a tie-break, not a choice)
DECISIVE = 10.0         # score points: a replan where the best of K beats the worst by about a drop or more
EXAMPLES = 4            # most decisive replans drawn with their camera frame


def data_dir(recording):
    """A recording's folder of <unit>.npz: here, else downloaded from the Hub (None: not recorded yet)."""
    local = RESULTS / recording / MODEL / "data"
    if any(local.glob("*.npz")):
        return local
    hub = Path(snapshot_download(HUB, repo_type="dataset", allow_patterns=f"worldmodel/{recording}/{MODEL}/data/*"))
    hub = hub / "worldmodel" / recording / MODEL / "data"
    return hub if any(hub.glob("*.npz")) else None


def score(d):
    """Each chunk's score after its look-ahead, by the replan's phase (module docstring) -> (R, K)."""
    released = d["hand_cm"] > ev.RELEASE_CM
    knock = -KNOCK * ((d["tilt"] > ev.TOPPLED_DEG) & released)
    water = DROP_WEIGHT * (d["over"] - d["elsewhere"]) - np.linalg.norm(d["mouth_cm"][..., :2], axis=-1)
    grasp = LIFTED * (~released & (d["up_cm"] >= ev.LIFT_CM))
    placed = released & (d["home_cm"] <= ev.PLACED_CM) & (d["tilt"] <= ev.PLACED_DEG) & (np.abs(d["up_cm"]) < 1)
    back = -HOME_WEIGHT * d["home_cm"] + PLACED * placed - DROP_WEIGHT * d["elsewhere"]
    phase = d["phase"][:, None]
    return knock + np.select([phase == "grasp", phase == "put back"], [grasp, back], water)


def load(images=False):
    """Every replan of both recordings, stacked: the recorders' arrays (R, ...) plus trial (index in files), row (replan
    number in its trial), file_row (its row in the file), plant, watered (the trial's outcome), recording; the camera frames only if images.
    shortcut: judge.py's rows have no phase or state then; kept as the tip where a chunk tipped the mug past TIP_DEG
    within its 1 s, else dropped (the start of the pour there was labelled at 1 s); exact once judge_full.py's data
    is all there is"""
    files = [f for r in RECORDINGS if data_dir(r) for f in sorted(data_dir(r).glob("*.npz"))]
    assert files, "no recording: run ./run judge-full (or ./run judge-record) first"
    parts = []
    for i, f in enumerate(files):
        z = np.load(f)
        part = {k: z[k] for k in z.files if z[k].ndim and (images or k not in CAMERAS)}
        n = len(part["pick"])
        keep = np.ones(n, bool)
        if "phase" not in part:  # judge.py's
            keep = part["tilt"].max(1) >= TIP_DEG
            part.update(phase=np.full(n, "tip"), hand_cm=np.zeros_like(part["tilt"]), home_cm=np.zeros_like(part["tilt"]),
                        up_cm=np.zeros_like(part["tilt"]),
                        horizon=np.full(n, 20), now_tilt=np.full(n, np.nan), now_in_mug=np.full(n, -1))
        part["file_row"] = np.arange(n)
        part = {k: v[keep] for k, v in part.items()}
        n = int(keep.sum())
        part.update(trial=np.full(n, i), row=np.arange(n), plant=np.full(n, str(z["plant"])),
                    watered=np.full(n, bool(z["watered"])), recording=np.full(n, f.parent.parent.parent.name))
        parts.append(part)
    keys = set.intersection(*(set(p) for p in parts))
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d["phase"] = d["phase"].astype(str)
    d["files"] = files
    return d


def check(d):
    """The numbers, overall and per phase: how often the K chunks differ, what picking the best would gain over the one
    that ran."""
    s = score(d)  # (R, K)
    ran, best, worst = s[np.arange(len(s)), d["pick"]], s.max(1), s.min(1)
    decisive = best - worst >= DECISIVE

    def stats(m):
        return dict(replans=int(m.sum()), decisive=round(float(decisive[m].mean()), 3) if m.any() else None,
                    decisive_replans=int(decisive[m].sum()),
                    gain_best_vs_ran=round(float((best - ran)[m].mean()), 2) if m.any() else None,
                    gain_best_vs_ran_decisive=round(float((best - ran)[m & decisive].mean()), 2) if (m & decisive).any() else None)
    return s, dict(trials=len(d["files"]), all=stats(np.ones(len(s), bool)),
                   **{p: stats(d["phase"] == p) for p in PHASES},
                   **{r: stats(d["recording"] == r) for r in RECORDINGS})


def plot(d, s, stats, path):
    """Top: spread of the K scores, share decisive per phase, when in the episode the choice matters.
    Below: the most decisive replans, both cameras and each chunk's score."""
    best, worst = s.max(1), s.min(1)
    colors = dict(zip(PHASES, ("#8a6a2f", "#7b4fa0", "#1f6f99", "#5b6d75")))
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    grid = fig.add_gridspec(3, max(3, EXAMPLES), height_ratios=[1.1, 1, 0.7])
    fig.suptitle(f"Outcome judge data: {stats['trials']} trials, {stats['all']['replans']} replans x {s.shape[1]} chunks · "
                 f"decisive (≥ {DECISIVE:g} pts) in {stats['all']['decisive']:.0%} ({stats['all']['decisive_replans']})")

    ax = fig.add_subplot(grid[0, 0])
    ax.hist(np.clip(best - worst, 0, 100), bins=40, color="#1f6f99")
    ax.axvline(DECISIVE, color="#a8442f", ls="--", label=f"decisive ≥ {DECISIVE:g}")
    ax.set(title="Best − worst of the K scores per replan", xlabel="score points (clipped at 100)", ylabel="replans", yscale="log")
    ax.legend()

    ax = fig.add_subplot(grid[0, 1])
    shares = [stats[p]["decisive"] or 0 for p in PHASES]
    bars = ax.bar(PHASES, shares, color=[colors[p] for p in PHASES])
    ax.bar_label(bars, [f"{stats[p]['decisive_replans']}/{stats[p]['replans']}" for p in PHASES])
    ax.set(title="Share of replans where the choice matters", ylabel="share decisive", ylim=(0, max(shares + [0.1]) * 1.25))

    ax = fig.add_subplot(grid[0, 2])
    full = d["recording"] == "judge-full"
    for p in PHASES:
        m = full & (d["phase"] == p)
        ax.scatter(d["step"][m] / CONTROL_HZ, (best - worst)[m], s=6, alpha=0.5, color=colors[p], label=p)
    ax.axhline(DECISIVE, color="#a8442f", ls="--", lw=1)
    ax.set(title="When in the episode (whole-episode recording)", xlabel="time in the episode (s)",
           ylabel="best − worst (score points)", yscale="symlog")
    ax.legend(fontsize=8)

    shown, trials = [], set()
    for r in np.argsort(-(best - worst)):  # most decisive first, one per trial
        if d["trial"][r] not in trials:
            shown.append(r)
            trials.add(d["trial"][r])
        if len(shown) == EXAMPLES:
            break
    for j, r in enumerate(shown):
        f = d["files"][d["trial"][r]]
        z = np.load(f)
        row = d["file_row"][r]
        ax = fig.add_subplot(grid[1, j])
        ax.imshow(np.concatenate([z[cam][row] for cam in CAMERAS], 1))
        ax.set_title(f"{f.stem} · {d['phase'][r]} · {d['step'][r] / CONTROL_HZ:.1f} s", fontsize=10)
        ax.axis("off")
        ax = fig.add_subplot(grid[2, j])
        k = np.arange(s.shape[1])
        ax.bar(k, s[r], color=colors[d["phase"][r]])
        ax.axvline(d["pick"][r], color="#5b6d75", ls=":", label="ran")
        ax.axvline(np.argmax(s[r]), color="#a8442f", ls="--", label="best")
        ax.set(xticks=k, xlabel=f"chunk ({d['horizon'][r] / CONTROL_HZ:g} s look-ahead)", ylabel="score" if j == 0 else None)
        if j == 0:
            ax.legend(fontsize=8)
    fig.savefig(path, dpi=110)


def main():
    d = load()
    s, stats = check(d)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(stats, indent=2))
    plot(d, s, stats, OUT / "check.png")
    print(json.dumps(stats, indent=2))
    print(f"-> {OUT / 'check.png'}")


if __name__ == "__main__":
    main()
