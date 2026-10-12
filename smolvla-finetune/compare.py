"""Every evaluated model side by side: results/summary.md, success_vs_k.png, droplets.png, losses.png. See README.md.

Usage: ./run compare
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

import runs
from sim.watering.assets import DROPLET
from train import LOG_EVERY

RESULTS = Path(__file__).parent / "results"
COLORS = {runs.LIBERO: "#2a6fdb", runs.BASE: "#e07b20"}  # one hue per start checkpoint (blue / orange: colorblind-safe)
DROPLET_COLORS = {"in_target": "#2a6fdb", "in_wrong": "#d6453d", "spilled": "#9aa0a6", "in_mug": "#e0b84a"}


def losses():
    """Training loss of every fine-tuned model, from the train.log on its Hub repo -> results/losses.png."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, m in runs.MODELS.items():
        if not m["k"]:
            continue
        try:
            log = Path(hf_hub_download(m["repo"], "train.log")).read_text()
        except (EntryNotFoundError, RepositoryNotFoundError):
            continue  # not trained yet
        loss = [float(x) for x in re.findall(r" loss:([\d.]+)", log)]  # one line per LOG_EVERY steps
        ax.plot(np.arange(1, len(loss) + 1) * LOG_EVERY, loss, color=COLORS[m["start"]],
                ls="-." if "data" in m else {1: ":", 2: "--", 4: "-"}[m["k"]], lw=1.5, label=name)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "losses.png", dpi=150)
    print(f"-> {RESULTS}/losses.png")


def load():
    """{model: [trial dicts, each with plant, spot, trained_spot added]} for models with results."""
    out = {}
    for name in [n for m in runs.MODELS for n in [m, *sorted(p.name for p in RESULTS.glob(f"{m}@*"))]]:
        files = sorted((RESULTS / name).glob("*.json"))
        rows = [dict(t, plant=u["plant"], spot=u["spot"], trained_spot=u["trained_spot"])
                for f in files for u in [json.loads(f.read_text())] for t in u["trials"]]
        if rows:
            out[name] = rows
    return out


def rate(rows, key="success"):
    k, n = sum(r[key] for r in rows), len(rows)
    return f"{k}/{n} ({k / n:.0%})" if n else "-"


def main():
    losses()
    data = load()
    if not data:
        raise SystemExit(f"no results in {RESULTS}: ./run eval first")

    # --- table
    head = "| model | demos/pair | success, trained spots | success, new spots | success, all | watered | placed | lifted | droplets in plant | wrong plant |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, rows in data.items():
        k = runs.MODELS[name.split("@")[0]]["k"]
        wrong = sum(r["in_wrong"] > r["in_target"] for r in rows)
        lines.append(f"| {name} | {k} | {rate([r for r in rows if r['trained_spot']])} "
                     f"| {rate([r for r in rows if not r['trained_spot']])} | {rate(rows)} | {rate(rows, 'watered')} "
                     f"| {rate(rows, 'placed')} | {rate(rows, 'lifted')} | {np.mean([r['in_target'] for r in rows]):.1f}/{DROPLET['count']} "
                     f"| {wrong}/{len(rows)} |")
    (RESULTS / "summary.md").write_text("# SmolVLA on the watering scene\n\n"
        f"Success = at least half of the {DROPLET['count']} droplets in the commanded plant and the mug put back upright within 3 cm of its "
        "start. Trained spots 1-4, new spots 5-8. Wrong plant = more droplets in another "
        "plant than in the commanded one.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- success vs k
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, trained) in zip(axes, [("Trained start spots (1-4)", True), ("New start spots (5-8)", False)]):
        for start in COLORS:
            pts = []
            for name, rows in data.items():
                model = runs.MODELS.get(name, {})
                if "@" in name or "data" in model:
                    continue  # checkpoints and other datasets: table only
                if model["start"] == start:
                    sub = [r for r in rows if r["trained_spot"] == trained]
                    pts.append((model["k"], np.mean([r["success"] for r in sub])))
            if not pts:
                continue
            ks, p = zip(*sorted(pts))
            ax.plot(ks, p, "-o", color=COLORS[start], lw=2, ms=6, label=f"from {start}")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(sorted({m["k"] for m in runs.MODELS.values()}))
        ax.set_xlabel("phone demos per plant x spot")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e5e5e5")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("success rate")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "success_vs_k.png", dpi=150)

    # --- droplet destinations
    names = list(data)
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 3.6))
    bottom = np.zeros(len(names))
    for key, label in [("in_target", "commanded plant"), ("in_wrong", "wrong plant"), ("spilled", "spilled"),
                       ("in_mug", "still in mug")]:
        vals = np.array([np.mean([r[key] for r in data[n]]) for n in names])
        ax.bar(names, vals, bottom=bottom, color=DROPLET_COLORS[key], label=label, edgecolor="white", linewidth=2)
        bottom += vals
    ax.set_ylabel(f"droplets (mean of {DROPLET['count']})")
    ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "droplets.png", dpi=150)
    print(f"\n-> {RESULTS}/summary.md, success_vs_k.png, droplets.png")


if __name__ == "__main__":
    main()
