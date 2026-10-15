"""Every evaluated model side by side: results/summary.md, success_vs_k.png, droplets.png, losses.png, tolerance.png,
spot_map.png. See README.md.

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
from dataset import episodes
from dataset.retarget.robot_replay import PLACED_CM, PLACED_DEG
from sim.watering.assets import DROPLET
from train import LOG_EVERY

RESULTS = Path(__file__).parent / "results"
COLORS = {runs.LIBERO: "#2a6fdb", runs.BASE: "#e07b20"}  # one hue per start checkpoint (blue / orange: colorblind-safe)
DROPLET_COLORS = {"in_target": "#2a6fdb", "in_wrong": "#d6453d", "spilled": "#9aa0a6", "in_mug": "#e0b84a"}
TRIALS = 5  # trials per plant x spot counted (the first evals ran 7)
TOLERANCE_CM = np.arange(1, 8)  # put-back distances tolerance.png scores success at
TOLERANCE_MODELS = ["libero-4s-dart-full-xxl-sd-rel-b64", "libero-4s-dart-xl-sd-rel", "libero-4s-dart-xl-sd-rel@15000",
                    "libero-4s-dart-xl-sd-gp2", "libero-4s-dart-xl-sd-memiso", "libero-4s-dart-xl-sd-mem",
                    "libero-4s-dart-xl-sd"]  # tolerance.png: the memory models and the same recipe without memory
MAP_MODELS = ["libero-4s-dart-full-xxl-sd-rel-b64", "libero-4s-dart-xl-sd-rel", "libero-4s-dart-xl-sd-mem",
              "libero-4s-dart-xl-sd"]  # spot_map.png: one row each
SHIFT_CM = 3.0  # training mugs were shifted up to this per axis around each recorded start (robot_replay.SHIFT_CM)


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
        ax.plot(np.arange(1, len(loss) + 1) * LOG_EVERY, loss, color=COLORS[runs.pretrained(name)], ls=(0, (3, 1, 1, 1, 1, 1)) if "batch" in m else "-." if "data" in m else {1: ":", 2: "--", 4: "-"}[m["k"]], lw=1.5, label=name)
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
        rows = [dict(t, plant=u["plant"], spot=u["spot"], trained_spot=u["trained_spot"], trial=i, max_steps=u["max_steps"],
                     spot_cm=u["spot_cm"])
                for f in files for u in [json.loads(f.read_text())] for i, t in enumerate(u["trials"])]
        if rows:
            out[name] = rows
    return out


def rate(rows, key="success"):
    k, n = sum(r[key] for r in rows), len(rows)
    return f"{k}/{n} ({k / n:.0%})" if n else "-"


def success_within(r, cm):
    """Success with the mug put back within cm (instead of PLACED_CM); the saved success up to PLACED_CM (mug_off_cm
    is rounded to 0.1 cm). Released is not saved: a trial that ended before max_steps did (it ends once the mug
    rests released), or the mug was placed."""
    within = r["mug_off_cm"] <= cm
    released = r["placed"] or r["steps"] < r["max_steps"]
    return (r["success"] and within) or (cm > PLACED_CM and r["watered"] and released and within
                                         and r["mug_tilt_deg"] <= PLACED_DEG)


def tolerance(data):
    """Success of TOLERANCE_MODELS as the put-back distance allowed grows -> results/tolerance.png."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (title, trained) in zip(axes, [("Trained start spots (1-4)", True), ("New start spots (5-8)", False)]):
        for name, color in zip(TOLERANCE_MODELS, plt.cm.tab10.colors):
            rows = [r for r in data.get(name, []) if r["trained_spot"] == trained and r["trial"] < TRIALS]
            if rows:
                ax.plot(TOLERANCE_CM, [np.mean([success_within(r, c) for r in rows]) for c in TOLERANCE_CM], "-o",
                        color=color, lw=1.8, ms=4, label=name.replace("libero-4s-", ""))
        ax.axvline(PLACED_CM, color="#9aa0a6", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("put-back distance allowed (cm)")
        ax.grid(axis="y", color="#e5e5e5")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("success rate")
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "tolerance.png", dpi=150)


def spot_map(data):
    """Top view of the table for MAP_MODELS, one row each: every start spot coloured by its grasp, watering and success rate
    over the 3 plant commands x TRIALS -> results/spot_map.png. Recorded starts sit on the +-SHIFT_CM area the
    training data covered; the other spots are new."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle, FancyBboxPatch, Patch
    from matplotlib.lines import Line2D

    cmap = LinearSegmentedColormap.from_list("rate", ["#f3efe6", "#9ccfc4", "#2f8f8a", "#0d3b4c"])
    ink, mute, table, spread = "#1d2530", "#7b8494", "#faf8f3", "#e6e0d3"
    pots = episodes.SETUP["pots"]
    names = [m for m in MAP_MODELS if m in data]
    cols = [("lifted", "Grasp", "mug lifted"), ("watered", "Water", "at least half the droplets in the commanded plant"),
            ("success", "Success", "watered and put back within 3 cm")]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, axes = plt.subplots(len(names), len(cols), figsize=(15, 2.75 * len(names) + 1.2), squeeze=False,
                                 gridspec_kw=dict(wspace=0.04, hspace=0.12))
        for i, (row, name) in enumerate(zip(axes, names)):
            rows = [r for r in data[name] if r["trial"] < TRIALS]
            tr, nw = (sum(r["success"] for r in rows if r["trained_spot"] == t) for t in (True, False))
            row[0].text(-0.03, 0.5, name.replace("libero-4s-", ""), transform=row[0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=9.5, weight="bold", color=ink)
            row[0].text(-0.08, 0.5, f"success {tr} / {nw} of 60", transform=row[0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=8, color=mute)
            for ax, (key, title, sub) in zip(row, cols):
                ax.add_patch(FancyBboxPatch((-40, 11), 68, 42, boxstyle="round,pad=0,rounding_size=3", fc=table,
                                            ec="#d9d3c5", lw=0.8))
                for plant in ("left", "middle", "right"):
                    x, y = pots[plant]
                    ax.add_patch(Circle((x, y), pots["diameter"][plant] / 2, fc="#e4eedc", ec="#93b383", lw=0.9))
                    ax.text(x, y, plant.upper(), ha="center", va="center", fontsize=6.5, color="#5d7a50",
                            weight="bold")
                for spot in sorted({r["spot"] for r in rows}):
                    sub_rows = [r for r in rows if r["spot"] == spot]
                    (x, y), trained = sub_rows[0]["spot_cm"], sub_rows[0]["trained_spot"]
                    k, n = sum(r[key] for r in sub_rows), len(sub_rows)
                    if trained:
                        ax.add_patch(FancyBboxPatch((x - SHIFT_CM, y - SHIFT_CM), 2 * SHIFT_CM, 2 * SHIFT_CM,
                                                    boxstyle="round,pad=0,rounding_size=1.2", fc=spread, ec="none"))
                    ax.add_patch(Circle((x, y), 2.3, fc=cmap(k / n), ec=ink if trained else mute, lw=1.1,
                                        ls="-" if trained else (0, (2, 1.5))))
                    ax.text(x, y, k, ha="center", va="center", fontsize=7.5, weight="bold",
                            color="white" if k / n > 0.55 else ink)
                ax.set_xlim(-41, 29)
                ax.set_ylim(10, 54)
                ax.set_aspect("equal")
                ax.axis("off")
                if i == 0:
                    ax.set_title(f"{title}\n", fontsize=11, weight="bold", color=ink, loc="left")
                    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=mute)
        cax = fig.add_axes([0.62, 0.035, 0.25, 0.012])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_ticks([0, 0.5, 1], labels=["0", "50%", f"all {TRIALS * 3}"])
        cb.outline.set_visible(False)
        cax.tick_params(length=0, labelsize=7.5, colors=mute)
        cax.set_title("trials per spot (3 plants x 5)", fontsize=7.5, color=mute, loc="left")
        fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=ink, ms=9, label="recorded start"),
                            Patch(fc=spread, label=f"training spread (±{SHIFT_CM:g} cm)"),
                            Line2D([], [], marker="o", ls="", mfc="white", mec=mute, ms=9, label="new spot",
                                   markeredgewidth=1, fillstyle="full"),
                            Patch(fc="#e4eedc", ec="#93b383", label="plant")],
                   loc="lower left", bbox_to_anchor=(0.12, 0.012), ncol=4, frameon=False, fontsize=7.5,
                   labelcolor=mute, handletextpad=0.4, columnspacing=1.2)
        fig.savefig(RESULTS / "spot_map.png", dpi=200, bbox_inches="tight", facecolor="white")


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
    tolerance(data)
    spot_map(data)
    print(f"\n-> {RESULTS}/summary.md, success_vs_k.png, droplets.png, tolerance.png, spot_map.png")


if __name__ == "__main__":
    main()
