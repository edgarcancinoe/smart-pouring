"""Outcome judge, step 3: learn which of SmolVLA's K chunks is better, from the cameras, without the simulator.

Per replan: both cameras through the policy's own frozen vision encoder (SmolVLA's SigLIP + connector, mean over its
image tokens, cut to their IMAGE_DIMS main directions), the state and the plant; per chunk: its first LOOKAHEAD actions minus the K's mean. An MLP predicts the chunk's relative
score: its score (judge_check.score, by phase) minus the mean of the K at that replan. Choosing only needs the ranking
among the K, so the replan's own level (how well the pour is going) is taken out of the target. Every phase and every
replan is used: where the K are equal the target is 0, no preference; decisive replans are weighted up to
DECISIVE_SHARE of the loss.

FOLDS-fold cross-validation by trial: every replan is judged by a model that never saw its trial. Does the judge's pick
score like the best of K or like a random one, overall and per phase? Then a final judge on all trials
-> results/judge/<model>/judge.pt (weights + normalization), judge.json, judge.png.

Usage: ./run judge-train
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import evaluate as ev
from dataset.export_lerobot import CAMERAS
from judge_check import DECISIVE, MODEL, OUT, PHASES, load, score

LOOKAHEAD = 20          # actions of each chunk the judge sees (cheat.LOOKAHEAD)
FOLDS = 5               # cross-validation folds, split by trial
DECISIVE_SHARE = 0.5    # of the loss from decisive replans (~20% of replans; the rest teach no preference)
SEED = 0
HIDDEN = 256
DROPOUT = 0.3            # strong: a few hundred decisive scenes (fake-data check: less overfit, more of the gap closed)
EPOCHS = 40
BATCH = 512             # (replan, chunk) pairs per step
LR = 1e-3
WEIGHT_DECAY = 1e-2
FEATURE_BATCH = 32      # replans per pass through the vision encoder
IMAGE_DIMS = 64         # image features kept (PCA): ~2k per replan for a few hundred decisive scenes memorizes the scenes


def features(d):
    """Both cameras of every replan through the policy's frozen vision encoder, mean over image tokens
    -> (R, cameras x width), in load()'s order; cached per recorded trial in features/."""
    cache = OUT / "features"
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for i, f in enumerate(d["files"]):
        path = cache / f"{f.parent.parent.parent.name}-{f.stem}.npy"
        if not path.exists():
            policy, pre, _ = ev.load_model(MODEL)
            z, emb = np.load(f), []
            for a in range(0, len(z["pick"]), FEATURE_BATCH):
                batch = {cam: torch.from_numpy(z[cam][a:a + FEATURE_BATCH]).permute(0, 3, 1, 2).float().div(255) for cam in CAMERAS}
                batch["observation.state"] = torch.from_numpy(z["state"][a:a + FEATURE_BATCH]).float()
                batch["task"] = [f"Water the {z['plant']} plant"] * len(batch["observation.state"])
                with torch.inference_mode():
                    images, _ = policy.prepare_images(pre(batch))
                    emb.append(torch.cat([policy.model.vlm_with_expert.embed_image(im).float().mean(1) for im in images], 1).cpu().numpy())
            np.save(path, np.concatenate(emb))
            print(f"features [{i + 1}/{len(d['files'])}] {f.stem}", flush=True)
        out.append(np.load(path)[d["file_row"][d["trial"] == i]])
    return np.concatenate(out)


class Judge(nn.Module):
    """(replan context, chunk) -> the chunk's relative score, normalized."""

    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))

    def forward(self, ctx, chunk):
        return self.net(torch.cat([ctx, chunk.flatten(1)], 1))[:, 0]


def inputs(d, feats):
    """ctx (R, IMAGE_DIMS image directions + state + plant one-hot), chunks (R, K, LOOKAHEAD, 7) minus the mean of the K:
    the K are near alike, raw their difference is lost (probe: pour start's gap closed 0.05 -> 0.34).
    The directions are fit on every replan's images (no labels involved), so all folds share them."""
    feats = feats - feats.mean(0)
    feats = feats @ np.linalg.svd(feats, full_matrices=False)[2][:IMAGE_DIMS].T
    plant = np.stack([d["plant"] == p for p in ev.PLANTS], 1).astype(np.float32)
    chunks = d["chunks"][:, :, :LOOKAHEAD].astype(np.float32)
    return np.concatenate([feats, d["state"], plant], 1).astype(np.float32), chunks - chunks.mean(1, keepdims=True)


def predict(model, norm, ctx, chunks):
    """The judge's relative score per (replan, chunk): (R, K)."""
    R, K = chunks.shape[:2]
    model.eval()
    with torch.inference_mode():
        y = model((torch.from_numpy(ctx) - norm["ctx"][0]).div(norm["ctx"][1]).repeat_interleave(K, 0),
                  (torch.from_numpy(chunks) - norm["chunk"][0]).div(norm["chunk"][1]).flatten(0, 1))
    return (y * norm["y"]).view(R, K).numpy()


def train(ctx, chunks, y, w, tr, te=None):
    """Fit a judge on replans tr, each weighted by w (R,), and the loss on te per epoch. Returns model, normalization,
    loss curves."""
    torch.manual_seed(SEED)
    stat = lambda x: (torch.from_numpy(x.mean(0)), torch.from_numpy(x.std(0) + 1e-6))
    norm = dict(ctx=stat(ctx[tr]), chunk=stat(chunks[tr].reshape(-1, *chunks.shape[2:])), y=float(y[tr].std() + 1e-6))
    K = chunks.shape[1]
    tensors = lambda idx: ((torch.from_numpy(ctx[np.repeat(idx, K)]) - norm["ctx"][0]) / norm["ctx"][1],
                           (torch.from_numpy(chunks[idx].reshape(-1, *chunks.shape[2:])) - norm["chunk"][0]) / norm["chunk"][1],
                           torch.from_numpy(y[idx].reshape(-1).astype(np.float32)) / norm["y"],
                           torch.from_numpy(np.repeat(w[idx], K).astype(np.float32)))
    train_set, test_set = tensors(tr), tensors(te) if te is not None else None
    model = Judge(ctx.shape[1] + LOOKAHEAD * chunks.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    curves = dict(train=[], test=[])
    for epoch in range(EPOCHS):
        model.train()
        order, losses = torch.randperm(len(train_set[0])), []
        for b in range(0, len(order), BATCH):
            c, a, t, wt = (x[order[b:b + BATCH]] for x in train_set)
            loss = (wt * (model(c, a) - t) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        curves["train"].append(float(np.mean(losses)))
        if test_set:
            model.eval()
            with torch.inference_mode():
                curves["test"].append(nn.functional.mse_loss(model(*test_set[:2]), test_set[2]).item())
    return model, norm, curves


def evaluate(true, pred, m):
    """On replans m: mean true score of the judge's pick, the best of K, a random one (mean of K); share of that gap
    the judge closes (0: random, 1: best); how often it picks the best."""
    if not m.any():
        return dict(replans=0)
    t, p = true[m], pred[m]
    best, rand, judge = t.max(1).mean(), t.mean(1).mean(), t[np.arange(len(t)), p.argmax(1)].mean()
    return dict(replans=int(m.sum()), best=round(float(best), 2), judge=round(float(judge), 2), random=round(float(rand), 2),
                gap_closed=round(float((judge - rand) / (best - rand)), 3) if best > rand else None,
                picks_best=round(float((p.argmax(1) == t.argmax(1)).mean()), 3))


def plot(curves, res, rel, pred, decisive, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    ax = axes[0]
    for k, c in enumerate(curves):
        ax.plot(c["train"], color="#1f6f99", alpha=0.6, label="train trials" if k == 0 else None)
        ax.plot(c["test"], color="#a8442f", alpha=0.6, label="held-out trials" if k == 0 else None)
    ax.set(title=f"Relative-score loss per fold (normalized MSE, {len(curves)} folds)", xlabel="epoch")
    ax.legend()
    ax = axes[1]
    subsets = ["all", "decisive"] + [f"{p} · decisive" for p in PHASES]
    gaps = [res[n].get("gap_closed") or 0 for n in subsets]
    bars = ax.bar(range(len(subsets)), gaps, color=["#5b6d75", "#1f6f99"] + ["#7da7c2"] * len(PHASES))
    ax.bar_label(bars, [f"{res[n]['replans']}" for n in subsets], fontsize=8)
    ax.axhline(1, color="#2f7a52", ls="--", lw=1)
    ax.set(xticks=range(len(subsets)), title="Held-out: share of the random → best gap the judge closes",
           ylabel="0 = random pick, 1 = best of K")
    ax.set_xticklabels(subsets, rotation=25, ha="right")
    ax = axes[2]
    ax.scatter(rel[decisive].ravel(), pred[decisive].ravel(), s=3, alpha=0.3, color="#1f6f99")
    lim = [rel[decisive].min(), rel[decisive].max()] if decisive.any() else [-1, 1]
    ax.plot(lim, lim, color="#5b6d75", lw=1)
    ax.set(title="Held-out decisive replans: judged vs true", xlabel="true relative score (sim)", ylabel="judge")
    fig.savefig(path, dpi=110)


def main():
    d = load(images=False)
    ctx, chunks = inputs(d, features(d))
    s = score(d)
    rel = (s - s.mean(1, keepdims=True)).astype(np.float32)
    decisive = s.max(1) - s.min(1) >= DECISIVE
    w = len(s) * np.where(decisive, DECISIVE_SHARE / max(decisive.sum(), 1), (1 - DECISIVE_SHARE) / max((~decisive).sum(), 1))  # mean 1

    trials = np.random.default_rng(SEED).permutation(d["trial"].max() + 1)
    pred, curves = np.zeros_like(rel), []
    for k, fold in enumerate(np.array_split(trials, FOLDS)):
        te, tr = np.flatnonzero(np.isin(d["trial"], fold)), np.flatnonzero(~np.isin(d["trial"], fold))
        model, norm, c = train(ctx, chunks, rel, w, tr, te)
        pred[te] = predict(model, norm, ctx[te], chunks[te])
        curves.append(c)
        print(f"fold {k + 1}/{FOLDS}: train {c['train'][-1]:.3f} · held-out {c['test'][-1]:.3f} · "
              f"gap closed (decisive) {evaluate(s[te], pred[te], decisive[te]).get('gap_closed')}", flush=True)

    res = dict(all=evaluate(s, pred, np.ones(len(s), bool)), decisive=evaluate(s, pred, decisive),
               **{p: evaluate(s, pred, d["phase"] == p) for p in PHASES},
               **{f"{p} · decisive": evaluate(s, pred, (d["phase"] == p) & decisive) for p in PHASES})
    model, norm, _ = train(ctx, chunks, rel, w, np.arange(len(s)))  # the judge to use: every trial
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), norm=norm, n_in=ctx.shape[1] + LOOKAHEAD * chunks.shape[-1],
                    model=MODEL, lookahead=LOOKAHEAD), OUT / "judge.pt")
    (OUT / "judge.json").write_text(json.dumps(res, indent=2))
    plot(curves, res, rel, pred, decisive, OUT / "judge.png")
    print(json.dumps(res, indent=2))
    print(f"-> {OUT / 'judge.png'}, {OUT / 'judge.pt'}")


if __name__ == "__main__":
    main()
