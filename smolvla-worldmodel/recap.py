"""RECAP (Physical Intelligence's pi*0.6 recipe) on the judge recordings: offline RL without a judge at run time.

At every recorded pour replan the sim scored K options from the same state; an option's score minus the K's mean is its
advantage. Options clearly better than their group (>= MARGIN) are tagged "Advantage: positive" in the prompt, clearly
worse ones "Advantage: negative" (in between: no clear lesson, left out), and a share UNTAGGED is also shown with the
plain prompt so the untagged policy stays what it was. A copy of MODEL is fine-tuned on them (flow-matching loss on the
options' recorded actions) and pushed as RECAP_MODEL. At run time the pour benchmark adds TAG_GOOD to the prompt once
the mug is carried: the policy is asked for its better options.
shortcut: options span only the recorded LOOKAHEAD actions (the rest of the chunk is padding, out of the loss), and
the training data is these options alone (no demos mixed in); upgrade if the untagged grasp or put-back degrades

Usage: ./run recap                fine-tune -> Hub (RECAP_MODEL)
       ./run pour-recap [--jobs 9]  the pour-only benchmark of RECAP_MODEL, tagged positive once carried
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi

import eval_relative  # noqa: F401 (the policy's state with its relative grasp slots)
import evaluate as ev
import grasp_position as gp
import runs
import state_dropout
from dataset.export_lerobot import CAMERAS
from judge_check import MODEL, load, score
from judge_train import grasp_memory

RECAP_MODEL = f"{MODEL}-recap"  # runs.MODELS
TAG_GOOD, TAG_BAD = " Advantage: positive", " Advantage: negative"  # appended to "Water the <plant> plant"
USE_PHASES = ("pour start", "tip")  # where the tag is used at run time (once carried)
MARGIN = 5.0            # score points (half a drop) from the K's mean: tagged; closer: left out
UNTAGGED = 0.3          # share of samples shown with the plain prompt
STEPS = 2_000           # optimizer steps
BATCH = 32
LR = 2e-5               # dagger1's own fine-tune ended near 0: small, so the copy stays close
DROPOUT = 0.3           # state dropout, as dagger1 trained (runs.MODELS)
SEED = 0


def samples():
    """Tagged options -> frames {camera: (n, H, W, 3) uint8}, state (n, 8), grasp memory (n, 3), plant (n,),
    their replan's row in those arrays, the option's actions (m, LOOKAHEAD, 7) and its tag (m,): True good, False bad."""
    d = load(images=False)
    s = score(d)
    adv = s - s.mean(1, keepdims=True)
    keep = np.isin(d["phase"], USE_PHASES) & (np.abs(adv) >= MARGIN).any(1)
    frames, state, grasp, plant, rows, chunks, good = {c: [] for c in CAMERAS}, [], [], [], [], [], []
    for i, f in enumerate(d["files"]):
        mine = np.flatnonzero(keep & (d["trial"] == i))
        if not len(mine):
            continue
        z = np.load(f)
        g = grasp_memory(f, z)
        for r in mine:
            fr = d["file_row"][r]
            for c in CAMERAS:
                frames[c].append(z[c][fr])
            state.append(z["state"][fr]), grasp.append(g[fr]), plant.append(d["plant"][r])
            for k in np.flatnonzero(np.abs(adv[r]) >= MARGIN):
                rows.append(len(state) - 1), chunks.append(d["chunks"][r, k]), good.append(adv[r, k] > 0)
    print(f"{len(state)} replans, {len(rows)} tagged options ({sum(good)} good, {len(good) - sum(good)} bad)", flush=True)
    return ({c: np.stack(v) for c, v in frames.items()}, np.stack(state), np.stack(grasp), np.array(plant),
            np.array(rows), np.stack(chunks).astype(np.float32), np.array(good))


def train():
    """Fine-tune a copy of MODEL on the tagged options; push it as RECAP_MODEL."""
    policy, pre, post = ev.load_model(MODEL)  # first: it loads the state stats grasp_memory() rebuilds with
    frames, state, grasp, plant, rows, chunks, good = samples()
    cfg, dev = policy.config, policy.config.device
    state_dropout.P = DROPOUT
    gp.TABLE = torch.from_numpy(grasp)  # training reads the grasp slots by the batch's "index": here, the replan
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    params = [p for p in policy.parameters() if p.requires_grad]  # SmolVLA's config freezes the vision encoder
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    policy.train()
    n = chunks.shape[1]
    for step in range(1, STEPS + 1):
        pick = rng.integers(len(rows), size=BATCH)
        r = rows[pick]
        plain = rng.random(BATCH) < UNTAGGED
        batch = {c: torch.from_numpy(frames[c][r]).permute(0, 3, 1, 2).float().div(255) for c in CAMERAS}
        batch["observation.state"] = torch.from_numpy(state[r])
        batch["task"] = [f"Water the {p} plant" + ("" if u else TAG_GOOD if g else TAG_BAD)
                         for p, u, g in zip(plant[r], plain, good[pick])]
        batch["action"] = torch.from_numpy(chunks[pick])
        b = pre(batch)
        act = torch.zeros(BATCH, cfg.chunk_size, b["action"].shape[-1], device=b["action"].device)  # normalized: 0 = mean action
        act[:, :n] = b["action"]
        b["action"] = act
        b["action_is_pad"] = (torch.arange(cfg.chunk_size, device=act.device) >= n).expand(BATCH, -1)
        b["index"] = torch.from_numpy(r)
        loss, _ = policy.forward(b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step}/{STEPS}: loss {loss.item():.4f}", flush=True)
    policy.eval()
    repo = runs.MODELS[RECAP_MODEL]["repo"]
    with tempfile.TemporaryDirectory() as tmp:
        policy.save_pretrained(tmp)
        pre.save_pretrained(tmp)
        post.save_pretrained(tmp)
        Path(tmp, "recap.json").write_text(json.dumps(dict(start=MODEL, margin=MARGIN, untagged=UNTAGGED, steps=STEPS,
                                                           lr=LR, options=len(rows), good=int(good.sum())), indent=1))
        HfApi().create_repo(repo, exist_ok=True)
        HfApi().upload_folder(folder_path=tmp, repo_id=repo)
    print(f"-> https://huggingface.co/{repo}")


if __name__ == "__main__":
    train()
