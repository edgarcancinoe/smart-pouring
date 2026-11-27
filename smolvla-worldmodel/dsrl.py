"""DSRL (Wagenmaker et al. 2025, "Steering Your Diffusion Policy with Latent Space RL"): online RL that leaves
SmolVLA frozen and learns only the noise its chunk is denoised from. A small SAC actor maps what the policy knows (robot
state, grasp memory, plant) to that noise; SmolVLA turns it into the chunk as usual. The actor steers only N_Z
directions of the noise (directions()): the rest stays plain random noise, so a random actor is the plain policy and the
critic learns from N_Z numbers instead of all NOISE_STEPS x 7. Only the pour is learned: the
plain policy carries the mug (pour.py's trials), the actor picks the noise at every replan from then until the mug is
empty. Reward per replan: drops newly in the target pot minus drops newly in a wrong pot or spilled.
The critic also sees the sim (mug mouth vs pot, tilt, drops: cheat.state), the actor never does (asymmetric).

Training runs in rounds: EPISODES pour trials with the current actor (random mug starts, as judge_full.py: away from the
benchmark's spots 5-8), then SAC updates on every transition so far, then the next round. Resumable: a rerun skips the
finished trials and continues from learner.pt. A fresh learner beside recorded pours (an earlier learner's, moved aside)
first learns from all of them at once, then pours from the next round on (catch_up).
-> results/dsrl/<model>/: <unit>.json + data/<unit>.npz per trial, actor.pt, learner.pt, dsrl.json + dsrl.png (per
round); after every round all but the videos -> the Hub's worldmodel/dsrl/<model>/, and the round's numbers -> W&B
(project WANDB_PROJECT; needs `uv run wandb login` once per machine)
shortcut: the actor sees no camera (the pots don't move, the plant says which); add SigLIP features if it plateaus
shortcut: the directions are hand-picked (whole-chunk lean per action slot, tilt ramps); learn them if this plateaus
shortcut: a trial that hangs stalls its round (as in pour-plain): stop and rerun, it resumes

Held-out check while training: from round EVAL_FROM, every EVAL_EVERY rounds and after the last, the actor pours
EVAL_UNITS (fixed random starts no training pour uses) with its mean push; plain pours them once, at the first check.
Both go to W&B (eval_*, plain_*) beside the training pours, so the curve compares like with like.

Usage: ./run dsrl [--jobs 9]        train -> results/dsrl/<model>/
       DSRL_RUN=clean ./run dsrl    a separate run: results/dsrl-clean/<model>/, the Hub's worldmodel/dsrl-clean/
       ./run pour-dsrl [--jobs 9]   the pour benchmark at spots 5-8 with the trained actor (its mean noise) -> results/pour-dsrl/
       SAMPLE=1 ./run pour-dsrl ...  the same with sampled pushes (as in training), not the actor's mean
       ./run starts-plain | starts-dsrl [--jobs 18]  training-style random mug starts with new seeds, nothing trained:
                                    plain vs the actor (SAMPLE=1: sampled pushes) -> results/starts-<arm>[-sampled]/
       ./run full-dsrl [--jobs 18]  the whole task (grasp, the actor's pour, put back) as the README headline: spots 1-8,
                                    5 trials per plant x spot, 3 seed sets, 90 s -> results/full-dsrl/[s<seed>/]<model>/
"""
import json
import os
import time

os.environ.setdefault("ARM", "dsrl")  # before cheat reads it (also in the spawned workers)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pour  # noqa: F401 (patches evaluate and cheat: pour-only trials); this file patches on top
import cheat
import evaluate as ev
import grasp_position as gp
import runs
from dataset.export_lerobot import CAMERAS
from judge_check import HUB

ROOT = cheat.RESULTS.parent  # results/ (pour.py, old or new, points cheat.RESULTS at results/pour-<arm>)
TRAIN = os.environ.get("DSRL") not in ("eval", "full", "starts")  # pour-dsrl: DSRL=eval, full-dsrl: full, starts-*: starts
FULL = os.environ.get("DSRL") == "full"
STARTS = os.environ.get("DSRL") == "starts"
SAMPLE = os.environ.get("SAMPLE") == "1"  # outside training: sampled pushes as in training (default: the actor's mean)
STARTS_SEED = 70_000    # starts-*' trial seeds: none of training's (40_000 + 10_000 x plant + t < 65_000)
STARTS_TRIALS = 24      # starts-* trials per plant
FULL_SEEDS = (0, 10_000, 20_000)  # evaluate.SEED of the headline's 3 seed sets (README: 3 seeds x 120 trials)


def full_results(seed):
    """full-dsrl's folder for a seed set, laid out as evaluate's (seed 0: no subfolder)."""
    return ROOT / "full-dsrl" / ("" if seed == 0 else f"s{seed}")


if FULL:  # whole episodes: pour.py's end-of-pour rules off; its carried() stays the actor's gate, as in training
    ev.act, ev.reset, cheat.trial = cheat.act, pour._reset, pour._trial
    ev.HIT = "success"
    ev.UNITS = [f"{p}-{s}-{t}" for p in ev.PLANTS for s in range(1, 9) for t in range(pour.TRIALS)]
    cheat.RESULTS = ev.RESULTS = full_results(ev.SEED)
ROUNDS = 30
EPISODES = 18           # pour trials per round (a third per plant)
NOISE_STEPS = 10        # chunk steps the directions span (= the N_ACTION_STEPS run)
Z_MAX = 2.0             # the actor's push along each direction is squashed to +- this (plain noise: N(0, 1) along each)
LEASH = 1.0             # actor loss + LEASH x |z|^2: stay near the noise SmolVLA was trained on (a drop is worth REWARD_SCALE)
HIDDEN = 256
GAMMA = 0.95            # per replan (0.5 s): about 10 s ahead
UTD = 2                 # gradient steps per new transition (10 let the critic diverge: Q ~400 vs returns ~14)
REWARD_SCALE = 10.0     # reward x this in the critic: else SAC's entropy bonus dwarfs a few drops
WARMUP = 3              # rounds whose updates train only the critic: the actor stays random until the critic has data
BATCH = 256
LR = 3e-4
TAU = 0.005             # target critic update rate
SEED = 40_000           # training trial seeds: no eval, judge or distill seed
WANDB_PROJECT = "smart-pouring"  # live curves at wandb.ai/<you>/smart-pouring, as smolvla-finetune/train.py
AREA = ((-24.0, 14.0), (27.0, 30.0))  # cm: random mug starts, as judge_full.py (importing it would patch evaluate)
KEEP_OUT = 4.0          # cm from spots 5-8
RUN = os.environ.get("DSRL_RUN", "")  # a run's name: its own folders here and on the Hub ("" = the first run's)
EVAL = os.environ.get("DSRL_EVAL", "")  # set by train() for its held-out check: "plain" or the round
EVAL_FROM, EVAL_EVERY = 15, 5  # held-out check from this round, then every EVAL_EVERY rounds and after the last
EVAL_UNITS = [f"{p}-0-{9000 + i}" for i in range(6) for p in ev.PLANTS]  # 18 fixed random starts: t >= 9000, no training round's
OUT = ROOT / f"dsrl{'-' + RUN if RUN else ''}" / cheat.MODEL
ACTOR = OUT / "actor.pt"
HUB_DIR = f"worldmodel/dsrl{'-' + RUN if RUN else ''}/{cheat.MODEL}"  # report() uploads OUT here (rltoken.py: its own)
N_A = len(ev.OPEN)      # action dims (7)
N_W = NOISE_STEPS * N_A  # noise numbers of the steps run
N_OBS = 8 + 3 + len(ev.PLANTS)  # state, grasp memory, plant
N_PRIV = 9              # priv()

def directions():
    """The noise patterns the actor steers, (N_W, N_Z) orthonormal columns over the (NOISE_STEPS, 7) noise grid: per
    action slot the same push on every step (lean the chunk in x, y, z, roll, pitch, yaw, gripper), and per rotation a
    ramp over the steps (less early, more late: when to tilt)."""
    cols = []
    for d in range(N_A):
        v = np.zeros((NOISE_STEPS, N_A))
        v[:, d] = 1
        cols.append(v.ravel())
    for d in (3, 4, 5):  # the rotation slots (action: xyz, rotation, gripper)
        v = np.zeros((NOISE_STEPS, N_A))
        v[:, d] = np.linspace(-1, 1, NOISE_STEPS)
        cols.append(v.ravel())
    b = np.stack(cols, 1)
    return torch.from_numpy(b / np.linalg.norm(b, axis=0)).float()


B = directions()
N_Z = B.shape[1]
assert torch.allclose(B.T @ B, torch.eye(N_Z), atol=1e-6), "directions must be orthonormal"

_w = {}  # this process: the actor (and its file's mtime), the running trial's unit and replans


def obs(frame, task):
    """What the actor sees: SmolVLA's state (8), its grasp memory (3), the plant (one-hot)."""
    plant = task.split()[2]  # "Water the <plant> plant"
    return np.r_[frame["observation.state"], gp.CURRENT.numpy(), [plant == p for p in ev.PLANTS]].astype(np.float32)


def priv(s):
    """What only the critic sees, from cheat.state(): mug mouth vs pot (cm), tilt, drops in mug / over pot / elsewhere,
    hand-to-mug and mug height (cm)."""
    return np.r_[s["mouth_cm"], s["tilt"], s["in_mug"], s["over"], s["elsewhere"], s["hand_cm"], s["up_cm"]].astype(np.float32)


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_out))


class Actor(nn.Module):
    """obs -> push z along each direction (N_Z,), tanh-squashed Gaussian; explore=False: its mean."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_OBS))
        self.register_buffer("std", torch.ones(N_OBS))
        self.net = mlp(N_OBS, 2 * N_Z)
        nn.init.zeros_(self.net[-1].weight), nn.init.zeros_(self.net[-1].bias)  # fresh: z ~ tanh(N(0, 1)) x Z_MAX for every state, about the plain policy

    def forward(self, o, explore=True):
        mu, log_std = self.net((o - self.mean) / self.std).chunk(2, -1)
        if not explore:
            return torch.tanh(mu) * Z_MAX, None
        log_std = log_std.clamp(-5, 1)
        u = mu + log_std.exp() * torch.randn_like(mu)
        logp = (-0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)  # the tanh's change of density
        return torch.tanh(u) * Z_MAX, logp


class Critic(nn.Module):
    """(obs, priv, z) -> two Q estimates."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_OBS + N_PRIV))
        self.register_buffer("std", torch.ones(N_OBS + N_PRIV))
        self.q1, self.q2 = mlp(N_OBS + N_PRIV + N_Z, 1), mlp(N_OBS + N_PRIV + N_Z, 1)

    def forward(self, o, p, z):
        x = torch.cat([(torch.cat([o, p], -1) - self.mean) / self.std, z / Z_MAX], -1)
        return self.q1(x)[:, 0], self.q2(x)[:, 0]


def actor():
    """This process's actor, reloaded when actor.pt changes (training: after every round). Eval: from the Hub if not here."""
    path = ACTOR
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HUB, f"worldmodel/dsrl/{cheat.MODEL}/actor.pt", repo_type="dataset")
    mtime = os.path.getmtime(path)
    if _w.get("mtime") != mtime:
        a = Actor()
        a.load_state_dict(torch.load(path, weights_only=True))
        _w.update(actor=a.eval(), mtime=mtime)
    return _w["actor"]


def pick(model, frame, task, s):
    """cheat's noise_policy: the chunk SmolVLA denoises from plain random noise with the actor's push along the
    directions (w = B z + the noise's part off the directions); the replan and its noise w are recorded."""
    o = obs(frame, task)
    with torch.no_grad():
        z = actor()(torch.from_numpy(o)[None], explore=(TRAIN and not EVAL) or SAMPLE)[0][0]
    policy, pre, post = model
    cfg = policy.config
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim)  # plain policy's noise everywhere...
    eps = noise[0, :NOISE_STEPS, :N_A].reshape(-1)
    w = B @ z + eps - B @ (B.T @ eps)  # ...but along the directions: the actor's z
    noise[0, :NOISE_STEPS, :N_A] = w.view(NOISE_STEPS, N_A)
    e = cheat._run["env"].env
    st = e.watering_stats()
    _w["steps"].append(dict(obs=o, priv=priv(s), w=w.numpy(), good=st["target"],
                            bad=st["spilled"] + sum(st[p] for p in e.pot_names if p != e.target_pot)))
    batch = {key: torch.from_numpy(frame[key]).permute(2, 0, 1).float().div(255)[None] for key in CAMERAS}
    batch["observation.state"] = torch.from_numpy(frame["observation.state"])[None]
    batch["task"] = [task]
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(pre(batch), noise=noise.to(cfg.device))
    return post(chunk)[0, :ev.N_ACTION_STEPS].cpu().numpy()


def random_start(rng):
    """A mug start (cm) in AREA, at least KEEP_OUT from spots 5-8 (as judge_full.py)."""
    while True:
        xy = tuple(rng.uniform(*AREA))
        if min(np.hypot(*np.subtract(xy, ev.SPOTS[s])) for s in (5, 6, 7, 8)) > KEEP_OUT:
            return xy


def reset(env, spot, rng, jitter=ev.MUG_JITTER):
    if spot == 0:  # evaluate.reset reads the start from SPOTS
        ev.SPOTS[0] = random_start(rng)
    return _reset(env, spot, rng, 0.0 if spot == 0 else jitter)


def trial(*args, **kwargs):
    """pour's trial; training: its replans -> data/<unit>.npz, with the drops at the end."""
    _w["steps"] = []
    r = _trial(*args, **kwargs)
    if TRAIN and not EVAL and _w["steps"]:  # the held-out check's pours never train
        (OUT / "data").mkdir(parents=True, exist_ok=True)
        rows = _w["steps"]
        np.savez(OUT / "data" / f"{_w['unit']}.npz", end_good=r["in_target"], end_bad=r["in_wrong"] + r["spilled"],
                 watered=r["watered"], **{k: np.array([row[k] for row in rows]) for k in rows[0]})
    return r


def run_unit(unit, *args, **kwargs):
    _w["unit"] = unit
    return _run_unit(unit, *args, **kwargs)


def transitions(files):
    """npz files -> tensors obs, priv, z (the recorded noise along the directions), reward, next obs, next priv, done
    (one row per replan)."""
    cols = {k: [] for k in ("o", "p", "z", "r", "o2", "p2", "d")}
    for f in files:
        z = np.load(f)
        good, bad = np.r_[z["good"], z["end_good"]], np.r_[z["bad"], z["end_bad"]]
        n = len(z["w"])
        cols["o"].append(z["obs"]), cols["p"].append(z["priv"]), cols["z"].append(z["w"] @ B.numpy())
        cols["r"].append(np.diff(good) - np.diff(bad))
        cols["o2"].append(np.r_[z["obs"][1:], z["obs"][-1:]]), cols["p2"].append(np.r_[z["priv"][1:], z["priv"][-1:]])
        cols["d"].append(np.arange(n) == n - 1)
    return {k: torch.from_numpy(np.concatenate(v).astype(np.float32)) for k, v in cols.items()}


def new_learner():
    a, c = Actor(), Critic()
    t = Critic()
    t.load_state_dict(c.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True)
    return dict(actor=a, critic=c, target=t, log_alpha=log_alpha, round=0, curve=[],
                opt_a=torch.optim.Adam(a.parameters(), LR), opt_c=torch.optim.Adam(c.parameters(), LR),
                opt_alpha=torch.optim.Adam([log_alpha], LR))


def update(L, data, steps, dev, actor=True):
    """steps SAC gradient steps on data (transitions()); actor=False: the critic only (WARMUP).
    Returns the last batch's critic loss, the critic's Q of the actor's noise, alpha, the actor's mean push |z| and,
    if it trained, its loss."""
    a, c, t = L["actor"].to(dev), L["critic"].to(dev), L["target"].to(dev)
    log_alpha = L["log_alpha"]
    d = {k: v.to(dev) for k, v in data.items()}
    n = len(d["r"])
    for _ in range(steps):
        i = torch.randint(n, (BATCH,), device=dev)
        alpha = log_alpha.exp().item()
        with torch.no_grad():
            z2, logp2 = a(d["o2"][i])
            y = REWARD_SCALE * d["r"][i] + GAMMA * (1 - d["d"][i]) * (torch.min(*t(d["o2"][i], d["p2"][i], z2)) - alpha * logp2)
        q1, q2 = c(d["o"][i], d["p"][i], d["z"][i])
        loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        L["opt_c"].zero_grad()
        loss_c.backward()
        L["opt_c"].step()
        with torch.no_grad():  # before the actor: the target must follow the critic in the warm-up too
            for pt, pc in zip(t.parameters(), c.parameters()):
                pt.lerp_(pc, TAU)
        if not actor:
            continue
        z, logp = a(d["o"][i])
        q = torch.min(*c(d["o"][i], d["p"][i], z))
        loss_a = (alpha * logp - q + LEASH * (z ** 2).sum(-1)).mean()
        L["opt_a"].zero_grad()
        loss_a.backward()
        L["opt_a"].step()
        loss_alpha = -(log_alpha * (logp.detach().cpu() - N_Z).mean())  # target entropy -N_Z
        L["opt_alpha"].zero_grad()
        loss_alpha.backward()
        L["opt_alpha"].step()
    with torch.no_grad():
        z, _ = a(d["o"][i])
        q = torch.min(*c(d["o"][i], d["p"][i], z))
    out = dict(critic_loss=loss_c.item(), q=q.mean().item(), alpha=log_alpha.exp().item(), push=z.abs().mean().item())
    if actor:
        out["actor_loss"] = loss_a.item()
    return out


def save(L):
    """learner.pt (to resume) and actor.pt (what the workers load).
    shortcut: the optimizers' state isn't kept (it lives on the GPU): a resumed run restarts Adam's moments"""
    torch.save({k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in L.items() if not k.startswith("opt")},
               OUT / "learner.pt")
    torch.save({k: v.cpu() for k, v in L["actor"].state_dict().items()}, OUT / "actor.tmp")
    os.replace(OUT / "actor.tmp", ACTOR)  # whole file at once: a worker never reads half of it


def load():
    L = new_learner()
    if (OUT / "learner.pt").exists():
        s = torch.load(OUT / "learner.pt", weights_only=False)
        for k, v in s.items():
            if hasattr(L.get(k), "load_state_dict"):
                L[k].load_state_dict(v)
            elif k == "log_alpha":
                L[k].data.copy_(v)
            else:
                L[k] = v
    return L


def report(curve):
    """dsrl.json and dsrl.png (drops per pour and critic loss per round), then the run so far -> the Hub, so progress
    can be watched from anywhere; a failed upload only warns."""
    import matplotlib.pyplot as plt
    (OUT / "dsrl.json").write_text(json.dumps(curve, indent=1))
    r = [c["round"] for c in curve]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(r, [c["drops"] for c in curve], "o-")
    ax[0].set(xlabel="round", ylabel="drops in the pot per pour (of 10)", title="pour")
    ax[1].plot(r, [c["critic_loss"] for c in curve], "o-")
    ax[1].set(xlabel="round", ylabel="critic loss", yscale="log", title="critic")
    fig.tight_layout()
    fig.savefig(OUT / "dsrl.png", dpi=120)
    plt.close(fig)
    try:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=OUT, repo_id=HUB, repo_type="dataset", path_in_repo=HUB_DIR,
                              allow_patterns=["dsrl.json", "dsrl.png", "actor.pt", "learner.pt", "*.json", "data/*"],
                              commit_message=f"DSRL round {curve[-1]['round']}")
    except Exception as err:  # training goes on; the next round uploads everything again
        print(f"upload to the Hub failed: {err}", flush=True)


def scale_inputs(L, data):
    """Fix the networks' input scaling from data's states (once per learner)."""
    L["actor"].mean.copy_(data["o"].mean(0)), L["actor"].std.copy_(data["o"].std(0).clamp(min=1e-2))
    x = torch.cat([data["o"], data["p"]], -1)
    for c in (L["critic"], L["target"]):
        c.mean.copy_(x.mean(0)), c.std.copy_(x.std(0).clamp(min=1e-2))
    L["scaled"] = True


def catch_up(L, update_fn, transitions_fn, scale_fn, dev):
    """A fresh learner beside pours recorded by an earlier one: learn from all of them at once (critic only, then
    critic and actor, UTD steps per replan each), and skip to the first round with no pours. Training and the logs then
    start with the current actor's own pours. update_fn(L, data, steps, dev, actor?) as update()."""
    old = sorted((OUT / "data").glob("*.npz"))
    if L["round"] or not old:
        return
    data = transitions_fn(old)
    scale_fn(L, data)
    n = len(data["r"])
    update_fn(L, data, UTD * n, dev, False)
    s = update_fn(L, data, UTD * n, dev, True)
    L["round"] = max(int(f.stem.split("-")[-1]) // 100 for f in old) + 1
    for k in ("actor", "critic", "target"):
        L[k].cpu()
    print(f"caught up on {len(old)} recorded pours ({n} replans): critic loss {s['critic_loss']:.3f}, Q {s['q']:.1f}; "
          f"new pours start at round {L['round']}", flush=True)


def held_out(tag, arm):
    """EVAL_UNITS poured by arm ("plain", or "dsrl": the current actor's mean push) -> OUT/eval/<tag>/; returns
    carried, watered (share of carried) and drops (per carried pour). Resumable like any eval (done trials are kept).
    shortcut: the workers learn arm and tag from env vars, so it needs --jobs > 1 (one job runs in this process)"""
    os.environ.update(DSRL_EVAL=str(tag), ARM=arm)
    ev.UNITS, cheat.RESULTS, ev.RESULTS = EVAL_UNITS, OUT / "eval" / str(tag), OUT / "eval" / str(tag)
    try:
        ev.main()
    finally:
        os.environ.pop("DSRL_EVAL"), os.environ.update(ARM="dsrl")
        cheat.RESULTS = ev.RESULTS = OUT.parent
    T = [json.load(open(f))["trials"][0] for f in (OUT / "eval" / str(tag) / cheat.MODEL).glob("*.json")]
    c = [x for x in T if x.get("carried")]
    return dict(carried=len(c), watered=round(sum(x["watered"] for x in c) / max(len(c), 1), 3),
                drops=round(sum(x["in_target"] for x in c) / max(len(c), 1), 2))


def wb_log(rows, **init):
    """rows (dicts with "round") -> the W&B run init names, opened only for this. A run kept open while evaluate.main
    forks its helper process stopped recording metrics: the console kept streaming, the logged rows never arrived."""
    import wandb
    run = wandb.init(project=WANDB_PROJECT, resume="allow", **init)
    for r in rows:
        run.log(r, step=r["round"])
    run.finish()


def train():
    """ROUNDS x (EPISODES pour trials with the current actor, then UTD SAC steps per new replan)."""
    OUT.mkdir(parents=True, exist_ok=True)
    dev = runs.device()
    L = load()
    L["wandb_id"] = L.get("wandb_id") or f"dsrl-{int(time.time())}"  # kept in learner.pt: a resumed run continues the same W&B run
    wb = dict(name=f"dsrl-{RUN or 'dirs' + str(N_Z)}-{cheat.MODEL}", id=L["wandb_id"],
              config=dict(model=cheat.MODEL, rounds=ROUNDS, episodes=EPISODES, noise_steps=NOISE_STEPS,
                          directions=N_Z, z_max=Z_MAX, leash=LEASH, reward_scale=REWARD_SCALE, warmup=WARMUP,
                          gamma=GAMMA, utd=UTD, batch=BATCH, lr=LR, tau=TAU))
    if L["curve"]:
        wb_log(L["curve"], **wb)  # a restart: the rounds so far (W&B drops any it already has)
    catch_up(L, update, transitions, scale_inputs, dev)
    save(L)  # the workers need an actor (fresh: noise about N(0, 1), like the plain policy)
    for rnd in range(L["round"], ROUNDS):
        units = [f"{p}-0-{rnd * 100 + i}" for i in range(EPISODES // len(ev.PLANTS)) for p in ev.PLANTS]
        ev.UNITS = units
        ev.main()
        files = sorted((OUT / "data").glob("*.npz"))
        mine = [f for f in files if f.stem in units]
        if not mine:
            print(f"round {rnd}: no pour in this round (none carried?), next round")
            continue
        data = transitions(files)
        if not (L.get("scaled") or L["curve"]):  # input scaling, fixed from the first round's states
            scale_inputs(L, data)
        new = transitions(mine)
        t0 = time.time()
        stats = update(L, data, UTD * len(new["r"]), dev, actor=rnd >= WARMUP)
        ends = [np.load(f) for f in mine]
        row = dict(round=rnd, episodes=len(mine), replans=len(new["r"]), all_replans=len(data["r"]),
                   drops=round(float(np.mean([z["end_good"] for z in ends])), 2),
                   watered=round(float(np.mean([z["watered"] for z in ends])), 2),
                   reward=round(float(new["r"].sum()) / len(mine), 2), **{k: round(v, 4) for k, v in stats.items()},
                   update_s=round(time.time() - t0))
        L["round"] = rnd + 1
        for k in ("actor", "critic", "target"):
            L[k].cpu()
        save(L)  # first: the held-out check pours with this round's actor
        if rnd >= EVAL_FROM and ((rnd - EVAL_FROM) % EVAL_EVERY == 0 or rnd == ROUNDS - 1):
            L["plain_eval"] = L.get("plain_eval") or held_out("plain", "plain")
            row.update({f"plain_{k}": v for k, v in L["plain_eval"].items()}, **{f"eval_{k}": v for k, v in held_out(rnd, "dsrl").items()})
        L["curve"].append(row)
        save(L)
        report(L["curve"])
        wb_log([row], **wb)
        print(f"\n=== DSRL round {rnd}: {row}\n", flush=True)


_reset, _trial, _run_unit = ev.reset, cheat.trial, ev.run_unit
# at import, so the spawned eval workers get them too
cheat.noise_policy = pick
cheat.trial, ev.run_unit = trial, run_unit
if TRAIN:
    ev.reset = reset
    ev.trial_seed = lambda plant, spot, t: SEED + 10_000 * ev.PLANTS.index(plant) + t
    cheat.RESULTS = ev.RESULTS = OUT / "eval" / EVAL if EVAL else OUT.parent
if STARTS:  # random starts drawn as in training, new seeds; nothing is trained or saved beyond the trial results
    ev.reset = reset
    ev.trial_seed = lambda plant, spot, t: STARTS_SEED + 10_000 * ev.PLANTS.index(plant) + t
    ev.UNITS = [f"{p}-0-{t}" for t in range(STARTS_TRIALS) for p in ev.PLANTS]
    cheat.RESULTS = ev.RESULTS = ROOT / f"starts-{cheat.ARM}{'-sampled' if SAMPLE and cheat.ARM == 'dsrl' else ''}"


def full():
    """The whole task at every spot for each of the headline's seed sets, the actor pouring with its mean noise."""
    for seed in FULL_SEEDS:
        os.environ["SEED"] = str(seed)  # the spawned workers' evaluate reads it: their trial seeds and folder
        ev.SEED = seed
        cheat.RESULTS = ev.RESULTS = full_results(seed)
        ev.main()


if __name__ == "__main__":
    train() if TRAIN else full() if FULL else ev.main()
