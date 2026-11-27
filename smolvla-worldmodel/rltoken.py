"""RL Token (Xu et al. 2026, Physical Intelligence, "RL Token: Bootstrapping Online RL with Vision-Language-Action
Models") on the pour: SmolVLA stays frozen and proposes its chunk; a small actor, trained online, refines it.

1. The token (./run rltoken-token). A small encoder reads SmolVLA's prefix embeddings (cameras, task and state after
   its VLM layers: what SmolVLA "sees") through one learned query token -> z_rl. A decoder must rebuild every embedding
   from z_rl alone (autoregressive, teacher-forced), so z_rl is a compressed summary of SmolVLA's view. Trained on the
   frames of the judge recordings (the policy's own rollouts, every phase), SmolVLA frozen -> token.pt (+ Hub).
2. RL (./run rltoken). Pour trials in rounds, as dsrl.py (same random mug starts and seeds, reward, warm-up, W&B, Hub).
   At each pour replan SmolVLA proposes its chunk ã; the actor sees x = (z_rl, state, grasp memory, plant) and ã and
   outputs the 10 actions to run: ã + a bounded correction (the gripper stays as SmolVLA says). TD3-style: twin
   critics, exploration noise around the actor, actor loss -Q + BETA |a - ã|^2 (stay near SmolVLA's proposal). The
   critic also sees the sim (cheat.state), as in dsrl.py; the paper's critic sees only x.
-> results/rltoken/<model>/: token.pt, <unit>.json + data/<unit>.npz per trial, actor.pt, learner.pt, dsrl.json + .png
   (per round, dsrl.py's report) -> the Hub's worldmodel/rltoken/<model>/ and W&B
shortcut: the actor is a residual (ã + correction); the paper's actor outputs the chunk with ã as an input and target
shortcut: no reference-chunk dropout and no target actor (the paper drops ã for part of each batch; TD3 keeps one)
shortcut: the token's prefix is recomputed at each replan beside SmolVLA's own (one more VLM pass per pour replan)
shortcut: the token is trained on the judge recordings' frames, not on demos (the paper: the task's demos)

Usage: ./run rltoken-token              the token encoder-decoder -> token.pt (+ Hub)
       ./run rltoken [--jobs 18]        online RL -> results/rltoken/<model>/
       ./run pour-rltoken [--jobs 18]   the pour benchmark at spots 5-8 with the trained actor (no exploration)
"""
import json
import os
import tempfile
import time

os.environ.setdefault("ARM", "rltoken")  # before cheat reads it (also in the spawned workers)
os.environ.setdefault("DSRL", "eval" if os.environ.get("RLT") == "eval" else "train")  # dsrl.py's training patches

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dsrl  # noqa: F401 (patches evaluate and cheat: pour-only trials, random starts, per-trial npz); patched again below
import cheat
import evaluate as ev
import grasp_position as gp
import runs
from dataset.export_lerobot import CAMERAS
from judge_check import HUB, MODEL, data_files

TRAIN = dsrl.TRAIN
# the token
RECORDINGS = ("judge-data", "judge-full")  # judge recordings whose frames train it
TOKEN_ROWS = 6_000      # frames drawn from them (RAM: about 0.4 MB each)
TOKEN_WIDTH = 256       # z_rl's size, and the encoder-decoder's width
TOKEN_LAYERS = 2        # per encoder and decoder
TOKEN_STEPS = 3_000
TOKEN_BATCH = 32
TOKEN_LR = 3e-4
HELD_OUT = 0.1          # share of the frames kept to measure reconstruction
# RL
ROUNDS = 30
C = ev.N_ACTION_STEPS   # actions the actor outputs (the ones run before the next replan)
N_A = len(ev.OPEN)
SCALE = np.r_[0.16, 0.22, 0.18, 0.11, 0.04, 0.05]  # spread (std) of SmolVLA's pour actions per dim (judge recordings), the gripper's out
DELTA = 1.0             # the correction is at most DELTA x SCALE per action
SIGMA = 0.3             # exploration noise while training: SIGMA x SCALE
BETA = 0.1              # actor loss + BETA x sum((a - ã) / SCALE)^2: stay near SmolVLA's chunk
GAMMA = 0.95            # per replan
UTD = 2                 # gradient steps per new transition (dsrl.py's 10 let its critic diverge)
POLICY_EVERY = 2        # critic steps per actor step (TD3)
REWARD_SCALE = dsrl.REWARD_SCALE
WARMUP = dsrl.WARMUP    # rounds of critic-only updates: the actor stays at SmolVLA's chunk
BATCH = 256
LR = 3e-4
TAU = 0.005
OUT = cheat.RESULTS.parent / "rltoken" / MODEL
HUB_DIR = f"worldmodel/rltoken/{MODEL}"
N_OBS, N_PRIV = dsrl.N_OBS, dsrl.N_PRIV
N_X = TOKEN_WIDTH + N_OBS
N_REF = C * N_A
STEER = torch.from_numpy(np.tile(np.r_[SCALE, 0.0], C)).float()  # per action number: its scale, 0 = the gripper (not steered)
NORM = torch.from_numpy(np.tile(np.r_[SCALE, 1.0], C)).float()   # divides actions into about unit size

_r = {}  # this process: the token model, the actor (and its file's mtime)


# ------------------------------------------------------------------------------------------------ the token
def prefix(policy, b):
    """SmolVLA's prefix after its VLM layers for a preprocessed batch b -> embeddings (B, M, width) float32 and
    which tokens are real (B, M) bool (language padding is not)."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    m = policy.model
    images, img_masks = policy.prepare_images(b)
    state = policy.prepare_state(b)
    emb, pad, att = m.embed_prefix(images, img_masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK], state=state)
    out, _ = m.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
                                       past_key_values=None, inputs_embeds=[emb, None], use_cache=True)  # as sample_actions: the prefix-only pass
    return out[0].float(), pad.bool()


def layers(n):
    layer = nn.TransformerEncoderLayer(TOKEN_WIDTH, 4, 4 * TOKEN_WIDTH, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, n)


class Token(nn.Module):
    """SmolVLA's prefix embeddings -> z_rl (TOKEN_WIDTH,), and back (the reconstruction that trains it)."""

    def __init__(self, width, tokens):
        super().__init__()
        self.width, self.tokens = width, tokens
        self.inp = nn.Linear(width, TOKEN_WIDTH)
        self.pos = nn.Parameter(torch.randn(tokens + 1, TOKEN_WIDTH) * 0.02)
        self.query = nn.Parameter(torch.randn(TOKEN_WIDTH) * 0.02)  # the learned <rl> token
        self.enc, self.dec = layers(TOKEN_LAYERS), layers(TOKEN_LAYERS)
        self.out = nn.Linear(TOKEN_WIDTH, width)

    def encode(self, emb, real):
        """emb (B, M, width), real (B, M) -> z_rl (B, TOKEN_WIDTH): the query token's output after attending to all."""
        n, m = emb.shape[:2]
        x = torch.cat([self.inp(emb) + self.pos[:m], (self.query + self.pos[m]).expand(n, 1, -1)], 1)
        ignore = torch.cat([~real, torch.zeros(n, 1, dtype=torch.bool, device=real.device)], 1)
        return self.enc(x, src_key_padding_mask=ignore)[:, -1]

    def loss(self, emb, real):
        """Rebuild each embedding from z_rl and the ones before it (teacher-forced, causal) -> mean squared error over
        the real tokens, and that error over the embeddings' variance (1 = no better than their mean)."""
        m = emb.shape[1]
        z = self.encode(emb, real)
        x = torch.cat([z[:, None], self.inp(emb[:, :-1])], 1) + self.pos[:m]
        h = self.dec(x, mask=nn.Transformer.generate_square_subsequent_mask(m, device=emb.device), is_causal=True)
        err = ((self.out(h) - emb) ** 2).mean(-1)
        mse = (err * real).sum() / real.sum()
        var = ((emb - emb[real].mean(0)) ** 2).mean(-1)[real].mean()
        return mse, mse.detach() / var


def token_frames():
    """TOKEN_ROWS frames of the judge recordings -> {camera: (n, H, W, 3) uint8}, state (n, 8), grasp memory (n, 3),
    plant (n,). Needs the policy loaded (grasp_memory's stats)."""
    from judge_train import grasp_memory
    files = [f for r in RECORDINGS for f in data_files(r)]
    sizes = [len(np.load(f)["state"]) for f in files]
    rng = np.random.default_rng(0)
    pick = rng.choice(sum(sizes), size=min(TOKEN_ROWS, sum(sizes)), replace=False)
    starts = np.cumsum([0] + sizes)
    out = {c: [] for c in CAMERAS} | {"state": [], "grasp": [], "plant": []}
    for i, f in enumerate(files):
        rows = pick[(pick >= starts[i]) & (pick < starts[i + 1])] - starts[i]
        if not len(rows):
            continue
        z = np.load(f)
        g = grasp_memory(f, z)
        for c in CAMERAS:
            out[c].append(z[c][rows])
        out["state"].append(z["state"][rows]), out["grasp"].append(g[rows])
        out["plant"] += [str(z["plant"])] * len(rows)
    print(f"{len(pick)} frames from {len(files)} recorded trials", flush=True)
    return {k: (np.array(v) if k == "plant" else np.concatenate(v)) for k, v in out.items()}


def train_token():
    """The token encoder-decoder on SmolVLA's frozen prefix embeddings of the recorded frames -> OUT/token.pt + Hub."""
    import wandb
    from huggingface_hub import HfApi
    model = ev.load_model(MODEL)  # first: grasp_memory() needs its stats
    policy, pre, _ = model
    f = token_frames()
    n = len(f["state"])
    gp.TABLE = torch.from_numpy(f["grasp"])  # the grasp slots by the batch's "index", as in training
    policy.train()  # prepare_state reads TABLE while training (state dropout P stays 0)
    rng = np.random.default_rng(0)
    test = rng.random(n) < HELD_OUT

    def embed(rows):
        batch = {c: torch.from_numpy(f[c][rows]).permute(0, 3, 1, 2).float().div(255) for c in CAMERAS}
        batch["observation.state"] = torch.from_numpy(f["state"][rows])
        batch["task"] = [f"Water the {p} plant" for p in f["plant"][rows]]
        with torch.no_grad():
            b = pre(batch)
            b["index"] = torch.from_numpy(rows)
            return prefix(policy, b)

    emb, real = embed(np.arange(2))
    tok = Token(emb.shape[-1], emb.shape[1] + 48).to(emb.device)  # + room: the prompt pads to the batch's longest
    opt = torch.optim.AdamW(tok.parameters(), TOKEN_LR)
    wandb.init(project=dsrl.WANDB_PROJECT, name=f"rltoken-token-{MODEL}",
               config=dict(frames=n, width=TOKEN_WIDTH, layers=TOKEN_LAYERS, steps=TOKEN_STEPS, batch=TOKEN_BATCH, lr=TOKEN_LR))
    for step in range(1, TOKEN_STEPS + 1):
        emb, real = embed(rng.choice(np.flatnonzero(~test), TOKEN_BATCH))
        loss, share = tok.loss(emb, real)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                _, held = tok.loss(*embed(rng.choice(np.flatnonzero(test), TOKEN_BATCH)))
            row = dict(loss=loss.item(), unexplained=share.item(), held_out_unexplained=held.item())
            wandb.log(row, step=step)
            print(f"token step {step}/{TOKEN_STEPS}: {row}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=tok.cpu().state_dict(), width=tok.width, tokens=tok.tokens), OUT / "token.pt")
    HfApi().upload_file(path_or_fileobj=OUT / "token.pt", path_in_repo=f"{HUB_DIR}/token.pt", repo_id=HUB, repo_type="dataset")
    wandb.finish()
    print(f"-> {OUT / 'token.pt'} (+ Hub {HUB_DIR}/token.pt)")


def token_model(device):
    """This process's frozen token encoder (OUT/token.pt, else the Hub's)."""
    if "token" not in _r:
        path = OUT / "token.pt"
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(HUB, f"{HUB_DIR}/token.pt", repo_type="dataset")
        t = torch.load(path, weights_only=True)
        tok = Token(t["width"], t["tokens"])
        tok.load_state_dict(t["state"])
        _r["token"] = tok.to(device).eval()
    return _r["token"]


# ------------------------------------------------------------------------------------------------ RL
class Actor(nn.Module):
    """x = (z_rl, obs), ã (N_REF,) -> the actions to run: ã + a bounded correction (gripper untouched)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_X))
        self.register_buffer("std", torch.ones(N_X))
        self.net = dsrl.mlp(N_X + N_REF, N_REF)
        nn.init.zeros_(self.net[-1].weight), nn.init.zeros_(self.net[-1].bias)  # fresh: exactly SmolVLA's chunk

    def forward(self, x, ref):
        h = self.net(torch.cat([(x - self.mean) / self.std, ref / NORM.to(ref)], -1))
        return ref + DELTA * STEER.to(ref) * torch.tanh(h)


class Critic(nn.Module):
    """(x, priv, actions) -> two Q estimates."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(N_X + N_PRIV))
        self.register_buffer("std", torch.ones(N_X + N_PRIV))
        self.q1, self.q2 = dsrl.mlp(N_X + N_PRIV + N_REF, 1), dsrl.mlp(N_X + N_PRIV + N_REF, 1)

    def forward(self, x, p, a):
        h = torch.cat([(torch.cat([x, p], -1) - self.mean) / self.std, a / NORM.to(a)], -1)
        return self.q1(h)[:, 0], self.q2(h)[:, 0]


def actor():
    """This process's actor, reloaded when actor.pt changes (training: after every round). Eval: from the Hub if not here."""
    path = OUT / "actor.pt"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HUB, f"{HUB_DIR}/actor.pt", repo_type="dataset")
    mtime = os.path.getmtime(path)
    if _r.get("mtime") != mtime:
        a = Actor()
        a.load_state_dict(torch.load(path, weights_only=True))
        _r.update(actor=a.eval(), mtime=mtime)
    return _r["actor"]


def pick(model, frame, task, s):
    """cheat's noise_policy: SmolVLA's chunk, refined by the actor from the RL token; the replan is recorded."""
    policy, pre, post = model
    batch = {key: torch.from_numpy(frame[key]).permute(2, 0, 1).float().div(255)[None] for key in CAMERAS}
    batch["observation.state"] = torch.from_numpy(frame["observation.state"])[None]
    batch["task"] = [task]
    with torch.inference_mode():
        b = pre(batch)
        emb, real = prefix(policy, b)  # first: predict_action_chunk rewrites b's camera entries
        z = token_model(emb.device).encode(emb, real)[0].cpu()
        ref = post(policy.predict_action_chunk(b))[0, :C].cpu().reshape(-1).float()
    o = dsrl.obs(frame, task)
    x = torch.cat([z, torch.from_numpy(o)])
    with torch.no_grad():
        a = actor()(x[None], ref[None])[0]
    if TRAIN:
        a = a + SIGMA * STEER * torch.randn(N_REF)
    e = cheat._run["env"].env
    st = e.watering_stats()
    dsrl._w["steps"].append(dict(tok=z.numpy(), obs=o, priv=dsrl.priv(s), ref=ref.numpy(), a=a.numpy(), good=st["target"],
                                 bad=st["spilled"] + sum(st[p] for p in e.pot_names if p != e.target_pot)))
    return a.view(C, N_A).numpy()


def transitions(files):
    """npz files -> tensors x, priv, ref, a, reward, next x, next priv, next ref, done (one row per replan)."""
    cols = {k: [] for k in ("x", "p", "ref", "a", "r", "x2", "p2", "ref2", "d")}
    nxt = lambda v: np.r_[v[1:], v[-1:]]
    for f in files:
        z = np.load(f)
        good, bad = np.r_[z["good"], z["end_good"]], np.r_[z["bad"], z["end_bad"]]
        x = np.c_[z["tok"], z["obs"]]
        n = len(x)
        cols["x"].append(x), cols["p"].append(z["priv"]), cols["ref"].append(z["ref"]), cols["a"].append(z["a"])
        cols["r"].append(np.diff(good) - np.diff(bad))
        cols["x2"].append(nxt(x)), cols["p2"].append(nxt(z["priv"])), cols["ref2"].append(nxt(z["ref"]))
        cols["d"].append(np.arange(n) == n - 1)
    return {k: torch.from_numpy(np.concatenate(v).astype(np.float32)) for k, v in cols.items()}


def new_learner():
    a, c, t = Actor(), Critic(), Critic()
    t.load_state_dict(c.state_dict())
    return dict(actor=a, critic=c, target=t, round=0, curve=[],
                opt_a=torch.optim.Adam(a.parameters(), LR), opt_c=torch.optim.Adam(c.parameters(), LR))


def update(L, data, steps, dev, train_actor=True):
    """steps TD3-style gradient steps on data (transitions()); train_actor=False: the critic only (WARMUP).
    Returns the last batch's critic loss, the critic's Q of the actor's chunk, how far that chunk is from SmolVLA's (in
    SCALEs) and, if it trained, the actor's loss."""
    a, c, t = L["actor"].to(dev), L["critic"].to(dev), L["target"].to(dev)
    d = {k: v.to(dev) for k, v in data.items()}
    n, norm = len(d["r"]), NORM.to(dev)
    for step in range(steps):
        i = torch.randint(n, (BATCH,), device=dev)
        with torch.no_grad():
            y = REWARD_SCALE * d["r"][i] + GAMMA * (1 - d["d"][i]) * torch.min(*t(d["x2"][i], d["p2"][i], a(d["x2"][i], d["ref2"][i])))
        q1, q2 = c(d["x"][i], d["p"][i], d["a"][i])
        loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        L["opt_c"].zero_grad()
        loss_c.backward()
        L["opt_c"].step()
        if train_actor and step % POLICY_EVERY == 0:
            act = a(d["x"][i], d["ref"][i])
            leash = (((act - d["ref"][i]) / norm) ** 2).sum(-1)
            loss_a = (BETA * leash - torch.min(*c(d["x"][i], d["p"][i], act))).mean()
            L["opt_a"].zero_grad()
            loss_a.backward()
            L["opt_a"].step()
        with torch.no_grad():
            for pt, pc in zip(t.parameters(), c.parameters()):
                pt.lerp_(pc, TAU)
    with torch.no_grad():
        act = a(d["x"][i], d["ref"][i])
        q = torch.min(*c(d["x"][i], d["p"][i], act)).mean().item()
        off = (((act - d["ref"][i]) / norm) ** 2).sum(-1).sqrt().mean().item()
    out = dict(critic_loss=loss_c.item(), q=q, off_smolvla=off)
    if train_actor:  # its first step always trains the actor (step 0)
        out["actor_loss"] = loss_a.item()
    return out


def save(L):
    """learner.pt (to resume) and actor.pt (what the workers load).
    shortcut: the optimizers' state isn't kept (it lives on the GPU): a resumed run restarts Adam's moments"""
    torch.save({k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in L.items() if not k.startswith("opt")},
               OUT / "learner.pt")
    torch.save({k: v.cpu() for k, v in L["actor"].state_dict().items()}, OUT / "actor.tmp")
    os.replace(OUT / "actor.tmp", OUT / "actor.pt")  # whole file at once: a worker never reads half of it


def load():
    L = new_learner()
    if (OUT / "learner.pt").exists():
        for k, v in torch.load(OUT / "learner.pt", weights_only=False).items():
            if hasattr(L.get(k), "load_state_dict"):
                L[k].load_state_dict(v)
            else:
                L[k] = v
    return L


def scale_inputs(L, data):
    """Fix the networks' input scaling from data's states (once per learner)."""
    L["actor"].mean.copy_(data["x"].mean(0)), L["actor"].std.copy_(data["x"].std(0).clamp(min=1e-2))
    xp = torch.cat([data["x"], data["p"]], -1)
    for c in (L["critic"], L["target"]):
        c.mean.copy_(xp.mean(0)), c.std.copy_(xp.std(0).clamp(min=1e-2))
    L["scaled"] = True


def train():
    """ROUNDS x (EPISODES pour trials with the current actor, then UTD TD3 steps per new replan)."""
    OUT.mkdir(parents=True, exist_ok=True)
    token_model("cpu")  # fail now, not in every worker, if the token isn't trained yet
    dev = runs.device()
    L = load()
    L["wandb_id"] = L.get("wandb_id") or f"rltoken-{int(time.time())}"  # kept in learner.pt: a resumed run continues the same W&B run
    wb = dict(name=f"rltoken-{MODEL}", id=L["wandb_id"],
              config=dict(model=MODEL, rounds=ROUNDS, episodes=dsrl.EPISODES, delta=DELTA, sigma=SIGMA, beta=BETA,
                          gamma=GAMMA, utd=UTD, reward_scale=REWARD_SCALE, warmup=WARMUP, batch=BATCH, lr=LR, tau=TAU))
    wb = dsrl.wb_open(L["curve"], **wb)
    dsrl.catch_up(L, update, transitions, scale_inputs, dev)  # a fresh learner beside recorded pours: learn from them first
    save(L)  # the workers need an actor (fresh: SmolVLA's chunk, plus exploration noise)
    for rnd in range(L["round"], ROUNDS):
        units = [f"{p}-0-{rnd * 100 + i}" for i in range(dsrl.EPISODES // len(ev.PLANTS)) for p in ev.PLANTS]
        ev.UNITS = units
        ev.main()
        files = sorted((OUT / "data").glob("*.npz"))
        mine = [f for f in files if f.stem in units]
        if not mine:
            print(f"round {rnd}: no pour in this round (none carried?), next round")
            continue
        data = transitions(files)
        if not L.get("scaled"):  # input scaling, fixed from the first round's states
            scale_inputs(L, data)
        new = transitions(mine)
        t0 = time.time()
        stats = update(L, data, UTD * len(new["r"]), dev, train_actor=rnd >= WARMUP)
        ends = [np.load(f) for f in mine]
        row = dict(round=rnd, episodes=len(mine), replans=len(new["r"]), all_replans=len(data["r"]),
                   drops=round(float(np.mean([z["end_good"] for z in ends])), 2),
                   watered=round(float(np.mean([z["watered"] for z in ends])), 2),
                   reward=round(float(new["r"].sum()) / len(mine), 2), **{k: round(v, 4) for k, v in stats.items()},
                   update_s=round(time.time() - t0))
        L["curve"].append(row)
        L["round"] = rnd + 1
        for k in ("actor", "critic", "target"):
            L[k].cpu()
        save(L)
        dsrl.report(L["curve"])
        wb.log(row, step=rnd)
        print(f"\n=== RL token round {rnd}: {row}\n", flush=True)
    wb.finish()


# at import, so the spawned eval workers get them too: dsrl.py's trial machinery, this file's actor and folders
cheat.noise_policy = pick
dsrl.OUT, dsrl.HUB_DIR = OUT, HUB_DIR  # its trial() saves the replans to OUT/data, its report() uploads OUT
if TRAIN:
    cheat.RESULTS = ev.RESULTS = OUT.parent

if __name__ == "__main__":
    if os.environ.get("RLT") == "token":
        train_token()
    elif TRAIN:
        train()
    else:
        ev.main()
