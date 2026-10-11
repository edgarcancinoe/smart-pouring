"""Data paths, episode names, and running a job over many clips.

dataset/data/<group>/<clip>/ holds a clip's recording (<clip>.MOV) and everything made from it.
Episodes: water-pour/<plant><position>-<rep>, A1-0 ... C4-3 (plant A/B/C = left/middle/right,
start position 1-4, repetition 0-3).
"""
import json
import multiprocessing
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EPISODES = DATA / "water-pour"
CALIBRATION_VIDEOS = DATA / "calibration"
HUB_DATASET = "edgarcancinoe/water_pour"  # the LeRobot dataset on the Hugging Face Hub (./run upload)
SETUP = json.loads((ROOT / "setup.json").read_text())  # measured layout and sizes, floor frame (cm)
PLANTS = {"A": "left", "B": "middle", "C": "right"}
_NAME = re.compile(r"^([ABC])([1-4])-([0-3])$", re.IGNORECASE)


def parse(clip):
    """Expected plant, start position (1-4) and repetition (0-3) from the name; None for other names."""
    m = _NAME.match(clip)
    if not m:
        return None
    return dict(plant=PLANTS[m.group(1).upper()], position=int(m.group(2)), rep=int(m.group(3)))


def sort_key(clip):
    """A1-0, A1-1, ..., A4-3, B1-0, ...; other names after."""
    m = _NAME.match(clip)
    return (0, m.group(1).upper(), int(m.group(2)), int(m.group(3)), "") if m else (1, "", 0, 0, clip)


def groups():
    """The clip folders' parents: water-pour first, then the other folders in data/ but calibration and lerobot."""
    return [EPISODES, *sorted(p for p in DATA.iterdir()
                              if p.is_dir() and p not in (EPISODES, CALIBRATION_VIDEOS, DATA / "lerobot"))]


def clip_dir(clip):
    """The clip's folder: data/<group>/<clip>/ (water-pour/<clip>/ if there is none yet)."""
    for g in groups():
        if (g / clip).is_dir():
            return g / clip
    return EPISODES / clip


def video(clip):
    """The clip's recording: data/<group>/<clip>/<clip>.mov/.MOV/.mp4."""
    for ext in (".mov", ".MOV", ".mp4"):
        if (clip_dir(clip) / f"{clip}{ext}").exists():
            return clip_dir(clip) / f"{clip}{ext}"
    raise FileNotFoundError(f"no video for {clip} in {DATA}")


def episodes_first(names):
    """The A1-0-style episodes if there are any, else every name; in episode order."""
    named = [n for n in names if parse(n)]
    return sorted(named or names, key=sort_key)


def recorded():
    """The episodes in data/water-pour, in order."""
    return episodes_first({d.name for d in EPISODES.iterdir() if d.is_dir() and any(
        (d / f"{d.name}{ext}").exists() for ext in (".mov", ".MOV", ".mp4"))})


def having(name):
    """Clips whose folder has the file `name` (the episodes, once there are any), in order."""
    return episodes_first([d.name for g in groups() for d in g.iterdir() if (d / name).exists()])


def tracked():
    """Clips with a track, in order."""
    return having("track.npz")


# --- run one job per clip in separate processes (each clip is independent; MuJoCo is single-threaded)

def _safe(fn, clip, kwargs):
    """fn(clip, **kwargs), or {clip, error} if it fails: one bad clip must not stop a 48-clip batch."""
    try:
        return fn(clip, **kwargs)
    except (Exception, SystemExit) as err:
        traceback.print_exc()
        return dict(clip=clip, error=str(err) or type(err).__name__)


def _clock(s):
    s = int(s)
    return f"{s // 3600}h{s // 60 % 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def parallel_map(fn, clips, kwargs, jobs=0, label="clips"):
    """Yield fn(clip, **kwargs) for each clip as they finish ({clip, error} for a failed one), printing
    progress as each one ends. jobs=0: up to half the CPU cores."""
    jobs = jobs or max(1, min(len(clips), (os.cpu_count() or 2) // 2))
    n, t0 = len(clips), time.time()
    print(f"{label}: {n} clip{'s' * (n != 1)}, {jobs} at a time", flush=True)

    def progress(k, r):
        took = time.time() - t0
        left = f", ~{_clock(took / k * (n - k))} left" if k < n else ""
        status = "FAILED" if "error" in r else "done"
        print(f"[{k:{len(str(n))}d}/{n}] {r['clip']} {status} · {_clock(took)} elapsed{left}", flush=True)
        return r

    if jobs == 1 or n == 1:
        for k, c in enumerate(clips, 1):
            yield progress(k, _safe(fn, c, kwargs))
        return
    # spawn: a clean interpreter per worker (MuJoCo / OpenGL state doesn't survive fork well on macOS)
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_safe, fn, c, kwargs) for c in clips]
        for k, f in enumerate(as_completed(futures), 1):
            yield progress(k, f.result())
