"""Success rates of the replay (the mug alone follows the tracked pour) and the robot: overall, per plant
and per start position; plus one row per episode.

Reads dataset/data/water-pour/<clip>/{track,replay,robot}.json; writes dataset/data/stats.md.

Usage: ./run stats [A1-0 B3-2 ...]   (default: every tracked episode)
"""
import argparse
import json

from dataset import episodes

OUT = episodes.DATA / "stats.md"


def load(clip, name):
    """The clip's <name>.json as a dict ({} if missing)."""
    path = episodes.clip_dir(clip) / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def rate(rows, key):
    """'k/n (p%)' over the rows that have a result for `key`."""
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return "-"
    k, n = sum(vals), len(vals)
    return f"{k}/{n} ({k / n:.0%})"


def episode(clip):
    """One row: name parts, tracking quality, replay and robot results (None = not run)."""
    track, replay, robot = load(clip, "track"), load(clip, "replay"), load(clip, "robot")
    return dict(clip=clip, **(episodes.parse(clip) or dict(plant=None, position=None, rep=None)),
                detected=track.get("detected"), start_offset_cm=track.get("start_offset_cm"),
                replay=replay.get("success"), robot=robot.get("success"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*")
    rows = [episode(c) for c in p.parse_args().clips or episodes.tracked()]

    groups = [("all", rows)]
    groups += [(f"plant {s}", [r for r in rows if r["plant"] == s]) for s in episodes.PLANTS.values()]
    groups += [(f"position {k}", [r for r in rows if r["position"] == k]) for k in range(1, 5)]
    header = ("group", "episodes", "replay (mug alone)", "robot")
    cells = [(name, str(len(rs)), rate(rs, "replay"), rate(rs, "robot")) for name, rs in groups if rs]
    rates = ["| " + " | ".join(header) + " |", "|---|---|---|---|"] + ["| " + " | ".join(c) + " |" for c in cells]

    yn = lambda v: "-" if v is None else ("yes" if v else "**no**")
    fmt = lambda v, f: "-" if v is None else format(v, f)
    table = ["| clip | tag seen | start offset (cm) | replay | robot |", "|---|---|---|---|---|"]
    table += [f"| {r['clip']} | {fmt(r['detected'], '.0%')} | {fmt(r['start_offset_cm'], '.1f')} "
              f"| {yn(r['replay'])} | {yn(r['robot'])} |" for r in rows]

    OUT.write_text("\n".join(["# Dataset statistics", "", "## Success rates", "",
                              "Robot success = water in the commanded plant and the mug put back.", "",
                              *rates, "", "## Episodes", "", *table]) + "\n")
    widths = [max(len(row[i]) for row in [header, *cells]) for i in range(4)]
    line = lambda row: "  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(row, widths)))
    print(line(header), "-" * len(line(header)), *map(line, cells), f"\n-> {OUT.relative_to(episodes.ROOT.parent)}", sep="\n")


if __name__ == "__main__":
    main()
