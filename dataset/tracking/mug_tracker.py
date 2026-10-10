"""Track the mug in a recorded clip: camera pose from the floor board, mug pose from its code.

Outputs, per clip, in dataset/data/<group>/<clip>/:
  track.npz    t, ok, mug_pos (floor cm: bottom centre), mug_R (3x3, floor frame),
               cam_R/cam_t (camera -> floor, cm), K, dist, image_size
  track.json   for ./run stats: how often the mug code was found, the longest gap without it,
               the nearest measured start spot and the distance to it, max tilt

Floor frame (dataset/setup.json): cm, +x to the demonstrator's right, +y away from them, +z up.

Usage: ./run track [A1-0 B3-2 ...] [--jobs 4]   (default: every episode in dataset/data/water-pour/)
"""
import argparse
import json
import subprocess

import cv2
import numpy as np

from dataset import episodes
from dataset.calibration.calibrate import MARKERS, OUT as INTRINSICS, make_board
from dataset.episodes import SETUP

MIN_BOARD_CORNERS = 8   # fewer than this gives a shaky board pose
BOARD_FRAMES = 90       # frames searched for the board at the start (~1.5 s), before the hand covers it
START_FRAMES = 30       # first mug sightings averaged into its start position


# --- markers and frames ---

def mug_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKERS["mug"]["dictionary"]))


def board_to_floor():
    """Board coordinates (m; x along the printed rows, y down the page, z into the paper) -> floor (cm).

    The board lies readable from the demonstrator's spot with corner A (its origin) at the far
    left, so board x = floor +x, board y = floor -y, board z = floor -z (down).
    """
    A = np.array([*SETUP["board"]["corner_A_xy"], 0.0])
    R = np.diag([1.0, -1.0, -1.0])
    return R, A


def load_intrinsics(image_size):
    """K and distortion, rescaled if the clip has another resolution than the calibration."""
    cal = json.loads(INTRINSICS.read_text())
    K, dist = np.array(cal["K"]), np.array(cal["dist"])
    (w, h), (cal_w, cal_h) = image_size, cal["image_size"]
    if (w, h) != (cal_w, cal_h):
        if abs(w / h - cal_w / cal_h) > 1e-3:
            raise RuntimeError(f"calibration is {cal_w}x{cal_h} but the clip is {w}x{h} (different shape); recalibrate")
        K[:2] *= w / cal_w  # same aspect ratio: pixel intrinsics scale with the image
    return K, dist


# --- poses ---

def camera_pose(cap, K, dist, board):
    """Camera -> floor (R, t in cm) from the clearest board view in the first frames; None if no board.
    Rewinds the video."""
    det = cv2.aruco.CharucoDetector(board)
    views = []  # (R board->camera, t board->camera in m, corners seen)
    for _ in range(BOARD_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, _, _ = det.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) < MIN_BOARD_CORNERS:
            continue
        obj, img = board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        if ok:
            views.append((cv2.Rodrigues(rvec)[0], tvec.ravel(), len(ids)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not views:
        return None

    R_bc, t_bc, _ = max(views, key=lambda v: v[2])
    R_fb, A = board_to_floor()
    # x_floor = R_fb @ R_bc^T (x_cam - t_bc) * 100 + A
    R_cf = R_fb @ R_bc.T
    t_cf = -R_cf @ t_bc * 100 + A
    return R_cf, t_cf


def tag_pose(gray, det, K, dist, size_m, tag_id, choose):
    """Pose of the mug code (tag -> camera, m); None if not seen.

    A small flat square has two near-equal solutions (tilted toward or away from the camera);
    `choose(list of (R, t))` picks one, so the pose doesn't flip.
    """
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or tag_id not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(tag_id)].reshape(4, 2)
    s = size_m / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not n:
        return None
    return choose([(cv2.Rodrigues(r)[0], t.ravel()) for r, t in zip(rvecs, tvecs)])


def pick_solution(candidates, R_cf, R_prev):
    """Of the two tag solutions, the one closest to the last frame, or on first sight the one
    whose normal is most horizontal (the mug starts upright, so its code faces sideways)."""
    if R_prev is None:
        return min(candidates, key=lambda rt: abs((R_cf @ rt[0])[2, 2]))
    return min(candidates, key=lambda rt: np.linalg.norm(R_cf @ rt[0] - R_prev))


def upright_mug_axes(R_tf):
    """Mug axes (x, y = front, z = up) in the floor frame while it stands upright facing the tag normal."""
    front = R_tf[:, 2].copy()
    front[2] = 0
    front /= np.linalg.norm(front)
    up = np.array([0.0, 0.0, 1.0])
    return np.column_stack([np.cross(front, up), front, up])


def tag_offset():
    """Where the tag sits on the mug: height above the bottom and the mug radius at that height (cm)."""
    mug = SETUP["mug"]
    h = SETUP["mug_tag"]["center_height_above_mug_bottom"]
    r_bot, r_top = mug["diameter_bottom"] / 2, mug["diameter_top"] / 2
    return h, r_bot + (r_top - r_bot) * h / mug["height"]  # the mug is a truncated cone


def tilt_deg(R_mug):
    """Angle between the mug's axis and floor up."""
    return np.degrees(np.arccos(np.clip(R_mug[..., 2, 2], -1, 1)))


# --- video ---

def to_h264(src, dst):
    """OpenCV can only write mp4v, which many players (and browsers) won't play; re-encode with ffmpeg."""
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", str(dst)])
    if r.returncode == 0:
        src.unlink()
    else:
        src.rename(dst)  # keep the mp4v file rather than nothing


# --- tracking ---

def longest_run(mask):
    """Length of the longest run of True in a boolean sequence."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def summarize(clip, fps, ok, pos, rot):
    """Numbers for track.json (read by ./run stats)."""
    found = np.flatnonzero(ok)
    if not len(found):
        return dict(clip=clip, error="mug code never found")
    start = np.median(pos[found[:START_FRAMES]], axis=0)  # standing on its start spot
    dist = np.linalg.norm(np.array(SETUP["mug_starts"]) - start[:2], axis=1)
    return dict(clip=clip, detected=round(float(ok.mean()), 3), longest_gap_s=round(longest_run(~ok) / fps, 2),
                start_spot=int(dist.argmin()) + 1, start_offset_cm=round(float(dist.min()), 1),
                max_tilt_deg=round(float(np.nanmax(tilt_deg(rot))), 1))


def track(clip):
    """Track one clip; writes track.npz and track.json (returned) next to its video."""
    out_dir = episodes.clip_dir(clip)
    cap = cv2.VideoCapture(str(episodes.video(clip)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K, dist = load_intrinsics((w, h))

    cam = camera_pose(cap, K, dist, make_board())
    if cam is None:
        summary = dict(clip=clip, error=f"floor board not found in the first {BOARD_FRAMES} frames")
        (out_dir / "track.json").write_text(json.dumps(summary, indent=2))
        return summary
    R_cf, t_cf = cam

    tag_id, tag_m = SETUP["mug_tag"]["id"], SETUP["mug_tag"]["size_mm"] / 1000
    tag_height, tag_radius = tag_offset()
    det = cv2.aruco.ArucoDetector(mug_dictionary())

    ok, pos, rot = [], [], []
    R_tag_to_mug = R_prev = None  # fixed on first sighting / last frame's tag rotation
    while True:
        read, frame = cap.read()
        if not read:
            break
        tp = tag_pose(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), det, K, dist, tag_m, tag_id,
                      lambda cands: pick_solution(cands, R_cf, R_prev))
        p, R_mug = np.full(3, np.nan), np.full((3, 3), np.nan)
        if tp is not None:
            R_tc, t_tc = tp
            R_tf = R_cf @ R_tc               # tag axes, floor frame
            t_tf = R_cf @ t_tc * 100 + t_cf  # tag centre, floor cm
            if R_tag_to_mug is None:
                R_tag_to_mug = R_tf.T @ upright_mug_axes(R_tf)
            R_mug = R_tf @ R_tag_to_mug
            p = t_tf - R_mug[:, 2] * tag_height - R_mug[:, 1] * tag_radius  # down the axis, back to the centre
            R_prev = R_tf
        ok.append(tp is not None)
        pos.append(p)
        rot.append(R_mug)

    ok, pos, rot = np.array(ok), np.array(pos), np.array(rot)
    np.savez(out_dir / "track.npz", t=np.arange(len(ok)) / fps, ok=ok, mug_pos=pos, mug_R=rot,
             cam_R=R_cf, cam_t=t_cf, K=K, dist=dist, image_size=np.array([w, h]))
    summary = summarize(clip, fps, ok, pos, rot)
    (out_dir / "track.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="*", help="clip names (dataset/data/*/<name>/<name>.MOV); default: every episode")
    p.add_argument("--jobs", type=int, default=0, help="clips tracked in parallel (default: up to half the CPU cores)")
    args = p.parse_args()

    for r in episodes.parallel_map(track, args.clips or episodes.recorded(), {}, args.jobs, "tracking"):
        if "error" in r:
            print(f"{r['clip']}: {r['error']}")


if __name__ == "__main__":
    main()
