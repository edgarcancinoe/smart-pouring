"""Phone lens intrinsics from videos of the ChArUco board (layout: dataset/calibration/markers.json).

Detects the board in every EVERY-th frame, calibrates, drops the worst views and calibrates again.
Writes dataset/calibration/intrinsics.json: image_size, K (3x3), dist (5,), rms_px, views, videos.

Usage: ./run calibrate [video ...]   (default: every .MOV in dataset/data/calibration/)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import episodes

OUT = Path(__file__).resolve().parent / "intrinsics.json"
MARKERS = json.loads((Path(__file__).resolve().parent / "markers.json").read_text())  # written by ./run markers
EVERY = 10        # use every Nth frame
MIN_CORNERS = 10  # skip frames that see only a sliver of the board
MIN_VIEWS = 15    # fewer usable frames than this: calibration is unreliable
KEEP = 85         # percentile: views with a larger reprojection error are dropped before the refit
FLAGS = cv2.CALIB_FIX_K3  # no k3: it runs away when the views don't reach the image edges


def make_board():
    """The printed ChArUco floor board, scaled to its measured square size (m)."""
    b = MARKERS["board"]
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b["dictionary"]))
    square = episodes.SETUP["board"]["square_mm_measured"] / 1000
    marker = square * b["marker_mm"] / b["square_mm"]  # printing scales both together
    return cv2.aruco.CharucoBoard((b["squares_x"], b["squares_y"]), square, marker, dic)


def board_views(videos, board):
    """Board corners in every EVERY-th frame that sees at least MIN_CORNERS of them.

    videos -> video paths
    board  -> cv2.aruco.CharucoBoard
    returns obj_pts [(n, 1, 3)] board points, img_pts [(n, 1, 2)] pixels, one pair per view; image size (w, h)
    """
    detector = cv2.aruco.CharucoDetector(board)
    obj_pts, img_pts, size = [], [], None
    for path in videos:
        cap = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % EVERY == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                corners, ids, _, _ = detector.detectBoard(gray)
                if ids is not None and len(ids) >= MIN_CORNERS:
                    obj, img = board.matchImagePoints(corners, ids)
                    obj_pts.append(obj)
                    img_pts.append(img)
            i += 1
        cap.release()
    return obj_pts, img_pts, size


def view_errors(obj_pts, img_pts, size, K, dist):
    """RMS reprojection error of each view (px), with the board pose re-solved for the given intrinsics."""
    _, _, _, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, K, dist, flags=FLAGS | cv2.CALIB_USE_INTRINSIC_GUESS)
    errs = []
    for o, im, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(np.sqrt(np.mean(np.sum((proj - im) ** 2, axis=-1))))
    return np.array(errs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="*", help="default: every .MOV in dataset/data/calibration/")
    videos = p.parse_args().videos or sorted(map(str, episodes.CALIBRATION_VIDEOS.glob("*.MOV")))
    if not videos:
        raise SystemExit(f"no calibration videos in {episodes.CALIBRATION_VIDEOS}")

    # 1. Find the board: each usable frame gives matching board points (m) and image points (px).
    obj_pts, img_pts, size = board_views(videos, make_board())
    if len(obj_pts) < MIN_VIEWS:
        raise SystemExit(f"only {len(obj_pts)} board views (need {MIN_VIEWS}); record a longer calibration video")

    # 2. First calibration over every view.
    _, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=FLAGS)

    # 3. Drop the views that fit worst (blurred frames, bent paper) and calibrate again on the rest.
    errs = view_errors(obj_pts, img_pts, size, K, dist)
    keep = errs < np.percentile(errs, KEEP)
    obj_pts = [o for o, k in zip(obj_pts, keep) if k]
    img_pts = [im for im, k in zip(img_pts, keep) if k]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=FLAGS)

    # 4. Save. rms_px: average distance between detected and re-projected corners (good: < 1 px).
    result = {"image_size": list(size), "K": K.tolist(), "dist": dist.ravel().tolist(),
              "rms_px": rms, "views": len(obj_pts), "videos": [Path(v).name for v in videos]}
    OUT.write_text(json.dumps(result, indent=2))
    print(f"dataset/calibration/intrinsics.json: RMS {rms:.2f} px over {len(obj_pts)} views")


if __name__ == "__main__":
    main()
