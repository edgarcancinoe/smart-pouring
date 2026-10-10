"""Printable marker sheets (A4, print at 100%) and their layout.

Page 1: the ChArUco board, filmed for lens calibration and then taped to the floor (camera pose).
Page 2: the mug's ArUco codes, ids 0 and 1, three sizes each.
Writes dataset/calibration/markers.pdf and markers.json (read by calibration and tracking).

Usage: ./run markers
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DPI = 300
A4_PX = (round(210 / 25.4 * DPI), round(297 / 25.4 * DPI))  # (w, h)
MM = DPI / 25.4  # pixels per mm

BOARD = dict(dictionary="DICT_5X5_100", squares_x=5, squares_y=7, square_mm=30.0, marker_mm=22.0)
MUG = dict(dictionary="DICT_4X4_50", ids=[0, 1], sizes_mm=[30, 40, 50])

OUT = Path(__file__).resolve().parent  # dataset/calibration/


def page():
    return np.full((A4_PX[1], A4_PX[0]), 255, np.uint8)


def text(img, s, y, scale=1.6, x=120):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 3, cv2.LINE_AA)


def board_page():
    b = BOARD
    sq = round(b["square_mm"] * MM)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b["dictionary"]))
    board = cv2.aruco.CharucoBoard((b["squares_x"], b["squares_y"]), b["square_mm"] / 1000, b["marker_mm"] / 1000, dic)
    img_b = board.generateImage((b["squares_x"] * sq, b["squares_y"] * sq), marginSize=0, borderBits=1)
    p = page()
    x0, y0 = (A4_PX[0] - img_b.shape[1]) // 2, 400
    p[y0:y0 + img_b.shape[0], x0:x0 + img_b.shape[1]] = img_b
    text(p, f"ChArUco board: {b['square_mm']:.0f} mm squares, print at 100%", 200, 1.6)
    # corner A = the board's origin (top-left of the generated image)
    cv2.circle(p, (x0, y0), 26, 0, 7)
    text(p, "A", y0 - 40, 1.6, x=x0 - 15)
    return p


def mug_page():
    m = MUG
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, m["dictionary"]))
    p = page()
    text(p, "Mug codes (ArUco 4x4), print at 100%", 200, 1.6)
    quiet = round(5 * MM)  # white border the detector needs around the code
    for col, tag_id in enumerate(m["ids"]):
        x, y = 300 + col * 1150, 520
        text(p, f"id {tag_id}", y - 110, 1.5, x=x - quiet)
        for size in m["sizes_mm"]:
            px = round(size * MM)
            cv2.rectangle(p, (x - quiet, y - quiet), (x + px + quiet, y + px + quiet), 150, 2)  # cut line
            p[y:y + px, x:x + px] = cv2.aruco.generateImageMarker(dic, tag_id, px)
            text(p, f"{size} mm", y + px // 2 + 20, 1.4, x=x + px + quiet + 50)
            y += px + 2 * quiet + 160
    return p


def main():
    pages = [Image.fromarray(board_page()), Image.fromarray(mug_page())]
    pdf = OUT / "markers.pdf"
    pages[0].save(pdf, resolution=DPI, save_all=True, append_images=pages[1:])
    (OUT / "markers.json").write_text(json.dumps({"board": BOARD, "mug": MUG}, indent=2))
    print("dataset/calibration/markers.pdf, markers.json")


if __name__ == "__main__":
    main()
