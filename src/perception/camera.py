"""Camera calibration from checkerboard images (Zhang's method) and its quality check.

Adapted from 41014 Week 3 Part B (calibrate from a checkerboard set: gridshape = internal corners,
squaresize sets the metric scale; report f, principal point and field of view) and Part C
(measured_corners with sub-pixel refinement; per-image reprojection error e_i = ||p_i - p_hat_i||).
OpenCV is called directly instead of the Machine Vision Toolbox wrapper; the steps are the same.
Works on frames from the simulator and on images saved from the real Pi camera.
"""
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np

GRIDSHAPE = (9, 6)      # internal corners (columns, rows), the W3 board
SQUARESIZE = 0.025      # m, side of one square, the W3 board


def checkerboard_image(gridshape=GRIDSHAPE, px=60, pad=60):
    """Printable checkerboard with gridshape internal corners and a white margin (greyscale)."""
    squares = (np.indices((gridshape[1] + 1, gridshape[0] + 1)).sum(axis=0) % 2 * 255).astype(np.uint8)
    board = np.kron(squares, np.ones((px, px), np.uint8))
    return cv2.copyMakeBorder(board, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def board_points(gridshape=GRIDSHAPE, squaresize=SQUARESIZE):
    """3D corners in the board frame, N x 3; the board is the plane Z = 0 (W3 Part C P_board)."""
    gx, gy = gridshape
    return np.array([[i * squaresize, j * squaresize, 0.0] for j in range(gy) for i in range(gx)], np.float32)


def measured_corners(gray, gridshape=GRIDSHAPE):
    """Sub-pixel checkerboard corners as an N x 2 array, or None (W3 Part C measured_corners).

    The refinement window follows the square size in the image (W3 uses 11 x 11 on large squares;
    a window wider than about half a square pulls corners towards the neighbouring ones).
    """
    ok, c = cv2.findChessboardCorners(gray, gridshape)
    if not ok:
        return None
    sq_px = np.median(np.linalg.norm(np.diff(c.reshape(gridshape[1], gridshape[0], 2), axis=1), axis=2))
    win = int(np.clip(0.4 * sq_px, 3, 11))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    c = cv2.cornerSubPix(gray, c, (win, win), (-1, -1), crit)
    return c.reshape(-1, 2)


def calibrate(gray_images, gridshape=GRIDSHAPE, squaresize=SQUARESIZE):
    """Zhang calibration from greyscale images of the board (W3 Part B), with W3 Part C errors.

    Returns a dict: K (3x3), dist (k1, k2, p1, p2, k3), rms (px), per_image_rms (px),
    used (indices of the images where the board was found), image_size (w, h).
    """
    obj, img, used = [], [], []
    for i, g in enumerate(gray_images):
        c = measured_corners(g, gridshape)
        if c is not None:
            obj.append(board_points(gridshape, squaresize))
            img.append(c.astype(np.float32))
            used.append(i)
    if len(used) < 5:
        raise RuntimeError(f"board found in only {len(used)} of {len(gray_images)} images; need at least 5")
    size = (gray_images[0].shape[1], gray_images[0].shape[0])
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None)
    per_image = []
    for o, p, rv, tv in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per_image.append(float(np.sqrt(np.mean(np.sum((p - proj.reshape(-1, 2)) ** 2, axis=1)))))
    return dict(K=K, dist=dist.ravel(), rms=float(rms), per_image_rms=per_image, used=used,
                n_images=len(gray_images), image_size=size)


def fov_deg(K, image_size):
    """Horizontal and vertical field of view [deg] from the calibrated K (W3 Part B2)."""
    w, h = image_size
    return np.degrees(2 * np.arctan(w / (2 * K[0, 0]))), np.degrees(2 * np.arctan(h / (2 * K[1, 1])))


def save_intrinsics(path, calib, source):
    """Write K, distortion and the calibration quality to JSON."""
    data = dict(source=source, date=str(date.today()), image_size=list(calib["image_size"]),
                K=np.asarray(calib["K"]).tolist(), dist=np.asarray(calib["dist"]).tolist(),
                rms_px=calib["rms"], per_image_rms_px=calib["per_image_rms"],
                images_used=len(calib["used"]), images_total=calib["n_images"],
                gridshape=list(GRIDSHAPE), squaresize_m=SQUARESIZE)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def load_intrinsics(path):
    """K (3x3), distortion and image size (w, h) from a file written by save_intrinsics."""
    data = json.loads(Path(path).read_text())
    return np.array(data["K"]), np.array(data["dist"]), tuple(data["image_size"])