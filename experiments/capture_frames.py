"""Save frames from the real robot camera: checkerboard views for calibration, or ArUco test shots.

    python experiments/capture_frames.py 192.168.1.213                     # SPACE (window) or Enter (terminal) saves, Q quits
    python experiments/capture_frames.py 192.168.1.213 --every 1.5 --n 25  # auto: a frame every 1.5 s with the board in view
    python experiments/calibrate_camera.py --images results/hardware/calib --out config/camera_real.json

The window shows the live camera with the checkerboard corners (green when all 9 x 6 are found) and
the ArUco IDs in view. Frames are saved untouched (PNG, no overlay) to --out, default
results/hardware/calib. The robot never moves: this only reads the camera. For calibration, aim for
20-25 frames with the board at 0.3-0.8 m, tilted different ways, covering the corners of the image too.
Adapted from 41014 Week 3 Part B (the 9 x 6 internal-corner board, findChessboardCorners) and Part E
(ArUco detection); the calibration itself is experiments/calibrate_camera.py (Week 3 Parts B and C).
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402

from src.perception import aruco                                # noqa: E402
from src.perception.camera import GRIDSHAPE                     # noqa: E402
from src.robot.real_robot import RealRobot                      # noqa: E402
from src.robot.sim_robot import PROJECT_ROOT                    # noqa: E402


def terminal_keys(requests):
    """Enter in the terminal asks for a save, q + Enter for quitting (the OpenCV window can miss keys on Wayland)."""
    for line in sys.stdin:
        requests.append("q" if line.strip().lower() == "q" else "save")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Save real camera frames (calibration board, ArUco shots).")
    ap.add_argument("host", nargs="?", help="robot IP, 192.168.1.2XX (default: host in config/robot_real.json)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "hardware" / "calib",
                    help="folder for the PNG frames (default results/hardware/calib)")
    ap.add_argument("--every", type=float, default=None,
                    help="also save automatically every EVERY s while the whole board is in view")
    ap.add_argument("--n", type=int, default=None, help="stop after saving this many frames in this run")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    n_saved = len(list(args.out.glob("*.png")))
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(aruco.ARUCO_DICT))
    with RealRobot(args.host) as robot:
        print(f"connected to {robot.host}, camera {robot.camera.width}x{robot.camera.height}; "
              f"saving to {args.out} ({n_saved} frames there already).\n"
              "Save: SPACE in the camera window, or Enter here. Quit: Q in the window, or q + Enter here.")
        requests = []
        threading.Thread(target=terminal_keys, args=(requests,), daemon=True).start()
        t_last, t_saved, n_new, save, quit_ = None, -1e9, 0, False, False
        while (args.n is None or n_new < args.n) and not quit_:
            key = cv2.waitKey(10) & 0xFF                  # keys pressed between frames count too
            while requests:
                r = requests.pop(0)
                save, quit_ = save or r == "save", quit_ or r == "q"
            save = save or key == ord(" ")
            quit_ = quit_ or key in (ord("q"), 27)
            t, rgb = robot.read_image()
            if t == t_last:
                continue
            t_last = t
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(gray, GRIDSHAPE, flags=cv2.CALIB_CB_FAST_CHECK)
            _, ids, _ = det.detectMarkers(gray)
            view = bgr.copy()
            cv2.drawChessboardCorners(view, GRIDSHAPE, corners, found)
            ids_txt = "none" if ids is None else " ".join(str(i) for i in sorted(ids.ravel()))
            cv2.putText(view, f"saved {n_saved}   board {'FOUND' if found else '-'}   ArUco {ids_txt}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if found or ids is not None else (0, 0, 255), 2)
            cv2.imshow("robot camera (SPACE save, Q quit)", view)
            auto = args.every is not None and found and t - t_saved >= args.every
            if save or auto:
                save = False
                path = args.out / f"frame_{time.strftime('%H%M%S')}_{n_saved:03d}.png"
                cv2.imwrite(str(path), bgr)
                n_saved, n_new, t_saved = n_saved + 1, n_new + 1, t
                print(f"saved {path.name}  (board {'found' if found else 'NOT found'}, ArUco {ids_txt})")
    cv2.destroyAllWindows()
    print(f"{n_saved} frames in {args.out}")


if __name__ == "__main__":
    main()
