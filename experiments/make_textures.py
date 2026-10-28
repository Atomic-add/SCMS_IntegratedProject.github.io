"""Generate the image textures used by the scenes.

    python experiments/make_textures.py

Writes into models/robotis_tb3/assets/:
  aruco_4x4_50_id{0,1,2}.png  dock marker (id 0) and landmarks (ids 1, 2): W3 marker_image
                            with a 768 px marker (6 cells of 128 px) and a 128 px (one cell) white
                            quiet zone, 1024 x 1024 px in total, so each 0.1333 m sheet in the scenes
                            carries a 0.10 m marker.
                            Why 1024: the GPU shows a distant marker from a halved copy of the texture
                            (mipmap). With a power-of-two size every copy is an exact half and the marker
                            edges stay on texel boundaries. With the W3 default (600 + 2 x 60 = 720 px) the
                            small copies are 45, 22, 11 px, the edges move by part of a texel, and the
                            measured range was 2-3 times more biased beyond 1.5 m (evaluation/check_camera.py).
  checkerboard_9x6_25mm.png calibration board: 9 x 6 internal corners, 60 px squares + 60 px margin,
                            so the 0.30 x 0.225 m plane in scene_calib.xml has 25 mm squares. It is only seen
                            from 0.35-0.55 m, where the GPU uses the full-size or half-size copy.

Adapted from 41014 Week 3 E (marker_image).
Additions: power-of-two textures, so mipmaps keep the marker edges on texel boundaries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so "src" imports work

import cv2                                                      # noqa: E402

from src.perception.aruco import marker_image                   # noqa: E402
from src.perception.camera import checkerboard_image            # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "models" / "robotis_tb3" / "assets"

if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    images = [(f"aruco_4x4_50_id{i}.png", marker_image(i, px=768, pad=128)) for i in (0, 1, 2)]
    for name, img in images + [("checkerboard_9x6_25mm.png", checkerboard_image())]:
        cv2.imwrite(str(ASSETS / name), img)
        print(f"wrote {name}: {img.shape[1]} x {img.shape[0]} px")