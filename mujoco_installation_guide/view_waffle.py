import sys
from pathlib import Path

import mujoco
import mujoco.viewer

# Find the model relative to this file, so it works from any folder (and from VS Code's Run button)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE = PROJECT_ROOT / "models" / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"

if not SCENE.exists():
    sys.exit(
        f"Model not found: {SCENE}\n"
        "Get the latest team repository first (setup guide, step 9):\n"
        "    git pull"
    )

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

mujoco.viewer.launch(model, data)