
from mujoco.viewer import launch


import mujoco


# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("otherfiles/humanoid/human_ab.xml")

data = mujoco.MjData(model)

with launch(model, data) as viewer:
        print("")

