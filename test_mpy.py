import mujoco_py
from os.path import dirname
model = mujoco_py.load_model_from_path("mujoco-py/xmls/claw.xml")
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)
sim.step()
print(sim.data.qpos)
