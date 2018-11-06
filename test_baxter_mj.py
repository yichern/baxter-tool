#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np

# model loading
model = load_model_from_path("simulation/envs/assets/baxter/reach.xml")
sim = MjSim(model)
viewer = MjViewer(sim)

# get original state (for looping back)
sim_state = sim.get_state()

# debugging
print(sim.get_state())
print(sim.data.mocap_pos)

# check qpos names and index
names = ["head_pan",
"right_s0",
"right_s1",
"right_e0",
"right_e1",
"right_w0",
"right_w1",
"right_w2",
"r_gripper_l_finger_joint",
"r_gripper_r_finger_joint",
"left_s0",
"left_s1",
"left_e0",
"left_e1",
"left_w0",
"left_w1",
"left_w2",
"l_gripper_l_finger_joint",
"l_gripper_r_finger_joint"]
for i, name in enumerate(names):
    sim.data.set_joint_qpos(name, i)
print(sim.get_state())


# simulation
while True:

    # reset to original state
    sim.set_state(sim_state)

    # time loop
    # sim.forward()
    # viewer.render()
    # position = np.array([0, 0, 0])
    # position = np.array([-0.3, 0.3, 0.0])
    # sim.data.set_mocap_pos("mocap", position)
    for i in range(2000):
        # new mocap position
        # if i < 150:
        #     sim.data.ctrl[:] = 0.0
        # else:
        #     sim.data.ctrl[:] = -1.0
        sim.step()
        viewer.render()

    if os.getenv('TESTING') is not None:
        break
