#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np
import gym

# model loading
model = load_model_from_path("../gym/gym/envs/robotics/assets/fetch/reach.xml")
sim = MjSim(model)
viewer = MjViewer(sim)

# get original state (for looping back)
sim_state = sim.get_state()

# debugging
print(sim.get_state())
print(sim.data.mocap_pos)

# check qpos names and index
names = ["robot0:slide0",
"robot0:slide1",
"robot0:slide2",
"robot0:torso_lift_joint",
"robot0:head_pan_joint",
"robot0:head_tilt_joint",
"robot0:shoulder_pan_joint",
"robot0:shoulder_lift_joint",
"robot0:upperarm_roll_joint",
"robot0:elbow_flex_joint",
"robot0:forearm_roll_joint",
"robot0:wrist_flex_joint",
"robot0:wrist_roll_joint",
"robot0:r_gripper_finger_joint",
"robot0:l_gripper_finger_joint"]
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
