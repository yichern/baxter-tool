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
print(sim.get_state())

# mocap position
position = np.array([0.6, 0.0, 0.0])
sim.data.set_mocap_pos("mocap", position)
print(sim.data.mocap_pos)

# simulation
while True:

    # # time loop
    # sim.forward()
    # viewer.render()
    #
    # for i in range(2000):
        # new mocap position
        # if i < 150:
        #     sim.data.ctrl[:] = 0.0
        # else:
        #     sim.data.ctrl[:] = -1.0
    sim.step()
    viewer.render()

    if os.getenv('TESTING') is not None:
        break
