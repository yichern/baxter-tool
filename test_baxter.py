import numpy as np
import gym
import simulation

env = gym.make('BaxterReachEnv-v1')
while True:
    pos_delta = np.random.rand(3)
    # pos_delta = np.array([0.5, 0.5, 0.5])
    # quat_delta = np.zeros(4)
    # gripper_ctrl = np.array([0,0])
    # action = np.concatenate([pos_delta, quat_delta, gripper_ctrl])
    # env.step(action)
    env.render()
