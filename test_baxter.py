import numpy as np
import gym
import simulation

env = gym.make('BaxterReachEnv-v1')
while True:
    action = np.random.rand(4)
    env.step(action)
    env.render()
