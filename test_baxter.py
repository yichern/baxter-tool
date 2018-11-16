import numpy as np
import gym
import simulation

env = gym.make('BaxterReachEnv-v1')
target = np.array([0.6, 0.0, 0.0])
while True:
    # observed = env._get_obs()
    # current_pos = observed['observation'][:3]
    # deltas = target - current_pos
    # action = np.array(deltas)
    # action = np.append(action, [0.0])
    # env.step(action)
    env.render()
