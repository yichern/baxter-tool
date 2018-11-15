import numpy as np
import gym
import simulation

env = gym.make('BaxterReachEnv-v1')
while True:
    grip_pos = env._get_obs()['observation'][:3]
    goal = env._get_obs()['desired_goal']
    action = grip_pos - goal
    # print(action)
    # action = np.array(action)
    action = np.random.rand(4)
    # print(grip_pos)
    env.step(action)
    env.render()
