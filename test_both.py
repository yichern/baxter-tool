import gym
env = gym.make('Humanoid-v2')
env.reset()
while True:
    env.render()
env.close()
