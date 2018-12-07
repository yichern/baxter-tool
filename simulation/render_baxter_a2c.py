import gym
import numpy as np
import sys
import os
import fire
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.a2c import A2C

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(fig_path, log_path, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_path), 'timesteps')
    y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]
    print(x.shape)
    print(y.shape)

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")

    fig_path = fig_path + ".png"
    print("saving figure in: " + fig_path)
    plt.savefig(fig_path)

def main(env, load_path, fig_path):

    # arguments
    print("env %s; load_path %s; fig_path %s;" % (env, load_path, fig_path))
    log_path = os.getcwd() + "/log/" + load_path
    os.makedirs(os.getcwd() + "/figs/" + "/", exist_ok=True)
    fig_path = os.getcwd() + "/figs/" + "/" + fig_path
    load_path = os.getcwd() + "/models/" + load_path

    # make environment, flattened environment, vectorized environment
    env = gym.make(env)
    env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    env = DummyVecEnv([lambda: env])

    # load model
    model = A2C.load(load_path, env=env)
    obs_initial = env.reset()
    obs = obs_initial

    # plot results
    plot_results(fig_path, log_path)

    # initializations
    niter = 10
    counter = 0
    timestep = 0
    results = [[[0,0,0] for i in range(100)], [[0,0,0,0] for i in range(100)]]
    current = [[[0,0,0] for i in range(100)], [[0,0,0,0] for i in range(100)]]
    print("==============================")

    # check initial positions and quaternions
    print("grip", env.envs[0].env.env.sim.data.get_site_xpos('grip'))
    print("box", env.envs[0].env.env.sim.data.get_site_xpos('box'))
    print("tool", env.envs[0].env.env.sim.data.get_site_xpos('tool'))
    print("mocap", env.envs[0].env.env.sim.data.mocap_pos)
    print("quat", env.envs[0].env.env.sim.data.mocap_quat)
    print("==============================")

    # mocap quaternion check
    for i in range(5):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        quat = env.envs[0].env.env.sim.data.mocap_quat
        print("obs", obs)
        print("quat", quat)
    print("==============================")

    # start rendering
    dists = []
    box_goal_pos = np.array([0.6, 0.05, -0.17])
    while True:
        if counter == niter:
            break
        action, _states = model.predict(obs)
        obs_old = obs
        obs, rewards, dones, info = env.step(action)
        quaternion = env.envs[0].env.env.sim.data.mocap_quat
        if obs.all() == obs_initial.all():
            if counter % 10 == 0:
                xyzs = current[0]
                quats = current[1]
                print(xyzs)
                print(quats)
                filename = log_path + "/" + "results_" + str(counter) + ".txt"
                os.makedirs(log_path + "/", exist_ok=True)
                file = open(filename, 'w+')
                for xyz, quat in zip(xyzs, quats):
                    for coord in xyz:
                        file.write(str(coord) + " ")
                    for quat_coord in quat:
                        file.write(str(quat_coord) + " ")
                    file.write("\n")
                file.close()

            box_end_pos = np.array(obs_old[0][3:6].tolist())
            print(box_end_pos)
            print(np.shape(box_end_pos))
            print(box_goal_pos)
            print(np.shape(box_goal_pos))
            dists.append(np.linalg.norm(box_goal_pos - box_end_pos))
            current = [[[0,0,0] for i in range(100)], [[0,0,0,0] for i in range(100)]]
            timestep = 0
            counter += 1
        print(timestep)
        print("obs", obs)
        print("quat", quaternion)

        # for average trajectory, smoothed
        for i in range(3):
            results[0][timestep][i] += obs[0][:3].tolist()[i]
        for j in range(4):
            results[1][timestep][j] += quaternion[0].tolist()[j]

        # for current trajectory
        for i in range(3):
            current[0][timestep][i] += obs[0][:3].tolist()[i]
        for j in range(4):
            current[1][timestep][j] += quaternion[0].tolist()[j]

        timestep += 1
        env.render()

    # smooth paths by taking average, and calculate mean distance to goal state
    for timestep in range(100):
        for i in range(3):
            results[0][timeste][i] /= niter
        for j in range(4):
            results[0][timestep][j] /= niter
    dist = np.mean(dists)

    # print and write to file
    xyzs = results[0]
    quats = results[1]
    filename = log_path + "/" + "results_avg.txt"
    os.makedirs(log_path + "/", exist_ok=True)
    file = open(filename, 'w+')
    for xyz, quat in zip(xyzs, quats):
        for coord in xyz:
            file.write(str(coord) + " ")
        for quat_coord in quat:
            file.write(str(quat_coord) + " ")
        file.write("\n")
    file.close()

    # print average distances
    print("average distance of box from end goal: %f" % dist)

if __name__ == "__main__":
    fire.Fire(main)
