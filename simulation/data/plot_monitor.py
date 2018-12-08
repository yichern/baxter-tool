import gym
import numpy as np
import sys
import os
import fire
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ppo2 import PPO2

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

def plot_results(fig_path, load_path1, load_path2, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """

    log_path1 = os.getcwd() + "/log/" + load_path1
    log_path2 = os.getcwd() + "/log/" + load_path2
    fig_path = os.getcwd() + "/figs/" + "/" + fig_path

    x, y = [], []
    timesteps = 0
    with open(log_path1 + "/" + "monitor.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0 or i == 1:
                continue
            y.append(float(row[0]))
            x.append(timesteps + int(row[1]))
            timesteps += int(row[1])
    with open(log_path2 + "/" + "monitor.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0 or i == 1:
                continue
            y.append(float(row[0]))
            x.append(timesteps + int(row[1]))
            timesteps += int(row[1])
    x = np.array(x)
    y = np.array(y)

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

    dotted = 1.25e6
    plt.axvline(x=dotted, color='k', linestyle='--')

    fig_path = fig_path + ".png"
    print("saving figure in: " + fig_path)
    plt.savefig(fig_path)

def main(fig_path, load_path1, load_path2):
    plot_results(fig_path, load_path1, load_path2)

if __name__ == "__main__":
    fire.Fire(main)
