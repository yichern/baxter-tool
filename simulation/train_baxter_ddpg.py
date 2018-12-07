import gym
import numpy as np
import sys
import os
import fire

from tqdm import tqdm

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg import DDPG
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

def evaluate(model, env, num_steps=1000):
    episode_rewards = [0.0]
    obs = env.reset()
    for i in tqdm(range(num_steps)):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)

    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward

def main(env, load, save_path, load_path=None, train_timesteps=1.25e6, eval_timesteps=5e3):

    # arguments
    print("env %s; load %s; save_path %s; load_path %s; train_timesteps %s; eval_timesteps %s;" %
          (env, load, save_path, load_path, train_timesteps, eval_timesteps))
    train_timesteps = int(float(train_timesteps))
    eval_timesteps = int(float(eval_timesteps))

    # models path
    model_dir = os.getcwd() + "/models/"
    os.makedirs(model_dir, exist_ok=True)

    # logging path
    log_dir = os.getcwd() + "/log/" + save_path
    os.makedirs(log_dir, exist_ok=True)

    # absolute save path and models path
    save_path = model_dir + save_path
    if load and not load_path:
        print("no load path given, exiting...")
        sys.exit()
    elif load:
        load_path = model_dir + load_path

    # make environment, flattened environment, monitor, vectorized environment
    env = gym.make(env)
    env = gym.wrappers.FlattenDictWrapper(env, ['observation', 'achieved_goal', 'desired_goal'])
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # load model, or start from scratch
    if load:
        print("loading model from: " + load_path)
        model = DDPG.load(load_path, env=env)
    else:
        print("training model from scratch")
        model = DDPG(MlpPolicy, env, verbose=1)

    # evaluate current model
    mean_reward_before_train = evaluate(model, env, num_steps=eval_timesteps)

    # train model
    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0
    model.learn(total_timesteps=train_timesteps, callback=None)

    # save model
    print("saving model to:" + save_path)
    model.save(save_path)

    # evaluate post training model
    mean_reward_after_train = evaluate(model, env, num_steps=eval_timesteps)

    # results
    print("reward before training:" + str(mean_reward_before_train))
    print("reward after training:" + str(mean_reward_after_train))
    print("done")

if __name__ == "__main__":
    fire.Fire(main)
