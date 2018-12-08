# Teaching Baxter to Use Tools

This repository contains the final project deliverables for Team Baxter in CPSC 473 Intelligent Robotics Laboratory at Yale University. The deliverables include:

* Conference-style paper
* Slides from in-class presentation
* 1-slide summary of project
* 1-paragraph description of team member contributions

In addition, the repository contains source code developed under the project. Videos of Baxter can be found [here](https://photos.app.goo.gl/E8h28XfiDFvbprtV6).
___

## Physical Baxter

The directory `baxter-tool/physical/` contains the source code for work on the physical Baxter system.

Run `python simulation.py` to run the script for Baxter to execute tool length detection, offset calculation for kinematics extension, and action trajectory execution. This command must be done on the Baxter workspace at the ScazLab with Baxter turned on, untucked and ROS launched.

1. Turn on Baxter using the switch on the back.
2. In `/home/scazlab/ros_devel_ws`, run `roslaunch humann_robot_collaboration baxter_controller.launch`.
3. Using another shell, in the same directory, run `untuck`.
4. In that same shell, in the directory that contains `simulation.py`, run `python simulation.py`.

___

## Simulation Baxter

### 1. Installation

#### Anaconda
Before beginning, ensure that Anaconda is installed. You can install Anaconda [here](https://conda.io/docs/user-guide/install/index.html).

#### MuJoCo
1. You can obtain a 30-day free trial on the [MuJoCO website](https://www.roboti.us/license.html). If you are a student, you can obtain a free license.
2. Download MuJoCo version 1.50 binaries.
3. Unzip the downloaded `mjpro150` directory into `~/.mujoco/mjpro150`, and place the license key (`mjkey.txt`) at `~/.mujoco/mjkey.txt`.

#### Anaconda Environment
1. Run `conda env create --name environment.yml` to create the Baxter environment
2. Run `conda activate baxter` to activate the Baxter environment

Alternatively, you can follow the steps below to create your own Anaconda environment

#### mujoco-py
1. Clone the mujoco-py [repository](https://github.com/openai/mujoco-py).
2. Build from source by running `pip install -e .` in the mujoco-py directory

mujoco-py is notoriously difficult to install. You may need to add the following to your `.bashrc`:
* `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin/libglew.so`
* `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin`
* `export MUJOCO_PY_MJPRO_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150`

#### OpenAI Gym
1. Clone the Gym [repository](https://github.com/openai/gym).
2. Build from source by running `pip install -e .` in the Gym directory

#### OpenAI Baselines
We do not use OpenAI's baselines [repository](https://github.com/openai/baselines) per se. Rather, we use a fork of that repository, called [stable-baselines](https://github.com/hill-a/stable-baselines). We find that stable-baselines is more readable and easier to use. stable-baselines also has [documentation](http://stable-baselines.readthedocs.io/)

1. Follow the instructions to install stable-baselines at their [repository](https://github.com/openai/baselines).

#### baxter-tool
1. Fork this repository
2. Add `export PYTHONPATH="${PYTHONPATH}:[PATH TO THIS REPOSITORY]"` to your `.bashrc`
3. Add the directory `/baxter-tool/simulation/baxter` to the directory `/gym/gym/envs/` in your Gym installation.
4. Add the file `/baxter-tool/simulation/__init__.py` to the directory `/gym/gym/envs/` in your Gym installation, replacing the old `__init__.py`.

### 2. Baxter Model
* The Baxter MuJoCo model is defined in `baxter-tool/simulation/baxter/assets/baxter/`. The specific task model is defined in that directory as `reach.xml`, whereas `robot.xml` and `shared.xml` are relevant to any tasks environments.
* The general Baxter Gym environment is defined in `baxter-tool/simulation/baxter/baxter_env.py` and `baxter-tool/simulation/baxter/baxter_rot_env.py`. The former forbids learning rotation control of the gripper, while the latter includes learning rotation control.
* The specific Baxter Gym environment for our task is defined in `baxter-tool/simulation/baxter/tasks/reach.py` and `baxter-tool/simulation/baxter/tasks/reach_rot.py`. The former involves environments using `baxter_env.py` while the latter involves environments using `baxter_rot_env.py`.

If you would like to create a new task environment, first create the xml file in `baxter-tool/simulation/baxter/assets/baxter/`, then create the environment in a python file in `baxter-tool/simulation/baxter/tasks/`.

### 3. Trained Models

`baxter-tool/simulation/commands.txt` contains all the commands used to train, evaluate, render, and plot the models for conducting experiments. It also contains an explanation of each of the trained models we provide in `baxter-tool/simulation/models/`. You can render a trained model by running the corresponding command. You can also re-train a model by running the corresponding command.

### 4. Training and Rendering New Models

#### Training

You can train a new PPO model by running `train_baxter.py`. `train_baxter_a2c.py`, `train_baxter_trpo.py` and `train_baxter_ddpg.py` use A2C, TRPO and DDPG respectively for training. The model will be saved under `models/` and the training history will be saved under `logs/`, under the name specified.

Example command: `python train_baxter.py --env BaxterReachEnv-v11 --load False --save-path 11-baxter --train-timesteps 1.25e6 --eval-timesteps 5e3`

Usage:
```
[TrainFile].py --env [EnvName] --load [Load] --save-path [SavePath] --load-path [LoadPath] --train-timesteps [TrainSteps] -- eval-timesteps [EvalSteps]
```
- `TrainFile`: one of `train_baxter.py`. `train_baxter_a2c.py`, `train_baxter_trpo.py` and `train_baxter_ddpg.py`
- `EnvName`: the corresponding Baxter environment with which the model should be trained
- `Load`: whether a trained model should be loaded for further training, either `True` or `False`
- `SavePath`: the name to save the model under
- `LoadPath`: if `--load True`, the name of the model saved in `models/` for loading
- `TrainSteps`: the number of timesteps with which to train the model
- `EvalSteps`: the number of timesteps with which to evaluate the model

#### Rendering

You can render a trained PPO model by running `render_baxter.py`. `render_baxter_a2c.py`, `render_baxter_trpo.py` and `render_baxter_ddpg.py` use A2C, TRPO and DDPG respectively for rendering. The plot will be saved under `figs/`, under the name specified.

Example command: `python render_baxter.py --env BaxterReachEnv-v11 --load-path 11-baxter --fig-path 11-baxter`

Usage:
```
[RenderFile].py --env [EnvName] --load-path [ModelName] --fig-path [FigName]
```
- `RenderFile`: one of `render_baxter.py`. `render_baxter_a2c.py`, `render_baxter_trpo.py` or `render_baxter_ddpg.py`
- `EnvName`: the corresponding Baxter environment with which the model was trained on, or a different environment as desired
- `ModelName`: the name of the model saved in `models/`
- `FigName`: the desired name of the plot
