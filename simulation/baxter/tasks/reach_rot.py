import numpy as np
from gym import utils
from gym.envs.baxter import baxter_rot_env
from gym.envs.baxter import utils as utils2

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

# -d reward + rot_ctrl
class BaxterReachEnv_8(baxter_rot_env.BaxterRotEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            #holding up to camera
            # 'right_s0': -0.97752925707998,
            # 'right_s1': -0.6059224112147384,
            # 'right_e0': 2.1609954349335765,
            # 'right_e1': 2.1705828148578603,
            # 'right_w0': 0.770441850715449,
            # 'right_w1': 0.9545195452616987,
            # 'right_w2': -1.4933302970064504,

            # old starting pos
            # 'right_s0': -0.27650003701634585,
            # 'right_s1': -0.4206942310775747,
            # 'right_w0': -0.99171857936792,
            # 'right_w1': 0.5932670697146838,
            # 'right_w2': 0.07401457301547121,
            # 'right_e0': 0.5311408478053246,
            # 'right_e1': 1.6961992562042962,

            'right_s0': -0.3286553838044499,
            'right_s1': -0.5150340495325276,
            'right_w0': -0.9625729443980972,
            'right_w1': 0.656543777214957,
            'right_w2': 0.06442719309118737,
            'right_e0': 0.6082233823965666,
            'right_e1': 1.758708973310627,

            'r_gripper_l_finger_joint': 0.0,
            'r_gripper_r_finger_joint': 0.0,
            'left_s0': 0.0,
            'left_s1': 0.0,
            'left_e0': 0.0,
            'left_e1': 0.0,
            'left_w0': 0.0,
            'left_w1': 0.0,
            'left_w2': 0.0,
            'l_gripper_l_finger_joint': 0.0,
            'l_gripper_r_finger_joint': 0.0,
        }
        baxter_rot_env.BaxterRotEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

# if/else reward with summed penalty + rot_ctrl
class BaxterReachEnv_9(baxter_rot_env.BaxterRotEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            #holding up to camera
            # 'right_s0': -0.97752925707998,
            # 'right_s1': -0.6059224112147384,
            # 'right_e0': 2.1609954349335765,
            # 'right_e1': 2.1705828148578603,
            # 'right_w0': 0.770441850715449,
            # 'right_w1': 0.9545195452616987,
            # 'right_w2': -1.4933302970064504,

            # old starting pos
            # 'right_s0': -0.27650003701634585,
            # 'right_s1': -0.4206942310775747,
            # 'right_w0': -0.99171857936792,
            # 'right_w1': 0.5932670697146838,
            # 'right_w2': 0.07401457301547121,
            # 'right_e0': 0.5311408478053246,
            # 'right_e1': 1.6961992562042962,

            'right_s0': -0.3286553838044499,
            'right_s1': -0.5150340495325276,
            'right_w0': -0.9625729443980972,
            'right_w1': 0.656543777214957,
            'right_w2': 0.06442719309118737,
            'right_e0': 0.6082233823965666,
            'right_e1': 1.758708973310627,

            'r_gripper_l_finger_joint': 0.0,
            'r_gripper_r_finger_joint': 0.0,
            'left_s0': 0.0,
            'left_s1': 0.0,
            'left_e0': 0.0,
            'left_e1': 0.0,
            'left_w0': 0.0,
            'left_w1': 0.0,
            'left_w2': 0.0,
            'l_gripper_l_finger_joint': 0.0,
            'l_gripper_r_finger_joint': 0.0,
        }
        baxter_rot_env.BaxterRotEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        grip = self.sim.data.get_site_xpos('grip')
        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')

        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            if goal_distance(tool, box) > goal_distance(grip, box):
                return -(goal_distance(tool, box) + d)
            else:
                return -d

# if/else reward with constant penalty + rot_ctrl
class BaxterReachEnv_10(baxter_rot_env.BaxterRotEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            #holding up to camera
            # 'right_s0': -0.97752925707998,
            # 'right_s1': -0.6059224112147384,
            # 'right_e0': 2.1609954349335765,
            # 'right_e1': 2.1705828148578603,
            # 'right_w0': 0.770441850715449,
            # 'right_w1': 0.9545195452616987,
            # 'right_w2': -1.4933302970064504,

            # old starting pos
            # 'right_s0': -0.27650003701634585,
            # 'right_s1': -0.4206942310775747,
            # 'right_w0': -0.99171857936792,
            # 'right_w1': 0.5932670697146838,
            # 'right_w2': 0.07401457301547121,
            # 'right_e0': 0.5311408478053246,
            # 'right_e1': 1.6961992562042962,

            'right_s0': -0.3286553838044499,
            'right_s1': -0.5150340495325276,
            'right_w0': -0.9625729443980972,
            'right_w1': 0.656543777214957,
            'right_w2': 0.06442719309118737,
            'right_e0': 0.6082233823965666,
            'right_e1': 1.758708973310627,

            'r_gripper_l_finger_joint': 0.0,
            'r_gripper_r_finger_joint': 0.0,
            'left_s0': 0.0,
            'left_s1': 0.0,
            'left_e0': 0.0,
            'left_e1': 0.0,
            'left_w0': 0.0,
            'left_w1': 0.0,
            'left_w2': 0.0,
            'l_gripper_l_finger_joint': 0.0,
            'l_gripper_r_finger_joint': 0.0,
        }
        baxter_rot_env.BaxterRotEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        grip = self.sim.data.get_site_xpos('grip')
        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')

        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            if goal_distance(tool, box) > goal_distance(grip, box):
                penalty = 100
                return -penalty
            else:
                return -d

# summed reward + is_success change + rot_ctrl
class BaxterReachEnv_11(baxter_rot_env.BaxterRotEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            #holding up to camera
            # 'right_s0': -0.97752925707998,
            # 'right_s1': -0.6059224112147384,
            # 'right_e0': 2.1609954349335765,
            # 'right_e1': 2.1705828148578603,
            # 'right_w0': 0.770441850715449,
            # 'right_w1': 0.9545195452616987,
            # 'right_w2': -1.4933302970064504,

            # old starting pos
            # 'right_s0': -0.27650003701634585,
            # 'right_s1': -0.4206942310775747,
            # 'right_w0': -0.99171857936792,
            # 'right_w1': 0.5932670697146838,
            # 'right_w2': 0.07401457301547121,
            # 'right_e0': 0.5311408478053246,
            # 'right_e1': 1.6961992562042962,

            'right_s0': -0.3286553838044499,
            'right_s1': -0.5150340495325276,
            'right_w0': -0.9625729443980972,
            'right_w1': 0.656543777214957,
            'right_w2': 0.06442719309118737,
            'right_e0': 0.6082233823965666,
            'right_e1': 1.758708973310627,

            'r_gripper_l_finger_joint': 0.0,
            'r_gripper_r_finger_joint': 0.0,
            'left_s0': 0.0,
            'left_s1': 0.0,
            'left_e0': 0.0,
            'left_e1': 0.0,
            'left_w0': 0.0,
            'left_w1': 0.0,
            'left_w2': 0.0,
            'l_gripper_l_finger_joint': 0.0,
            'l_gripper_r_finger_joint': 0.0,
        }
        baxter_rot_env.BaxterRotEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        grip = self.sim.data.get_site_xpos('grip')
        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')

        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -(goal_distance(tool, box) + d)

    def _is_success(self, achieved_goal, desired_goal):

        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')
        d = goal_distance(achieved_goal, desired_goal)
        return ((goal_distance(tool, box) + d) < self.distance_threshold).astype(np.float32)

#changed goal to be farther
class BaxterReachEnv_12(baxter_rot_env.BaxterRotEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            #holding up to camera
            # 'right_s0': -0.97752925707998,
            # 'right_s1': -0.6059224112147384,
            # 'right_e0': 2.1609954349335765,
            # 'right_e1': 2.1705828148578603,
            # 'right_w0': 0.770441850715449,
            # 'right_w1': 0.9545195452616987,
            # 'right_w2': -1.4933302970064504,

            # old starting pos
            # 'right_s0': -0.27650003701634585,
            # 'right_s1': -0.4206942310775747,
            # 'right_w0': -0.99171857936792,
            # 'right_w1': 0.5932670697146838,
            # 'right_w2': 0.07401457301547121,
            # 'right_e0': 0.5311408478053246,
            # 'right_e1': 1.6961992562042962,

            'right_s0': -0.3286553838044499,
            'right_s1': -0.5150340495325276,
            'right_w0': -0.9625729443980972,
            'right_w1': 0.656543777214957,
            'right_w2': 0.06442719309118737,
            'right_e0': 0.6082233823965666,
            'right_e1': 1.758708973310627,

            'r_gripper_l_finger_joint': 0.0,
            'r_gripper_r_finger_joint': 0.0,
            'left_s0': 0.0,
            'left_s1': 0.0,
            'left_e0': 0.0,
            'left_e1': 0.0,
            'left_w0': 0.0,
            'left_w1': 0.0,
            'left_w2': 0.0,
            'l_gripper_l_finger_joint': 0.0,
            'l_gripper_r_finger_joint': 0.0,
        }
        baxter_rot_env.BaxterRotEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        grip = self.sim.data.get_site_xpos('grip')
        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')

        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -(goal_distance(tool, box) + d)

    def _sample_goal(self):
        return np.array([0.6, 0.15, -0.17])

    def _is_success(self, achieved_goal, desired_goal):

        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')
        d = goal_distance(achieved_goal, desired_goal)
        return ((goal_distance(tool, box) + d) < self.distance_threshold).astype(np.float32)

#changed target to be randomized
class BaxterReachEnv_15(baxter_rot_env.BaxterRotEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            #holding up to camera
            # 'right_s0': -0.97752925707998,
            # 'right_s1': -0.6059224112147384,
            # 'right_e0': 2.1609954349335765,
            # 'right_e1': 2.1705828148578603,
            # 'right_w0': 0.770441850715449,
            # 'right_w1': 0.9545195452616987,
            # 'right_w2': -1.4933302970064504,

            # old starting pos
            # 'right_s0': -0.27650003701634585,
            # 'right_s1': -0.4206942310775747,
            # 'right_w0': -0.99171857936792,
            # 'right_w1': 0.5932670697146838,
            # 'right_w2': 0.07401457301547121,
            # 'right_e0': 0.5311408478053246,
            # 'right_e1': 1.6961992562042962,

            'right_s0': -0.3286553838044499,
            'right_s1': -0.5150340495325276,
            'right_w0': -0.9625729443980972,
            'right_w1': 0.656543777214957,
            'right_w2': 0.06442719309118737,
            'right_e0': 0.6082233823965666,
            'right_e1': 1.758708973310627,

            'r_gripper_l_finger_joint': 0.0,
            'r_gripper_r_finger_joint': 0.0,
            'left_s0': 0.0,
            'left_s1': 0.0,
            'left_e0': 0.0,
            'left_e1': 0.0,
            'left_w0': 0.0,
            'left_w1': 0.0,
            'left_w2': 0.0,
            'l_gripper_l_finger_joint': 0.0,
            'l_gripper_r_finger_joint': 0.0,
        }
        baxter_rot_env.BaxterRotEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        grip = self.sim.data.get_site_xpos('grip')
        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')

        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -(goal_distance(tool, box) + d)

    def _sample_goal(self):
        x = np.random.uniform(low=-0.2, high=0.4)
        y = np.random.uniform(low=-0.05, high=0.1)
        return np.array([x+0.6, -0.3+y, -0.17])

    def _is_success(self, achieved_goal, desired_goal):

        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')
        d = goal_distance(achieved_goal, desired_goal)
        return ((goal_distance(tool, box) + d) < self.distance_threshold).astype(np.float32)

#changed start to be randomized
class BaxterReachEnv_16(baxter_rot_env.BaxterRotEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            #holding up to camera
            # 'right_s0': -0.97752925707998,
            # 'right_s1': -0.6059224112147384,
            # 'right_e0': 2.1609954349335765,
            # 'right_e1': 2.1705828148578603,
            # 'right_w0': 0.770441850715449,
            # 'right_w1': 0.9545195452616987,
            # 'right_w2': -1.4933302970064504,

            # old starting pos
            # 'right_s0': -0.27650003701634585,
            # 'right_s1': -0.4206942310775747,
            # 'right_w0': -0.99171857936792,
            # 'right_w1': 0.5932670697146838,
            # 'right_w2': 0.07401457301547121,
            # 'right_e0': 0.5311408478053246,
            # 'right_e1': 1.6961992562042962,

            'right_s0': -0.3286553838044499,
            'right_s1': -0.5150340495325276,
            'right_w0': -0.9625729443980972,
            'right_w1': 0.656543777214957,
            'right_w2': 0.06442719309118737,
            'right_e0': 0.6082233823965666,
            'right_e1': 1.758708973310627,

            'r_gripper_l_finger_joint': 0.0,
            'r_gripper_r_finger_joint': 0.0,
            'left_s0': 0.0,
            'left_s1': 0.0,
            'left_e0': 0.0,
            'left_e1': 0.0,
            'left_w0': 0.0,
            'left_w1': 0.0,
            'left_w2': 0.0,
            'l_gripper_l_finger_joint': 0.0,
            'l_gripper_r_finger_joint': 0.0,
        }
        baxter_rot_env.BaxterRotEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        grip = self.sim.data.get_site_xpos('grip')
        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')

        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -(goal_distance(tool, box) + d)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils2.reset_mocap_welds(self.sim)
        self.sim.forward()

        object_qpos = self.sim.data.get_joint_qpos('box')
        assert object_qpos.shape == (7,)
        x = np.random.uniform(low=-0.2, high=0.4)
        y = np.random.uniform(low=-0.05, high=0.1)
        object_qpos[:2] = [x+0.6,-0.3+y]
        self.sim.data.set_joint_qpos('box', object_qpos)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('box')[2]

    def _is_success(self, achieved_goal, desired_goal):

        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')
        d = goal_distance(achieved_goal, desired_goal)
        return ((goal_distance(tool, box) + d) < self.distance_threshold).astype(np.float32)

#randomize both start and end
class BaxterReachEnv_17(baxter_rot_env.BaxterRotEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            #holding up to camera
            # 'right_s0': -0.97752925707998,
            # 'right_s1': -0.6059224112147384,
            # 'right_e0': 2.1609954349335765,
            # 'right_e1': 2.1705828148578603,
            # 'right_w0': 0.770441850715449,
            # 'right_w1': 0.9545195452616987,
            # 'right_w2': -1.4933302970064504,

            # old starting pos
            # 'right_s0': -0.27650003701634585,
            # 'right_s1': -0.4206942310775747,
            # 'right_w0': -0.99171857936792,
            # 'right_w1': 0.5932670697146838,
            # 'right_w2': 0.07401457301547121,
            # 'right_e0': 0.5311408478053246,
            # 'right_e1': 1.6961992562042962,

            'right_s0': -0.3286553838044499,
            'right_s1': -0.5150340495325276,
            'right_w0': -0.9625729443980972,
            'right_w1': 0.656543777214957,
            'right_w2': 0.06442719309118737,
            'right_e0': 0.6082233823965666,
            'right_e1': 1.758708973310627,

            'r_gripper_l_finger_joint': 0.0,
            'r_gripper_r_finger_joint': 0.0,
            'left_s0': 0.0,
            'left_s1': 0.0,
            'left_e0': 0.0,
            'left_e1': 0.0,
            'left_w0': 0.0,
            'left_w1': 0.0,
            'left_w2': 0.0,
            'l_gripper_l_finger_joint': 0.0,
            'l_gripper_r_finger_joint': 0.0,
        }
        baxter_rot_env.BaxterRotEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        grip = self.sim.data.get_site_xpos('grip')
        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')

        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -(goal_distance(tool, box) + d)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils2.reset_mocap_welds(self.sim)
        self.sim.forward()

        object_qpos = self.sim.data.get_joint_qpos('box')
        assert object_qpos.shape == (7,)
        x = np.random.uniform(low=-0.2, high=0.4)
        y = np.random.uniform(low=-0.05, high=0.1)
        object_qpos[:2] = [x+0.6,-0.3+y]
        self.sim.data.set_joint_qpos('box', object_qpos)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('box')[2]

    def _sample_goal(self):
        x = np.random.uniform(low=-0.2, high=0.4)
        y = np.random.uniform(low=-0.05, high=0.1)
        return np.array([x+0.6, -0.3+y, -0.17])

    def _is_success(self, achieved_goal, desired_goal):

        tool = self.sim.data.get_site_xpos('tool')
        box = self.sim.data.get_site_xpos('box')
        d = goal_distance(achieved_goal, desired_goal)
        return ((goal_distance(tool, box) + d) < self.distance_threshold).astype(np.float32)
