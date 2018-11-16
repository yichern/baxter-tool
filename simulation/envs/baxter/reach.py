from gym import utils
from simulation.envs import baxter_env

class BaxterReachEnv(baxter_env.BaxterEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'right_s0': -0.97752925707998,
            'right_s1': -0.6059224112147384,
            'right_e0': 2.1609954349335765,
            'right_e1': 2.1705828148578603,
            'right_w0': 0.770441850715449,
            'right_w1': 0.9545195452616987,
            'right_w2': -1.4933302970064504,
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
        baxter_env.BaxterEnv.__init__(
            self, 'baxter/reach.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
