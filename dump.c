regex for qpos:
^([\s]*)\<joint(.*)name
function for qpos:
get_joint_qpos(name)

fetch: robot.xml
MjSimState(time=0.0, qpos=array([0.  , 0.  , 0.  , 0.  , 0.  , 0.06, 0.  , 0.  , 0.  , 0.  , 0.  ,
       0.  , 0.  , 0.  , 0.  ]), qvel=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), act=None, udd_state={})
[[0. 0. 0.]]
qpos (name)
names = ["robot0:slide0",
"robot0:slide1",
"robot0:slide2",
"robot0:torso_lift_joint",
"robot0:head_pan_joint",
"robot0:head_tilt_joint",
"robot0:shoulder_pan_joint",
"robot0:shoulder_lift_joint",
"robot0:upperarm_roll_joint",
"robot0:elbow_flex_joint",
"robot0:forearm_roll_joint",
"robot0:wrist_flex_joint",
"robot0:wrist_roll_joint",
"robot0:r_gripper_finger_joint",
"robot0:l_gripper_finger_joint"]

baxter: robot.xml
MjSimState(time=0.0, qpos=array([0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0.6, 0. , 0. , 1. , 0. , 0. , 0. ]), qvel=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.]), act=None, udd_state={})
[[ 0.37   0.1   -0.065]]
qpos (name)
names = ["right_s0",
"right_s1",
"right_e0",
"right_e1",
"right_w0",
"right_w1",
"right_w2",
"r_gripper_l_finger_joint",
"r_gripper_r_finger_joint",
"left_s0",
"left_s1",
"left_e0",
"left_e1",
"left_w0",
"left_w1",
"left_w2",
"l_gripper_l_finger_joint",
"l_gripper_r_finger_joint"]











    top
    leg1
    leg2
    leg3
    leg4
    box
    box
