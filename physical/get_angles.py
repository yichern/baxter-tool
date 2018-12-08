#!/usr/bin/env python

import rospy
import baxter_interface

rospy.init_node('Hello_Baxter')
print(baxter_interface.Limb('right').joint_angles())
