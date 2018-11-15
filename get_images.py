#!/usr/bin/env python

import time
import rospy
import tf
import math
from human_robot_collaboration_msgs.msg import GoToPose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
import baxter_interface
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

bridge = CvBridge()

imageType = 0

# orientations
out = Quaternion()
out.x = 0.0
out.y = 1.0
out.z = 0.0
out.w = 0.0
up = Quaternion()
up.x = -0.5
up.y = 0.5
up.z = 0.5
up.w = 0.5
left = Quaternion()
left.x = 0.0
left.y = 0.0
left.z = -0.7
left.w = 0.7
right = Quaternion()
right.x = 0.0
right.y = 0.0
right.z = 0.7
right.w = 0.7

# positions
overTool = Point()
overTool.x = 0.5
overTool.y = -0.6
overTool.z = 0.4
atTool = Point()
atTool.x = 0.5
atTool.y = -0.6
atTool.z = -0.22
leftFace = Point()
leftFace.x = 0.35
leftFace.y = 0.15
leftFace.z = 0.8
rightFace = Point()
rightFace.x = 0.35
rightFace.y = -0.15
rightFace.z = 0.8
downFace = Point()
downFace.x = 0.35
downFace.y = 0.0
downFace.z = 0.65

pose = GoToPose()
pose.type = 'pose'
pose.ctrl_mode = 1
pose.orientation = out
pose.position = overTool
pose.check_mode = 'strict'

def find_length(img_name):

    img = cv2.imread(img_name)
    img = img[:680, :]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blur = cv2.bilateralFilter(hsv, 11, 17, 17)

    lower = np.array([6.0, 50, 50], dtype = "uint8")
    upper = np.array([13.0, 255, 255], dtype = "uint8")


    mask = cv2.inRange(blur, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)


    x,y,w,h = cv2.boundingRect(cnts[0])
    temp = cv2.rectangle(hsv,(x,y),(x+w,y+h),(0,255,0),2)

    x1,y1,w1,h1 = cv2.boundingRect(cnts[1])
    temp = cv2.rectangle(temp,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)


    scalar = 17.5/514

    # #for debugging - can view image
    # cv2.imshow("image", temp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if '1' in img_name:
        #claw on right
        length = (x+w) - x1
        #print(length)
        return length*scalar

    elif '2' in img_name:
        length = (x1+w1) - x
        #print(length)
        return length*scalar

    elif '3' in img_name:
        length = (y+h) - y1
        #print(length)
        return length*scalar
    else:
        print("error")

def adjusted_pose(p, l):
    # fancy math to determine axis of offset
    q1 = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    q2 = [1,0,0,0]
    offset = tf.transformations.quaternion_multiply(tf.transformations.quaternion_multiply(q1,q2),tf.transformations.quaternion_conjugate(q1))[:3]
    offset *= l
    print(offset)
    p.position.x += offset[0]
    p.position.y += offset[1]
    p.position.z += offset[2]

    return pose

def talker():
    global imageType, pose
    pub = rospy.Publisher('/baxter_controller/right/go_to_pose', GoToPose, queue_size=1)
    rospy.init_node('get_images', anonymous=True)
    #calibrate
    intermediate = Point() # helps solve IC hiccups
    rightHand = baxter_interface.Gripper('right')
    rightHand.calibrate()
    time.sleep(1)

    pose.orientation = out
    pose.position = overTool
    rospy.loginfo(pose)
    pub.publish(pose)
    time.sleep(3)

    pose.position = atTool
    rospy.loginfo(pose)
    pub.publish(pose)
    time.sleep(5)

    rightHand.close()
    intermediate = overTool
    intermediate.x += .2
    intermediate.y += .4
    intermediate.z += .2

    pose.position = intermediate
    pose.orientation = up
    pub.publish(pose)
    time.sleep(7)

    pose.position = rightFace
    pose.orientation = left
    rospy.loginfo(pose)
    pub.publish(pose)
    imageType = 1
    rospy.Subscriber('cameras/head_camera/image', Image, image_callback)
    time.sleep(7)
    imageType = 0

    pose.position = leftFace
    pose.orientation = right
    rospy.loginfo(pose)
    pub.publish(pose)
    imageType = 2
    time.sleep(10)
    imageType = 0

    pose.position = downFace
    pose.orientation = up
    rospy.loginfo(pose)
    pub.publish(pose)
    imageType = 3
    time.sleep(5)
    imageType = 0

    offset = (find_length('view_1.jpg')+find_length('view_2.jpg')+find_length('view_3.jpg'))/300

    pose = adjusted_pose(pose, offset)
    rospy.loginfo(pose)
    pub.publish(pose)
    time.sleep(3)

def image_callback(msg):
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        if imageType == 1:
            cv2.imwrite('view_1.jpg', cv2_img)
        elif imageType == 2:
            cv2.imwrite('view_2.jpg', cv2_img)
        elif imageType == 3:
            cv2.imwrite('view_3.jpg', cv2_img)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
