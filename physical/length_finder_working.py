import cv2
import numpy as np

def find_length(img_name):

	img = cv2.imread(img_name)
	img = img_long[:600, :]
	
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	blur = cv2.bilateralFilter(hsv, 11, 17, 17)

	lower = np.array([6.0, 50, 50], dtype = "uint8")
	upper = np.array([13.0, 255, 255], dtype = "uint8")


	mask = cv2.inRange(blur, lower, upper)

	_, contours, _= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(contours, key = cv2.contourArea, reverse = True)


	x,y,w,h = cv2.boundingRect(cnts[0])
	temp = cv2.rectangle(hsv,(x,y),(x+w,y+h),(0,255,0),2)

	x1,y1,w1,h1 = cv2.boundingRect(cnts[1])
	temp = cv2.rectangle(temp,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)


	scalar = 17.5/514

	if '1' in img_name:
		#claw on right
		length = (x+w) - x1
		print(length)
		return length*scalar

	elif '2' in img_name:
		length = (x1+w1) - x
		print(length)
		return length*scalar

	elif '3' in img_name:
		length = (y+h) - y1
		print(length)
		return length*scalar
	else:
		print("error")

	#for debugging - can view image
	# cv2.imshow("image", temp)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

