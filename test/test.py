from holistic import *
import math

def calculate_angle(a, b):
    radian = math.atan2(a[2]-b[2], a[1]-b[1])
    degree = math.degrees(radian)
    return abs(degree)

tracker = Tracker()

image = cv2.imread("./3.jpg")

image, results = tracker.mediapipe_connection(image)

tracker.draw_styled_landmarks(image, results)
leftHandLms, rightHandLms, poseLms = tracker.find_positions(image, results, draw = True)
print("sağ omuz - sağ dirsek ->", calculate_angle(poseLms[14], poseLms[12]))
print("sol omuz - sağ omuz ->", calculate_angle(poseLms[11], poseLms[12]))
print("burun - sağ bilek ->", calculate_angle(poseLms[0], rightHandLms[0]))
print("sol omuz - sol dirsek ->", calculate_angle(poseLms[11], poseLms[13]))
print("sağ bilek - sağ orta parmak ->", calculate_angle(rightHandLms[0], rightHandLms[9]))
"""
print("sol dirsek - sol bilek ->", calculate_angle(poseLms[13], leftHandLms[0]))
print("sol bilek - sol orta parmak ->", calculate_angle(leftHandLms[0], leftHandLms[9]))

"""

cv2.imshow('OpenCV Feed', image)
cv2.waitKey(0)