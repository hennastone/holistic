import cv2
from mediapipe.python.solutions import holistic, drawing_utils
import math


class Tracker():
    def __init__(self, detectionCon = 0.5, trackCon = 0.5, modelComplexity = 1):
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplexity = modelComplexity
        self.drawing = drawing_utils
        self.holistic = holistic
        self.model = self.holistic.Holistic(
                                    min_detection_confidence = self.detectionCon, 
                                    min_tracking_confidence = self.trackCon,
                                    model_complexity = self.modelComplexity)

    
    def mediapipe_connection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    
    def draw_styled_landmarks(self, image, results):
        #Draw Pose Landmarks
        self.drawing.draw_landmarks(
            image, results.pose_landmarks, self.holistic.POSE_CONNECTIONS,
            self.drawing.DrawingSpec(color = (80, 22, 10), thickness = 1, circle_radius = 2),
            self.drawing.DrawingSpec(color = (80, 44, 121), thickness = 1, circle_radius = 1))

        #Draw Left Hand Landmarks
        self.drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.holistic.HAND_CONNECTIONS,
            self.drawing.DrawingSpec(color = (121, 22, 76), thickness = 1, circle_radius = 2),
            self.drawing.DrawingSpec(color = (121, 44, 250), thickness = 1, circle_radius = 1))

        #Draw Right Hand Landmarks
        self.drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.holistic.HAND_CONNECTIONS,
            self.drawing.DrawingSpec(color = (245, 117, 66), thickness = 1, circle_radius = 2),
            self.drawing.DrawingSpec(color = (245, 66, 230), thickness = 1, circle_radius = 1))
          

    
    def find_positions(self, image, results):
        leftHandLms = []
        rightHandLms = []
        poseLms = []

        if results.left_hand_landmarks:
            for Id, lm in enumerate(results.left_hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                leftHandLms.append([Id, cx, cy])
        
        if results.right_hand_landmarks:
            for Id, lm in enumerate(results.right_hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                rightHandLms.append([Id, cx, cy])
        
        if results.pose_landmarks:
            for Id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                poseLms.append([Id, cx, cy])

        return leftHandLms, rightHandLms, poseLms
    
    def calculate_angle(self, a, b):
        radian = math.atan2(a[2]-b[2], a[1]-b[1])
        degree = math.degrees(radian)
        return abs(degree)
    
    def correction(self, leftHandLms, rightHandLms, poseLms):
        issues = []
        if len(leftHandLms) != 0 and len(rightHandLms) != 0 and len(poseLms) != 0:

            if(self.calculate_angle(poseLms[14], poseLms[12]) > 160):
                issues.append(1)

            elif(self.calculate_angle(poseLms[14], poseLms[12]) < 140):
                issues.append(2)

            elif(self.calculate_angle(poseLms[11], poseLms[12]) > 5):
                issues.append(3)


            elif(abs(self.calculate_angle(poseLms[11], poseLms[13]) - 180) > 10):
                issues.append(5)

            elif(abs(self.calculate_angle(poseLms[13], leftHandLms[0]) - 180) > 10):
                issues.append(6)

            elif(abs(self.calculate_angle(leftHandLms[0], leftHandLms[9]) - 170) > 10):
                issues.append(7)

            elif(abs(self.calculate_angle(rightHandLms[0], rightHandLms[9]) - 170) > 10):
                issues.append(8)
            
            else:
                issues = [-1]
        else:
            issues = [9]

        return issues

