import cv2
import mediapipe as mp
import numpy as np

class Tracker():
    def __init__(self, mode=False, smooth = True, maxHands = 2, detectionCon = 0.5, modelComplexity = 1, trackCon = 0.5):
        self.mode = mode
        self.smooth = smooth
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplexity = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                            smooth_landmarks=self.smooth,
                            min_detection_confidence=self.detectionCon,
                            min_tracking_confidence=self.trackCon)
        self.hands  = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image
    
    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for Id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([Id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmlist
    
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def getPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for Id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([Id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    tracker = Tracker()

    while True:
        success, hands = cap.read()
        hands = tracker.handsFinder(hands)
        lmList_h = tracker.positionFinder(hands)
        if len(lmList_h) != 0:
            print(lmList_h[4])

        cv2.imshow("hand tracker", hands)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()


