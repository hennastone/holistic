import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)  #Draw Face Landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  #Draw Pose Landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  #Draw Left Hand Landmarks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  #Draw Right Hand Landmarks

def draw_styled_landmarks(image, results):
    #Draw Face Landmarks
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color = (80, 110, 10), thickness = 1, circle_radius = 1),
        mp_drawing.DrawingSpec(color = (80, 256, 121), thickness = 1, circle_radius = 1))
    
    #Draw Pose Landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color = (80, 22, 10), thickness = 2, circle_radius = 4),
        mp_drawing.DrawingSpec(color = (80, 44, 121), thickness = 2, circle_radius = 2))
    
    #Draw Left Hand Landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color = (121, 22, 76), thickness = 2, circle_radius = 4),
        mp_drawing.DrawingSpec(color = (121, 44, 250), thickness = 2, circle_radius = 2))
    
    #Draw Right Hand Landmarks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color = (245, 117, 66), thickness = 2, circle_radius = 4),
        mp_drawing.DrawingSpec(color = (245, 66, 230), thickness = 2, circle_radius = 2))

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():

        success, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)

        draw_styled_landmarks(image, results)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()