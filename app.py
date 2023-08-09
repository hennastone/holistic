from flask import Flask, render_template, Response
from holistic import *
import pygame
import time

app = Flask(__name__)

cap = cv2.VideoCapture(0)
tracker = Tracker()

pygame.mixer.init()
sound_channel = pygame.mixer.Channel(0)

last_issue = None

def play_sound(issue):
    try:
        sound_channel.stop()  # Önceki ses çalıyorsa durdur
        sound = pygame.mixer.Sound('sounds/' + str(issue) + '.wav')
        sound_channel.play(sound)
    except Exception as e:
        print("Ses çalma hatası:", e)

def generate_frames():
    last_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, results = tracker.mediapipe_connection(frame)
            tracker.draw_styled_landmarks(frame, results)
            leftHandLms, rightHandLms, poseLms = tracker.find_positions(frame, results)
            issue, prev_issue = tracker.correction(leftHandLms, rightHandLms, poseLms)

            current_time = time.time()

            if current_time - last_time > 5:
                if issue != prev_issue and issue != 0:
                    play_sound(issue)
                    last_time = current_time

            frame = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8080)

