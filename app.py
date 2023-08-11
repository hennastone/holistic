from flask import Flask, request, render_template
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from playsound import playsound
from holistic import *


app = Flask(__name__)
tracker = Tracker()

def play_sound(issue):
    try:
        playsound('sounds/' + str(issue) + '.wav', False)
    except Exception as e:
        print("Ses çalma hatası:", e)

@app.route('/receive_image', methods=['POST'])
def receive_image():
    image_data = request.form['imageData']  # FormData'dan gelen 'imageData' alanı
    image_data = image_data.split(',')[1]  # Veri URI kısmını ayır
    image = base64.b64decode(image_data)
    image = Image.open(BytesIO(image))
    # Burada image ile yapmak istediğin işlemleri yapabilirsin
    
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)

    frame, results = tracker.mediapipe_connection(image)
    tracker.draw_styled_landmarks(frame, results)
    leftHandLms, rightHandLms, poseLms = tracker.find_positions(frame, results)
    issue, _ = tracker.correction(leftHandLms, rightHandLms, poseLms)
    
    if issue != 0:
        play_sound(issue)


    return 'Görüntü alındı ve işlendi'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
