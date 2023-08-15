from flask import Flask, request, render_template, jsonify, session
from flask_session import Session
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from holistic import *
import random
import string


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
tracker = Tracker()

def generate_session_key(length=16):
    characters = string.ascii_letters + string.digits
    session_key = ''.join(random.choice(characters) for _ in range(length))
    return session_key


@app.route('/receive_image', methods=['POST'])
def receive_image():
    session_key = session.get('session_key', None)

    if session_key is None:
        session_key = generate_session_key()  # Örnek bir oturum anahtarı oluşturma
        session['session_key'] = session_key

    image_data = request.form['imageData']  # FormData'dan gelen 'imageData' alanı
    image_data = image_data.split(',')[1]  # Veri URI kısmını ayır
    image = base64.b64decode(image_data)
    image = Image.open(BytesIO(image))
    # Burada image ile yapmak istediğin işlemleri yapabilirsin
    
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    frame, results = tracker.mediapipe_connection(image)
    tracker.draw_styled_landmarks(frame, results)
    leftHandLms, rightHandLms, poseLms = tracker.find_positions(frame, results)
    issue, _ = tracker.correction(leftHandLms, rightHandLms, poseLms)

    return jsonify({"issue": issue})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
