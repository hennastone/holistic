from flask import Flask, request, render_template, jsonify
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from holistic import *


app = Flask(__name__)


@app.route('/receive_image', methods=['POST'])
def receive_image():
    tracker = Tracker()

    image_data = request.form['imageData']  # FormData'dan gelen 'imageData' alani
    image_data = image_data.split(',')[1]  # Veri URI kısmını ayır
    image = base64.b64decode(image_data)
    image = Image.open(BytesIO(image))
    # Burada image ile yapmak istediğin işlemleri yapabilirsin
    
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    frame, results = tracker.mediapipe_connection(image)
    tracker.draw_styled_landmarks(frame, results)
    leftHandLms, rightHandLms, poseLms = tracker.find_positions(frame, results)
    issues= tracker.correction(leftHandLms, rightHandLms, poseLms)

    return jsonify({"issues": issues})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
