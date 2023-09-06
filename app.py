from flask import Flask, request, render_template, jsonify
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from holistic import *


app = Flask(__name__)

@app.route('/receive_image', methods=['POST'])
def receive_image():
    image_data = request.form['imageData'] 
    image_data = image_data.split(',')[1]  
    image = base64.b64decode(image_data)
    image = Image.open(BytesIO(image))    
    
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    tracker = Tracker()

    frame, results = tracker.mediapipe_connection(image)
    #tracker.draw_styled_landmarks(frame, results)
    leftHandLms, rightHandLms, poseLms = tracker.find_positions(frame, results)
    issues = tracker.correction(leftHandLms, rightHandLms, poseLms)

    tracker.close()
    del tracker

    return jsonify({"issues": issues})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def main():
    app.run(port=8080, debug=True)

if __name__ == '__main__':
    main()