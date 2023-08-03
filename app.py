from flask import Flask, render_template, Response
from holistic import *

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video_feed():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)