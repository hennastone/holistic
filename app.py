from flask import Flask
from holistic import *

app = Flask(__name__)

@app.route("/")
def index():
    main()
    return "Mete Gazoz"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)