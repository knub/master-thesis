from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('hello.html')
    # return "Hey, you already tagged xxx topics. Continue?"


@app.route("/topic/<int:intrusion_id>")
def intrusion(intrusion_id):
    return render_template('hello.html')
    return "Hey, you already tagged xxx topics. Continue?"

if __name__ == "__main__":
    app.run()
