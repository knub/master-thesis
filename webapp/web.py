from flask import Flask
from flask import render_template
from collections import namedtuple
import random
from random import shuffle

import os


Intrusion = namedtuple('Intrusion', ['method', 'id', 'topic_id', 'words', 'intrusion'])


def read_all_intrusions():
    intrusions_in_files = []
    for file_name in sorted(os.listdir("out/word-intrusion")):
        method = file_name.replace(".txt", "")
        with open("out/word-intrusion/%s" % file_name) as f:
            for line in f:
                line = line.rstrip().split("\t")
                intrusions_in_files.append(Intrusion(method, int(line[0]), int(line[1]), line[-6:-1], line[-1]))
    return intrusions_in_files


intrusions = read_all_intrusions()
random.seed(21011991)
shuffle(intrusions)


app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('hello.html')


@app.route("/topic/<int:intrusion_id>")
def intrusion(intrusion_id):
    if intrusion_id >= len(intrusions):
        return "Thank you, you tagged all word intrusion sets!"
    else:
        intrusion = intrusions[intrusion_id]
        words = [intrusion.intrusion] + intrusion.words
        random.seed(21011991)
        shuffle(words)
        return render_template('topic.html', words=words)

if __name__ == "__main__":
    app.run()
