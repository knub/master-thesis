from flask import Flask
from flask import render_template, request
from collections import namedtuple
import random
from random import shuffle

import os


Intrusion = namedtuple('Intrusion', ['method', 'id', 'topic_id', 'words', 'intruder'])

RESULTS = "out/results.txt"

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
    if os.path.exists(RESULTS):
        with open(RESULTS, "r") as f:
            intrusion_idxs = {int(line.rstrip().split("\t")[0]) for line in f}

        # print intrusion_idxs
        # print set(range(len(intrusions))) - intrusion_idxs
        min_untagged_idxs = set(range(len(intrusions))) - intrusion_idxs
        if len(min_untagged_idxs) > 0:
            min_untagged_idx = min(set(range(len(intrusions))) - intrusion_idxs)
            text = "Continue tagging with sample %d/%d" % (min_untagged_idx, len(intrusions))
            intrusion_idx = min_untagged_idx
        else:
            text = "You already tagged everything"
            intrusion_idx = len(intrusions)
    else:
        text = "Start tagging 0/%d" % len(intrusions)
        intrusion_idx = 0
    return render_template('hello.html', text=text, id=intrusion_idx)


@app.route("/topic/<int:intrusion_idx>", methods=["GET", "POST"])
def tag_intrusion(intrusion_idx):
    if request.method == "POST":
        words = request.form["words"]
        method = request.form["method"]
        intruder = request.form["intruder"]
        selected_word = request.form["selected_word"]
        tagged_intrusion_id = request.form["intrusion_id"]
        old_intrusion_idx = request.form["intrusion_idx"]
        with open(RESULTS, mode="a") as f:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" %
                    (old_intrusion_idx, tagged_intrusion_id, method, words, intruder, selected_word, str(intruder == selected_word)))
        print "%s %s: %s - %s ### %s" % (tagged_intrusion_id, method, words, intruder, selected_word)

    if intrusion_idx >= len(intrusions):
        return render_template('goodbye.html')
    else:
        intrusion = intrusions[intrusion_idx]
        words = [intrusion.intruder] + intrusion.words
        random.seed(21011991)
        shuffle(words)
        return render_template('topic.html',
                               words=words,
                               method=intrusion.method,
                               current_id=intrusion_idx,
                               intruder=intrusion.intruder,
                               intrusion_id=intrusion.id)

if __name__ == "__main__":
    app.run()
