import argparse
import numpy as np
import re
import subprocess
import os


def parse_topic_coherence(stdout):
    topic_coherences = []

    for line in stdout.splitlines():
        m = re.search("[01]\.\d{5}", line)
        if m:
            tc = float(m.group(0))
            topic_coherences.append(tc)

    assert len(topic_coherences) in {2 ** i for i in range(1, 10)}, str(
        len(topic_coherences)) + " is not a power of two"
    return topic_coherences


def create_palmetto_file(topic_file):
    palmetto_file = topic_file + ".palmetto"
    with open(topic_file, "r") as input:
        with open(palmetto_file, "w") as output:
            for line in input:
                split = line.split(" ")
                if "topic-count" not in line: # skip first line
                    new_line = " ".join(split[-10:])
                    output.write(new_line)
    return palmetto_file

def main():
    parser = argparse.ArgumentParser("Evaluating word2vec with analogy task")
    parser.add_argument("topic_files", type=str, nargs="+")
    args = parser.parse_args()

    for topic_file in args.topic_files:
        palmetto_file = create_palmetto_file(topic_file)
        print topic_file
        p = subprocess.Popen(
            ["java",
             "-jar",
             "/home/stefan.bunk/Palmetto/target/Palmetto-jar-with-dependencies.jar",
             "/data/wikipedia/2016-06-21/palmetto/wikipedia_bd",
             "C_V",
             palmetto_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = p.communicate()
        error = p.returncode
        if error:
            print stderr
            print "-" * 100
            print stdout
            return RuntimeError("Could not compute topics")
        else:
            topic_coherences = parse_topic_coherence(stdout)
            print "%s\t%.3f\t%.3f" % (topic_file, np.mean(topic_coherences), np.std(topic_coherences))

        os.remove(palmetto_file)

if __name__ == "__main__":
    main()
