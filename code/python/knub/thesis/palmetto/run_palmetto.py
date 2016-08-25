import argparse
import numpy as np
import re
import subprocess
import os


def parse_iteration(s):
    res = re.search("\\.(\\d\\d\\d)\\.", s)
    if not res:
        print "WARN: Could not parse iteration"
        return 0
    iteration = int(res.group(1))
    return iteration


def parse_lambda(s):
    res = re.search("lambda-(0-\\d+)", s)
    if res:
        _lambda = float(res.group(1).replace("-", "."))
        return _lambda
    else:
        return None


def parse_topic_coherence(stdout):
    topic_coherences = []

    for line in stdout.splitlines():
        m = re.search("[01]\.\d{5}", line)
        if m:
            tc = float(m.group(0))
            topic_coherences.append(tc)

    # assert len(topic_coherences) in {2 ** i for i in range(1, 10)}, str(
    #     len(topic_coherences)) + " is not a power of two"
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


def calculate_topic_coherences(f):
    palmetto_file = create_palmetto_file(f)
    p = subprocess.Popen(
        ["java",
         "-jar",
         "/home/stefan.bunk/Palmetto/target/Palmetto-jar-with-dependencies.jar",
         "/data/wikipedia/2016-06-21/palmetto/wikipedia_bd",
         "C_V",
         palmetto_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = p.communicate()
    os.remove(palmetto_file)
    error = p.returncode
    if error:
        print stderr
        print "-" * 100
        print stdout
        return RuntimeError("Could not compute topics")
    else:
        topic_coherences = parse_topic_coherence(stdout)
        return topic_coherences


def main():
    parser = argparse.ArgumentParser("Evaluating word2vec with analogy task")
    parser.add_argument("topic_files", type=str, nargs="+")
    args = parser.parse_args()

    for topic_file in args.topic_files:
        iteration = parse_iteration(topic_file)
        _lambda = parse_lambda(topic_file)
        topic_coherences = calculate_topic_coherences(topic_file)
        if _lambda:
            print "%s\t%d\t%.5f\t%.3f\t%.3f" % (topic_file, iteration, _lambda, np.mean(topic_coherences), np.std(topic_coherences))
        else:
            print "%s\t%d\t%.3f\t%.3f" % (topic_file, iteration, np.mean(topic_coherences), np.std(topic_coherences))


if __name__ == "__main__":
    main()
