import argparse
from datetime import datetime
import numpy as np
import re
import subprocess
from functools import partial
from multiprocessing import Pool
import sys

from knub.thesis.util import *


def parse_topic_coherence(stdout):
    topic_coherences = []

    for line in stdout.splitlines():
        m = re.search("[01]\.\d{5}", line)
        if m:
            tc = float(m.group(0))
            topic_coherences.append(tc)

    return topic_coherences


def create_palmetto_file(topic_file, best_k):
    palmetto_file = topic_file + "." + str(best_k) + ".palmetto"
    with open(topic_file, "r") as input:
        with open(palmetto_file, "w") as output:
            for line in input:
                split = line.split(" ")
                if "topic-count" not in line: # skip first line
                    new_line = " ".join(split[-100:(-100 + best_k)])
                    output.write(new_line.rstrip() + "\n")
    return palmetto_file


def calculate_topic_coherences(f, best_k):
    palmetto_file = create_palmetto_file(f, best_k)
    local_palmetto = "/home/knub/Repositories/Palmetto/target/Palmetto-jar-with-dependencies.jar"
    remote_palmetto = "/home/stefan.bunk/Palmetto/target/Palmetto-jar-with-dependencies.jar"
    local_wikidata = "/home/knub/Repositories/Palmetto/wikipedia_bd"
    remote_wikidata = "/data/wikipedia/2016-06-21/palmetto/wikipedia_bd"
    if os.path.exists(local_palmetto):
        p = subprocess.Popen(
            ["java",
             "-jar",
             local_palmetto,
             local_wikidata,
             "C_V",
             palmetto_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif os.path.exists(remote_palmetto):
        p = subprocess.Popen(
            ["java",
             "-jar",
             remote_palmetto,
             remote_wikidata,
             "C_V",
             palmetto_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        raise Exception("Palmetto not found")


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
        # print stdout
        # print map(lambda x: x[0], sorted(enumerate(topic_coherences), key=lambda x: x[1]))
        return topic_coherences


def calculate_line(best_k, topic_file):
    params_str = "\t".join(parse_params(topic_file).values())
    topic_coherences = calculate_topic_coherences(topic_file, best_k)
    if params_str:
        line = "%d\t%.3f\t%.3f" % (best_k, np.mean(topic_coherences), np.std(topic_coherences))
        print line
        return line
    else:
        line = "%d\t%.3f\t%.3f" % (best_k, np.mean(topic_coherences), np.std(topic_coherences))
        print line
        return line


def main():
    parser = argparse.ArgumentParser("Evaluating word2vec with analogy task")
    parser.add_argument("threads", type=int)
    parser.add_argument("topic_file", type=str)
    args = parser.parse_args()

    now = datetime.now()

    print now.strftime("%a, %Y-%m-%d %H:%M")

    if not os.path.isfile(args.topic_file):
        print "ERROR: file <%s> does not exist" % str(args.topic_file)
        sys.exit(1)

    best_ks = range(19, 50)
    if args.threads == 1:
        for r in best_ks:
            calculate_line(r, args.topic_file)
    else:
        p = Pool(args.threads)
        try:
            f = partial(calculate_line, topic_file=args.topic_file)
            r = list(best_ks)
            p.map(f, r)
        finally:
            p.close()


if __name__ == "__main__":
    main()
