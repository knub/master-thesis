import numpy as np
import re
import subprocess
import os


def parse_topic_coherence(stdout):
    topic_coherences = []

    for line in stdout.splitlines():
        m = re.search("[01]\.\d{5}", line)
        tc = float(m.group(0))
        topic_coherences.append(tc)

    assert len(topic_coherences) in {2 ** i for i in range(1, 10)}, str(
        len(topic_coherences)) + " is not a power of two"
    return topic_coherences


def main():
    ssv_files = [file for file in os.listdir("/data/wikipedia/2016-06-21/topic-models") if file.endswith(".ssv")]

    for ssv_file in ssv_files:
        print ssv_file
        p = subprocess.Popen(
            ["java",
             "-jar",
             "/home/stefan.bunk/Palmetto/target/Palmetto-jar-with-dependencies.jar",
             "/data/wikipedia/2016-06-21/palmetto/wikipedia_bd",
             ssv_file],
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
            print "Mean:", np.mean(topic_coherences)
            print "Variance:", np.var(topic_coherences)
            print "Raw:", topic_coherences


if __name__ == "__main__":
    main()
