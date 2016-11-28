with open("lda_topic_evolution.ssv", "r") as f:
    lines = f.readlines()
    start = 0
    while start + 50 < len(lines):
        group = lines[start:start+50]
        with open("%04d.txt" % (5 * start / 50), "w") as o:
            o.writelines(group)
        start += 50
