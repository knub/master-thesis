import sys


def main():
    args = sys.argv[1:]

    for file_name in args:
        lines = []
        with open(file_name, "r") as f:
            for line in f:
                lines.append(" ".join(sorted(line.rstrip().split(" ")[1:6])) + "\n")

        with open(file_name + ".intrusion", "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()
