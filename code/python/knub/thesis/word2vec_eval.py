import argparse
import logging
import sys
from collections import namedtuple

import mkl
from gensim.models.word2vec import Word2Vec


def eval_word2vec(word2vec):
    f = open("../../../../data/analogy-reasoning/questions-words.txt")
    AnalogyTask = namedtuple("AnalogyTask", "plus1 minus1 plus2 answer")
    analogy_tasks = []
    for l in f:
        if not l.startswith(":"):
            split = l.split(" ")
            if "GoogleNews" in model:
                analogy_tasks.append(AnalogyTask(split[1], split[0],
                                                 split[2], split[3].rstrip()))
            else:
                analogy_tasks.append(AnalogyTask(split[1].lower(), split[0].lower(),
                                                 split[2].lower(), split[3].rstrip().lower()))
    correct_count = 0
    task_count = 0
    for task in analogy_tasks:
        if task_count % 1000 == 0:
            print task_count,
            sys.stdout.flush()

        # s = "%s - %s + %s = %s" % (task.plus1, task.minus1, task.plus2, task.answer)

        most_similars = word2vec.most_similar(positive=[task.plus1, task.plus2], negative=[task.minus1], topn=4)
        ignore_words = {task.plus1, task.plus2, task.minus1}
        most_similar = next((word, prob) for word, prob in most_similars if word not in ignore_words)

        task_count += 1
        if most_similar[0] == task.answer:
            correct_count += 1
    return float(correct_count) / task_count


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    mkl.set_num_threads(3)

    parser = argparse.ArgumentParser("Evaluating word2vec with analogy task")
    parser.add_argument("model", type=str, nargs="+")
    args = parser.parse_args()

    for model in args.model:
        print model
        word2vec = Word2Vec.load_word2vec_format(model, binary=True)
        res = eval_word2vec(word2vec)
        print
        print res
