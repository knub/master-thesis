import argparse
import logging

import mkl
from gensim.models.word2vec import Word2Vec
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    mkl.set_num_threads(3)

    parser = argparse.ArgumentParser("Create topics from word2vec model")
    parser.add_argument("embedding_model", type=str)
    parser.add_argument("topic_model", type=str)
    args = parser.parse_args()

    assert "topic" in args.topic_model, "'%s' not a topic model" % args.topic_model
    assert "ssv" in args.topic_model, "need .ssv file for topic model"
    assert "embedding" in args.embedding_model, "'%s' not an embedding model" % args.embedding_model

    word2vec = Word2Vec.load_word2vec_format(args.embedding_model, binary=True)

    with open(args.topic_model, "r") as input:
        with open(args.embedding_model + ".ssv", "w") as output:
            for line in input:
                if "topic-count" not in line: # skip first header line
                    split = line.split(" ")
                    best_topic_word = split[2] # first is topic, second is topic count, third is first word
                    most_similars = word2vec.most_similar([best_topic_word], topn=9)
                    most_similars = map(lambda t: t[0], most_similars)
                    most_similars = map(lambda s: s.encode('utf8'), most_similars)
                    most_similars.insert(0, best_topic_word)
                    output.write("%s\n" % " ".join(most_similars))
