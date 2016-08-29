import argparse
from codecs import open
import logging
import os
import sys

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare model for Gaussian LDA")
    parser.add_argument("--vocabulary", type=str)
    parser.add_argument("--embedding-model", type=str)
    args = parser.parse_args()

    word2vec = Word2Vec.load_word2vec_format(args.embedding_model, binary=True)
    topic_folder = os.path.dirname(args.vocabulary)
    vocab_name = os.path.basename(args.vocabulary)
    embedding_name = os.path.basename(args.embedding_model)
    with open(topic_folder + "/" + embedding_name + "." + vocab_name.replace("." + embedding_name, "") + ".embedding",
              "w", encoding="utf-8") as output:
        with open(args.vocabulary, "r", encoding="utf-8") as f:
            for line in f:
                word = line.rstrip()
                if word in word2vec:
                    pass
                elif word.capitalize() in word2vec:
                    word = word.capitalize()
                elif word.upper() in word2vec:
                    word = word.upper()
                try:
                    output.write(" ".join(map(str, word2vec[word])))
                except KeyError:
                    print word
                    output.write(" ".join(map(str, word2vec["house"])))
                output.write("\n")
