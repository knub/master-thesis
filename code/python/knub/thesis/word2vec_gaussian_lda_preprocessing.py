import argparse
import logging
import os

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare model for Gaussian LDA")
    parser.add_argument("--topic_model", type=str)
    parser.add_argument("--embedding_model", type=str)
    args = parser.parse_args()

    word2vec = Word2Vec.load_word2vec_format(args.embedding_model, binary=True)
    embedding_name = os.path.basename(args.embedding_model)
    with open(args.topic_model + "." + embedding_name + ".gaussian-lda", "w") as output:
        with open(args.topic_model + ".restricted.alphabet", "r") as f:
            for line in f:
                word = line.split("#")[0]
                output.write(word + " ")
                output.write(" ".join(word2vec[word]))
                output.write("\n")
