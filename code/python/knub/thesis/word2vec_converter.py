import argparse
import logging
import os
from codecs import open

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert word2vec model from binary to txt")
    parser.add_argument("model", type=str)
    parser.add_argument("--vocabulary", type=str)
    args = parser.parse_args()

    model = Word2Vec.load_word2vec_format(args.model, binary=True)

    if args.vocabulary is None:
        print "No vocabulary set, converting all words"
        model.save_word2vec_format(args.model + ".txt", binary=False)
    else:
        print "Vocabulary is set, using only words from " + args.vocabulary
        vocab_name = os.path.basename(args.vocabulary)
        with open(args.model + "." + vocab_name + ".model.txt", "w", encoding="utf-8") as output:
            with open(args.vocabulary, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.rstrip()
                    try:
                        output.write(word + " ")
                        output.write(" ".join(map(str, model[word])))
                    except KeyError:
                        pass
                    output.write("\n")

    logging.info(model.most_similar(positive=['woman', 'king'], negative=['man']))
    logging.info(model.doesnt_match("breakfast cereal dinner lunch".split()))
    logging.info(model.similarity('woman', 'man'))

