import argparse
import logging

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert word2vec model from binary to txt")
    parser.add_argument("model", type=str)
    args = parser.parse_args()

    model = Word2Vec.load_word2vec_format(args.model, binary=True)
    model.save_word2vec_format(args.model + ".txt", fvocab=args.model + ".vocab", binary=False)

    logging.info(model.most_similar(positive=['woman', 'king'], negative=['man']))
    logging.info(model.doesnt_match("breakfast cereal dinner lunch".split()))
    logging.info(model.similarity('woman', 'man'))

