import argparse
import logging

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training word2vec with gensim")
    parser.add_argument("sentences", type=file)
    parser.add_argument("model", type=str)
    parser.add_argument("threads", type=int)
    args = parser.parse_args()

    logging.info("Training word2vec")
    sentences = LineSentence(args.sentences)
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=args.threads, sg=True)
    model.save(args.model + ".skipgram")
    logging.info("Finished training word2vec")

    logging.info(model.most_similar(positive=['woman', 'king'], negative=['man']))
    logging.info(model.doesnt_match("breakfast cereal dinner lunch".split()))
    logging.info(model.similarity('woman', 'man'))

    logging.info("=" * 100)

    logging.info("Training word2vec")
    sentences = LineSentence(args.sentences)
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=args.threads, sg=True)
    model.save(args.model + ".cbow")
    logging.info("Finished training word2vec")

    logging.info(model.most_similar(positive=['woman', 'king'], negative=['man']))
    logging.info(model.doesnt_match("breakfast cereal dinner lunch".split()))
    logging.info(model.similarity('woman', 'man'))
