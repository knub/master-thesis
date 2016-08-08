import argparse
import codecs
import logging

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def bigrams():
    logging.info("Training bigrams")
    sentences = LineSentence(args.sentences)
    bigram_model = Phrases(sentences, 20, 40)

    with codecs.open(args.sentences + ".bigram", "w", encoding="utf-8") as f:
        for tokens in bigram_model[sentences]:
            f.write(" ".join(tokens) + "\n")


def trigrams():
    logging.info("Training trigrams")
    sentences = LineSentence(args.sentences + ".bigram")
    bigram_model = Phrases(sentences, 20, 40)

    with codecs.open(args.sentences + ".trigram", "w", encoding="utf-8") as f:
        for tokens in bigram_model[sentences]:
            f.write(" ".join(tokens) + "\n")


def word2vec():
    logging.info("Training word2vec")
    sentences = LineSentence(args.sentences)
    model = Word2Vec(sentences, size=200, window=5, min_count=10, workers=args.threads, sg=True, hs=0,
                     negative=10, sample=0.001, iter=5)
    model.save_word2vec_format(args.model, binary=True, fvocab=args.model + ".counts")
    logging.info("Finished training word2vec")
    logging.info(model.most_similar(positive=['woman', 'king'], negative=['man']))
    logging.info(model.doesnt_match("breakfast cereal dinner lunch".split()))
    logging.info(model.similarity('woman', 'man'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training word2vec with gensim")
    parser.add_argument("mode", type=str)
    parser.add_argument("sentences", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("threads", type=int)
    args = parser.parse_args()

    if args.mode == "word2vec":
        word2vec()
    elif args.mode == "bigrams":
        bigrams()
    elif args.mode == "trigrams":
        trigrams()
    else:
        print "Nothing done."
