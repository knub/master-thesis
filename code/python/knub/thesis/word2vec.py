import argparse
import codecs
import logging
import os

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

    # base_path = "/home/stefan.bunk/master-thesis/data/outlier-detection/8-8-8_Dataset/"
    # [(_, _, files)] = os.walk(base_path)
    # words = []
    # for f in files:
    #     with open(base_path + f, "r") as f:
    #         for line in f.readlines():
    #             line = line.rstrip()
    #             if line:
    #                 words.append(line)
    #
    # print words
    # _, vocab = Phrases.learn_vocab(sentences, max_vocab_size=40000000)
    # delimiter = "_"
    # for word in words:
    #     split = word.split("_")
    #     if len(split) == 2:
    #         word_a, word_b = split
    #
    #         bigram_word = delimiter.join((word_a, word_b))
    #         if bigram_word in vocab:
    #             pa = float(vocab[word_a])
    #             pb = float(vocab[word_b])
    #             pab = float(vocab[bigram_word])
    #             score = (pab - 20) / pa / pb * len(vocab)
    #             is_bigram = score > 40
    #
    #             print "%s_%s %f = %s" % (word_a, word_b, score, str(is_bigram))


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
    model = Word2Vec(sentences, size=args.dimensions, window=5, min_count=5, workers=args.threads, sg=True, hs=0,
                     negative=10, sample=0.001, iter=40)
    model.save_word2vec_format(args.model, binary=True, fvocab=args.model + ".counts")
    logging.info("Finished training word2vec")

    if "nips" in args.sentences:
        for word in ["paper", "cortex", "brain", "learning", "posterior", "neural", "section", "optimization"]:
            try:
                logging.info(word)
                logging.info(model.most_similar([word]))
            except:
                pass
    else:
        logging.info(model.most_similar(positive=["woman", "king"], negative=["man"]))
        logging.info(model.doesnt_match("breakfast cereal dinner lunch".split()))
        logging.info(model.similarity('woman', 'man'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training word2vec with gensim")
    parser.add_argument("mode", type=str)
    parser.add_argument("sentences", type=str)
    parser.add_argument("dimensions", type=int)
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
