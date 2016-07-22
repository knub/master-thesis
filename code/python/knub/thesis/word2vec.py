import argparse
import codecs
import logging

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training word2vec with gensim")
    parser.add_argument("sentences", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("threads", type=int)
    args = parser.parse_args()

    logging.info("Training word2vec")
    sentences = LineSentence(args.sentences)
    bigram_model = Phrases(sentences, 20, 40)

    with codecs.open(args.sentences + ".bigram", "w", encoding="utf-8") as f:
        for tokens in bigram_model[sentences]:
            f.write(" ".join(tokens) + "\n")

    # sentences = bigram_model[sentences]
    # trigram_model = Phrases(sentences, 5, 10)
    # model = Word2Vec(trigram_model[sentences], size=256, window=5, min_count=50, workers=args.threads, sg=True, hs=0, negative=10, sample=0.001)
    # model.save_word2vec_format(args.model, binary=True)
    # logging.info("Finished training word2vec")
    #
    # logging.info(model.most_similar(positive=['woman', 'king'], negative=['man']))
    # logging.info(model.doesnt_match("breakfast cereal dinner lunch".split()))
    # logging.info(model.similarity('woman', 'man'))

