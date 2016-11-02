# coding=utf-8
import argparse
import logging
import os
import string
from codecs import open

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import dot, float32 as REAL, \
    array, ndarray

from gensim import matutils
from six import string_types
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

exclude = set(string.punctuation).union({u"“", u"’", u"”", "."})
def fix(s):
    s = ''.join(ch for ch in s if ch not in exclude)
    return s.lower()

def most_similar_all(word2vec, positive=[], negative=[], topn=10, restrict_vocab=None):
    word2vec.init_sims()

    if isinstance(positive, string_types) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in positive
        ]
    negative = [
        (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in negative
        ]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, ndarray):
            mean.append(weight * word)
        elif word in word2vec.vocab:
            mean.append(weight * word2vec.syn0norm[word2vec.vocab[word].index])
            all_words.add(word2vec.vocab[word].index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    limited = word2vec.syn0norm if restrict_vocab is None else word2vec.syn0norm[:restrict_vocab]
    dists = dot(limited, mean)
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # ignore (don't return) words from the input
    result = [(word2vec.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
    # do return words from the input
    # result = [(word2vec.index2word[sim], float(dists[sim])) for sim in best]
    return result[:topn]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--embedding-model", type=str)
    parser.add_argument("--topic-model", type=str)
    args = parser.parse_args()

    model = Word2Vec.load_word2vec_format(args.embedding_model, binary=True)
    print " ".join([w for w, _ in model.most_similar(["queen"])])

    embedding_name = os.path.basename(args.embedding_model)

    with open(args.topic_model + "." + embedding_name + ".raw-we-tm.head", "w", encoding="utf-8") as topics:
        with open(args.topic_model, "r") as f:
            for line in f:
                if "topic-count" not in line:
                    word_found = False
                    i = -10
                    while not word_found:
                        word = line.rstrip().split(" ")[i]
                        word_found = word in model
                        i += 1

                    similar_words = [fix(w) for w, _ in most_similar_all(model, [word], topn=30)]
                    used = []
                    similar_words = [x for x in similar_words if x not in used and (used.append(x) or True)]
                    topic_line = " ".join(similar_words[:10])
                    print word + " --> " + topic_line

                    topics.write(topic_line)
                    topics.write("\n")

    with open(args.topic_model + "." + embedding_name + ".raw-we-tm.avg", "w", encoding="utf-8") as topics:
        with open(args.topic_model, "r") as f:
            for line in f:
                if "topic-count" not in line:
                    words = line.rstrip().split(" ")[-10:]
                    words = [word for word in words if word in model]

                    similar_words = [fix(w) for w, _ in most_similar_all(model, words, topn=30)]
                    used = []
                    similar_words = [x for x in similar_words if x not in used and (used.append(x) or True)]
                    topic_line = " ".join(similar_words[:10])
                    print " ".join(words) + " --> " + topic_line

                    topics.write(topic_line)
                    topics.write("\n")
