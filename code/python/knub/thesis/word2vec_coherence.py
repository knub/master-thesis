import argparse
import logging
from math import factorial
import random
import sys
from itertools import combinations

import mkl
from gensim.models.word2vec import Word2Vec

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    mkl.set_num_threads(3)

    parser = argparse.ArgumentParser("Create topics from word2vec model")
    parser.add_argument("--embedding-model", type=str)
    parser.add_argument("--topic-model", type=str)
    args = parser.parse_args()

    assert "topic" in args.topic_model, "'%s' not a topic model" % args.topic_model
    assert "embedding" in args.embedding_model, "'%s' not an embedding model" % args.embedding_model

    print "Loading word2vec"
    word2vec = Word2Vec.load_word2vec_format(args.embedding_model, binary=True)

    # with open(args.topic_model + ".ssv", "r") as input:
    #     with open(args.embedding_model + ".ssv", "w") as output:
    #         for line in input:
    #             if "topic-count" not in line: # skip first header line
    #                 split = line.split(" ")
    #
    #                 i = 2
    #                 word_found = False
    #                 while not word_found:
    #                     try:
    #                         best_topic_word = split[2] # first is topic, second is topic count, third is first word
    #                         most_similars = word2vec.most_similar([best_topic_word], topn=9)
    #                         most_similars = map(lambda t: t[0], most_similars)
    #                         most_similars = map(lambda s: s.encode('utf8'), most_similars)
    #                         most_similars.insert(0, best_topic_word)
    #                         output.write("%s\n" % " ".join(most_similars))
    #                         word_found = True
    #                     except:
    #                         i += 1

    # frequent_words = [line.rstrip('\n') for line in open("/data/wikipedia/2016-06-21/vocab.txt")]
    # frequent_words = frequent_words[10000:-40000]
    # print "%d frequent words" % len(frequent_words)
    # random_sample = [frequent_words[i] for i in sorted(random.sample(xrange(len(frequent_words)), 100000))]
    #
    # with open(args.embedding_model + ".similars", "w") as output:
    #     for word in random_sample:
    #         try:
    #             similars = word2vec.most_similar([word], topn=3)
    #             for similar, prob in similars:
    #                 output.write("%s\t%s\t%s\n" % (word, similar, prob))
    #         except:
    #             pass

    print "Loading words"
    words = [line.rstrip('\n') for line in open(args.topic_model + ".vocab")]
    nr_words = len(words)
    print str(nr_words) + " words"
    N = factorial(nr_words) / (2 * factorial(nr_words - 2))
    output_every = N / 1000
    print str(N) + " pairs"

    print "Unknown words:"
    unknown_words = 0
    for word in words:
        if word not in word2vec:
            unknown_words += 1
    print unknown_words

    print "Calculating similarities"
    with open(args.topic_model + ".similarities", "w") as f:
        for i, (word1, word2) in enumerate(combinations(words, r=2)):
            if i % output_every == 0:
                print str(i * 100 / N) + " %",
                sys.stdout.flush()
            try:
                sim = word2vec.similarity(word1, word2)
                f.write(word1 + "\t" + word2 + "\t" + str(sim))
            except KeyError as e:
                print e
