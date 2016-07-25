import argparse
import codecs
import os
import string
from itertools import dropwhile
from random import shuffle

from nltk.tokenize import sent_tokenize, word_tokenize

parser = argparse.ArgumentParser("Training word2vec with gensim")
parser.add_argument("twentynews_folder", type=str)
args = parser.parse_args()


def get_filenames():
    files = []
    for dirpath, _, filenames in os.walk(args.twentynews_folder):
        if dirpath == args.twentynews_folder:
            continue
        for fname in filenames:
            files.append(dirpath + "/" + fname)
    shuffle(files)
    return files


filenames = get_filenames()

sentences_file = codecs.open(args.twentynews_folder + "/sentences.txt", "w", encoding="utf-8")
articles_file = codecs.open(args.twentynews_folder + "/articles.txt", "w", encoding="utf-8")

for filename in filenames:
    with codecs.open(filename, "r", encoding="iso-8859-1") as f:
        content = list(f.readlines())
        body = list(dropwhile(lambda x: x != "\n", content))
        # print "dropped %d lines" % (len(content) - len(body))
        body = " ".join(body)
        sentences = sent_tokenize(body)
        for sentence in sentences:
            words = word_tokenize(sentence)
            for word in words:
                is_punctuation = all([s in string.punctuation for s in word])
                if not is_punctuation:
                    articles_file.write(word)
                    articles_file.write(" ")
                    sentences_file.write(word)
                    sentences_file.write(" ")
            sentences_file.write("\n")
    articles_file.write("\n")

sentences_file.close()
articles_file.close()
