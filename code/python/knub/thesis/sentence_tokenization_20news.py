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
    class_mapper = {}
    class_index = 0
    for dirpath, _, filenames in os.walk(args.twentynews_folder):
        if dirpath == args.twentynews_folder:
            continue

        class_mapper[os.path.basename(dirpath)] = class_index
        for filename in filenames:
            files.append((dirpath + "/" + filename, class_index))
        class_index += 1

    # Shuffling to randomize
    shuffle(files)
    return files, class_mapper


def main():
    filenames, class_mapper = get_filenames()

    class_mapper_file = codecs.open(args.twentynews_folder + "/mapping.txt", "w", encoding="utf-8")
    for clazz, idx in class_mapper.iteritems():
        class_mapper_file.write(clazz + "\t" + str(idx) + "\n")
    class_mapper_file.close()

    sentences_file = codecs.open(args.twentynews_folder + "/sentences.txt", "w", encoding="utf-8")
    articles_file = codecs.open(args.twentynews_folder + "/articles.txt", "w", encoding="utf-8")
    articles_class_file = codecs.open(args.twentynews_folder + "/articles.class.txt", "w", encoding="utf-8")

    i = 0
    for filename, class_index in filenames:
        if i % 1000 == 0:
            print i
        i += 1
        basename = os.path.basename(filename)
        current_article = ""
        with codecs.open(filename, "r", encoding="iso-8859-1") as f:
            content = list(f.readlines())
            body = list(dropwhile(lambda x: x != "\n", content))
            # print "dropped %d lines" % (len(content) - len(body))
            body = " ".join(body)
            sentences = sent_tokenize(body)
            for sentence in sentences:
                words = word_tokenize(sentence)
                words = [word for word in words if not all([s in string.punctuation for s in word])]
                if words:
                    sentences_file.write(str(basename) + "\t" + str(class_index) + "\t")
                    words = " ".join(words)
                    current_article += words + " "
                    # articles_file.write(words)
                    sentences_file.write(words)
                    sentences_file.write("\n")
        if current_article:
            articles_file.write(filename + "\t" + current_article + "\n")
            articles_class_file.write(str(class_index) + "\n")

    sentences_file.close()
    articles_file.close()
    articles_class_file.close()

if __name__ == "__main__":
    main()
