import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from knub.thesis.preprocessing import remove_stop_words, remove_rare_words

from gensim import corpora, models, similarities

def main():
    print("Testd")
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
    documents = remove_stop_words(documents)
    documents = remove_rare_words(documents)


    dictionary = corpora.Dictionary(documents)
    dictionary.save('/tmp/deerwester.dict')
    print(dictionary)




if __name__ == "__main__":
    main()