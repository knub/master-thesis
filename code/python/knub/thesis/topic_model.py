import logging, gensim, bz2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    # print("Testd")
    # documents = ["Human machine interface for lab abc computer applications",
    #              "A survey of user opinion of computer system response time",
    #              "The EPS user interface management system",
    #              "System and human system engineering testing of EPS",
    #              "Relation of user perceived response time to error measurement",
    #              "The generation of random binary unordered trees",
    #              "The intersection graph of paths in trees",
    #              "Graph minors IV Widths of trees and well quasi ordering",
    #              "Graph minors A survey"]
    # documents = remove_stop_words(documents)
    # documents = remove_rare_words(documents)
    #
    #
    # dictionary = corpora.Dictionary(documents)
    # dictionary.save('/tmp/deerwester.dict')
    # print(dictionary)

    id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File("/data/wikipedia/2016-06-21/gensim_wordids.txt.bz2"))
    mm = gensim.corpora.MmCorpus("/san2/data/wikipedia/2016-06-21/gensim_tfidf.mm")
    print mm




if __name__ == "__main__":
    main()