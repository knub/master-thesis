import logging, gensim, bz2
from knub.thesis.util.memory import limit_memory

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    logging.info("Starting Wikipedia LDA")
    # limit memory to 32 GB
    limit_memory(32000)

    id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File("/data/wikipedia/2016-06-21/gensim_wordids.txt.bz2"))
    mm = gensim.corpora.MmCorpus("/data/wikipedia/2016-06-21/gensim_tfidf.mm")
    print mm
    lda = gensim.models.ldamodel.LdaModel(corpus=mm, num_topics=100, id2word=id2word, chunksize=10000, passes=1)
    # lda = gensim.models.ldamodel.LdaModel(corpus=mm, num_topics=100, id2word=id2word, workers=3)
    lda.save("/data/wikipedia/2016-06-21/topics.model")
    lda.print_topics()
    logging.info("Finished Wikipedia LDA")

if __name__ == "__main__":
    main()