import logging, gensim, bz2
import mkl
from knub.thesis.util.memory import limit_memory

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
mkl.set_num_threads(8)

def main():
    logging.info("Starting Wikipedia LDA")
    # limit memory to 32 GB
    limit_memory(32000)

    id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File("/data/wikipedia/2016-06-21/gensim_wordids.txt.bz2"))
    mm = gensim.corpora.MmCorpus("/data/wikipedia/2016-06-21/gensim_tfidf.mm")
    print mm
    # lda = gensim.models.ldamodel.LdaModel(corpus=mm, num_topics=100, id2word=id2word, update_every=0, passes=20,
    #                                       eval_every=None)
    lda = gensim.models.ldamulticore.LdaMulticore(corpus=mm, num_topics=100, id2word=id2word, batch=True, passes=20,
                                                  eval_every=None, workers=3)
    lda.save("/data/wikipedia/2016-06-21/topics.model")
    lda.print_topics(num_topics=30, num_words=10)
    logging.info("Finished Wikipedia LDA")

if __name__ == "__main__":
    main()