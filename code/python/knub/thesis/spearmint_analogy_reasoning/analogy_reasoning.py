import logging
import mkl

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from knub.thesis.word2vec_eval import eval_word2vec


def analogy_reasoning(size, window, negative, sample, job_id):
    logging.info("Training word2vec")
    sentences = LineSentence("/data/wikipedia/2016-06-21/sentences.txt")
    model = Word2Vec(sentences, size=size, window=window, min_count=50, workers=7, sg=True, hs=0,
                     negative=negative, sample=sample)
    model.save_word2vec_format("/data/wikipedia/2016-06-21/embedding-models/%s" % job_id, binary=True)
    logging.info("Finished training word2vec")

    logging.info(model.most_similar(positive=['woman', 'king'], negative=['man']))
    logging.info(model.doesnt_match("breakfast cereal dinner lunch".split()))
    logging.info(model.similarity('woman', 'man'))
    return eval_word2vec(model)


def main(job_id, params):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print "params: ", params, " job_id: ", job_id
    mkl.set_num_threads(3)
    return -analogy_reasoning(params["size"][0],
                             params["window"][0],
                             params["negative"][0],
                             params["sample"][0],
                             job_id)

