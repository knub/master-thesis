import gensim
import pandas as pnd
from sklearn.decomposition import RandomizedPCA
from sklearn.manifold import TSNE

def pca(embeddings, n=2):
    pca = RandomizedPCA(n_components=n)
    return pca.fit_transform(embeddings)

def tsne(embeddings, n=2):
    tsne = TSNE(n_components=n)
    return tsne.fit_transform(embeddings)

def tsne_with_init_pca(embeddings, n=2):
    tsne = TSNE(n_components=n, init="pca")
    return tsne.fit_transform(embeddings)


SKIP_GRAM_VECTOR_FILE = "/home/knub/Repositories/master-thesis/models/word-embeddings/embedding.model.skip-gram"
WORD2VEC_VECTOR_FILE = "/home/knub/Repositories/master-thesis/models/word-embeddings/GoogleNews-vectors-negative300.bin"

def load_embedding_model(model):
    return gensim.models.Word2Vec.load_word2vec_format(model, binary=True)

def load_skip_gram():
    return gensim.models.Word2Vec.load_word2vec_format(SKIP_GRAM_VECTOR_FILE, binary=True)

def load_word2vec():
    return gensim.models.Word2Vec.load_word2vec_format(WORD2VEC_VECTOR_FILE, binary=True)

class TopicModelLoader:

    def __init__(self, model, vectors):
        self.model = model
        self.vectors = vectors
        self.prob_columns = None
        self.topic_words = None
        self.sim_functions = ["max", "sum", "bhattacharyya", "hellinger", "jensen-shannon"]

    def load_topic_probs(self):
        df_probs = pnd.read_csv(self.model + ".topic-probs-normalized")
        self.prob_columns = map(str, list(range(len(df_probs.columns) - 2))) # - "word" and "word-prob"
        return df_probs

    def load_topics(self):
        df_topics = pnd.read_csv(self.model + ".ssv", sep=" ", encoding="utf-8")
        self.topic_words = set(df_topics.ix[:,-10:].values.flatten())
        return df_topics

    def load_all_topic_similars(self):
        df = pnd.DataFrame()
        for sim_function in self.sim_functions:
            df_sim = self.load_topic_similars(sim_function)
            if 'word' not in df.columns:
                df["word"] = df_sim["word"]
                df["similar_word"] = df_sim["similar_word"]
            sim_column = "tm_sim_%s" % sim_function
            df[sim_column] = df_sim[sim_column]

        df["we_sim"] = df[["word", "similar_word"]].apply(
            lambda x: self.get_similarity(x["word"], x["similar_word"], self.vectors), axis=1)
        df = df[df["we_sim"] != -1]

        return df

    def load_topic_similars(self, type):
        df_similars = pnd.read_csv(self.model + ".similars-%s" % type, sep="\t", header=None, encoding="utf-8")
        df_similars["tm_sim"] = df_similars[0]
        del df_similars[0] # delete original similarity column
        del df_similars[1] # delete "SIM" column
        df_similars.columns = ["word", "similar_word", "tm_sim_%s" % type]
        return df_similars

    def get_similarity(self, word1, word2, v):
        # ugly but works for now
        if word1 not in v:
            if word1.upper() in v:
                word1 = word1.upper()
            if word1.title() in v:
                word1 = word1.title()
        if word2 not in v:
            if word2.upper() in v:
                word2 = word2.upper()
            if word2.title() in v:
                word2 = word2.title()
        try:
            return v.similarity(word1, word2)
        except KeyError:
            return -1.0
