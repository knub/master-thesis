import numpy as np
import gensim, logging
import pandas as pnd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

WORD2VEC_VECTOR_FILE = "/home/knub/Repositories/master-thesis/data/GoogleNews-vectors-negative300.bin"
GLOVE_VECTOR_FILE = "/home/knub/Repositories/master-thesis/data/glove.6B.50d.txt"
# Three topics taken from original LDA paper by Blei et al.
topics = [["school", "students", "schools", "education", "teachers", "high", "public", "teacher", "elementary", "president"],
          ["million", "tax", "budget", "billion", "federal", "government", "spending", "state", "plan", "money"],
          ["film", "show", "music", "movie", "play", "musical", "actor", "opera", "theater", "actress"]]

word2vec = gensim.models.Word2Vec.load_word2vec_format(GLOVE_VECTOR_FILE, binary=False)

# print word2vec.most_similar(positive=['woman', 'king'], negative=['man'])

vectors = [np.array([word2vec[word] for word in topic]) for topic in topics]
# vectors = np.array(vectors, dtype=np.dtype([np.float32, np.float32]))
vectors = np.array(vectors)
vectors = np.reshape(vectors, (30, 50))

tsne = TSNE(n_components=2, random_state=0)
low_dim = tsne.fit_transform(vectors)
# low_dim = np.reshape(low_dim, (3, 10, 2))

df = pnd.DataFrame(low_dim, columns=["x", "y"])
df["word"] = [word for topic in topics for word in topic]
df["topic"] = ["education"] * 10 + ["taxes"] * 10 + ["arts"] * 10
print df

plt.scatter(x=df.x, y=df.y, c=df.topic)
plt.show()


