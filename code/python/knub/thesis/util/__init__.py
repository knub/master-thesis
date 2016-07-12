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