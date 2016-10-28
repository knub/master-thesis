import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn import mixture
import time
import mkl

warnings.filterwarnings("ignore", module="matplotlib")


def timing(f):
    TIMING = False
    if TIMING:
        def wrap(*args):
            time1 = time.time()
            ret = f(*args)
            time2 = time.time()
            print '%s function took %0.3f ms' % (f.func_name, (time2 - time1) * 1000.0)
            return ret
        return wrap
    else:
        return f


@timing
def read_vectors(file_name):
    word_vector_pairs = []
    with open(file_name, "r") as f:
        for line in f:
            split = line.rstrip().split("\t")
            word = split[0]
            vector = [float(s) for s in split[1:]]

            word_vector_pairs.append((word, vector))

    return word_vector_pairs


@timing
def determine_best_gmm(X):

    lowest_bic = np.infty
    n_components_range = range(1, 4)
    cv_types = ['diag', 'full']
    for n_components in n_components_range:
        for cv_type in cv_types:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic_score = gmm.bic(X)
            if bic_score < lowest_bic:
                lowest_bic = bic_score
                best_model = gmm

    # noinspection PyUnboundLocalVariable
    return best_model


@timing
def create_gmm_output(gmm, X, file_name):
    with open(file_name + ".output", "w") as f:
        nr_features = X.shape[1]
        nr_components = gmm.n_components
        # noinspection PyUnboundLocalVariable
        f.write(str(nr_components) + "\n")
        f.write(gmm.covariance_type + "\n")
        matrices = []
        covs = gmm.covariances_
        if gmm.covariance_type == "diag":
            for row in covs:
                matrices.append(np.diag(row))
        elif gmm.covariance_type == "spherical":
            for variance in covs:
                matrices.append(variance * np.identity(nr_features))
        elif gmm.covariance_type == "tied":
            for i in range(nr_components):
                matrices.append(covs)
        elif gmm.covariance_type == "full":
            for cov in covs:
                matrices.append(cov)
        else:
            raise RuntimeError("Not handled type!")
        f.write("\t".join([str(v) for v in gmm.weights_]))
        f.write("\n")
        for mean in gmm.means_:
            f.write("\t".join([str(v) for v in mean]))
            f.write("\n")
        for matrix in matrices:
            for row in matrix:
                f.write("\t".join([str(v) for v in row]))
                f.write("\n")


@timing
def create_plot(gmm, word_vector_pairs, X, file_name):
    plt.figure()
    x = np.linspace(-2., 2.)
    y = np.linspace(-2., 2.)
    X_grid, Y_grid = np.meshgrid(x, y)
    XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
    Z_grid = -gmm.score_samples(XX)
    Z_grid = Z_grid.reshape(X_grid.shape)
    CS = plt.contour(X_grid, Y_grid, Z_grid, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    # CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=50, c="red", marker="o")
    plt.scatter(X[:, 0], X[:, 1], .8)
    for word, vector in word_vector_pairs:
        plt.text(vector[0], vector[1], word, size=6, horizontalalignment='center', verticalalignment='top')
    plt.title('topic=%s n_components=%d, cov_type=%s' % (os.path.basename(file_name), gmm.n_components, gmm.covariance_type))
    plt.savefig(file_name + ".png", dpi=300)
    # plt.show()

def main():
    mkl.set_num_threads(1)
    args = sys.argv[1:]

    for file_name in args:
        word_vector_pairs = read_vectors(file_name)

        vectors = [vector for _, vector in word_vector_pairs]
        X = np.array(vectors)
        gmm = determine_best_gmm(X)

        create_gmm_output(gmm, X, file_name)

        # if X.shape[1] == 2 and os.path.basename(os.path.dirname(file_name)) == "0":
        #     create_plot(gmm, word_vector_pairs, X, file_name)

if __name__ == "__main__":
    main()
