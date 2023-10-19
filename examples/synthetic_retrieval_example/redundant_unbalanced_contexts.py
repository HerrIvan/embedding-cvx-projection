"""
En example where we produce one embedding as a linear combination of others of three different ones, A, B and C.

However, as the bases to generate the embeddings we will 100 vectors close to A, 10 close to B and 1 close to C.

We will use high dimensional vectors, just to prove that toy examples also work in high dimensions.
"""
import math
import argparse

import numpy as np

from embedding_cvx_projection import cone_projection

# np precision
np.set_printoptions(precision=3)

N_ITEMS_SOLUTION = 5  # think of this as the number of "contexts" retrieved for a RAG system.
N_PROBLEMS = 40  # number of random problems to generate
DISPERSION = 0.1  # dispersion of the context around the prototype "context" (the lower, the more compact the clusters)
MIN_CONTRIB = 0.3  # minimum contribution of each prototype to the solution (the higher, the more balanced the solution)


def construct_problem(dispersion: float, min_contrib: float):
    # Define the bases
    n_dims = 150

    # the amount of members derived from each "source" vector increments exponentially
    # with this specific structure, we can derive the class of a vector from its index
    n_bases = [1, 9, 90, 900]
    n_clusters = len(n_bases)

    # create the three basis vectors as normalized random vectors
    sources = np.random.randn(n_dims, n_clusters)

    # normalize them
    sources /= np.linalg.norm(sources, axis=0)

    # create empty bases matrix (each column is a basis vector)
    bases = None

    # fill the bases matrix with the sources
    for i in range(n_clusters):
        cluster_vecs = (np.random.randn(n_dims, n_bases[i]) * dispersion +
                        (1. - dispersion) * sources[:, i].reshape(-1, 1))

        cluster_vecs /= np.linalg.norm(cluster_vecs, axis=0)

        if bases is None:
            bases = cluster_vecs
        else:
            bases = np.hstack([bases, cluster_vecs])

    # create the embedding to be explained as a linear combination of the bases
    # create weights for the convex combination
    weights = np.random.rand(n_clusters) * (1. - min_contrib) + min_contrib
    weights /= weights.sum()

    # create the embedding as a linear combination of the sources
    embedding = (sources @ weights).reshape(-1, 1)
    embedding /= np.linalg.norm(embedding)

    return bases, embedding


def solve_by_projection(bases, embedding) -> np.array:
    _, mixture = cone_projection(bases, embedding)

    return mixture


def solve_by_similarities(bases, embedding) -> np.array:
    return (bases.T @ embedding).T.reshape(-1)


def solve_by_least_squares(bases, embedding) -> np.array:
    return np.linalg.lstsq(bases, embedding, rcond=None)[0].reshape(-1)


def evaluate_solution(coeffs: np.array, n_items: int) -> tuple[np.array, np.array]:
    # sort the mixture by value
    sorted_coeffs_idxs = np.argsort(np.abs(coeffs))[::-1]

    # lazy hack to define classes as the log, it only works if the n_bases array has that exact structure.
    cs = map(lambda x: math.ceil(np.log10(x + 1)), sorted_coeffs_idxs[:n_items])

    return len(set(cs))


def main(*, n_items_solution, n_problems, dispersion, min_contrib):

    classes_in_solution = [[], [], []]
    solvers = [solve_by_projection, solve_by_least_squares, solve_by_similarities]

    print("Solving {} problems with {} items in the solution.".format(n_problems, n_items_solution))

    for i in range(n_problems):
        bases, embedding = construct_problem(dispersion, min_contrib)

        for j, solver in enumerate(solvers):
            coeffs = solver(bases, embedding)

            classes_in_solution[j].append(evaluate_solution(coeffs, n_items_solution))

        if (i + 1) % 2 == 0:
            print(f"Problem {i + 1} solved.")

    print("Problem conf. summary:")
    print("----------------------")
    print(f"Number of problems run: {n_problems}")
    print(f"Number of items retrieved for each the solution: {n_items_solution}")
    print(f"Minimum contribution of each cluster to the solution: {min_contrib}")
    print(f"Cluster dispersion: {dispersion}")
    print("----------------------")
    # average number of classes in the solution
    classes_in_solution = np.array(classes_in_solution)
    print(
        f"Different clusters per solver (average and std):")
    for x in zip([s.__name__ for s in solvers], classes_in_solution.mean(1), classes_in_solution.std(1)):
        print(f"{x[0]}: mean {x[1]}, std: {x[2].round(2)}")
    print("----------------------")


if __name__ == "__main__":

    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_items_solution', type=int, default=N_ITEMS_SOLUTION)
    parser.add_argument('-p', '--n_problems', type=int, default=N_PROBLEMS)
    parser.add_argument('-d', '--dispersion', type=float, default=DISPERSION)
    parser.add_argument('-m', '--min_contrib', type=float, default=MIN_CONTRIB)

    args = parser.parse_args()

    main(**vars(args))
