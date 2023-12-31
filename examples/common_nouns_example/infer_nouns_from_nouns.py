"""
In this example we will get the embeddings of some common Nouns and will try to play the game of
explaining one in terms of others. For instance, King = Monarch + Man.

We will use the openai API to get the embeddings of the words.
"""
from typing import List, Tuple, Union
import os

import numpy as np
from dotenv import load_dotenv
import pandas as pd
from icecream import ic

from embedding_library import EmbeddingLibrary
from embedding_cvx_projection import cone_projection

# set precision
np.set_printoptions(precision=3)

load_dotenv()  # this allows us to read the OPENAI_API_KEY from the .env file


def compute_solution_measures(
        sols: List[Tuple],
        M: np.array,
        embds: EmbeddingLibrary,
        n: int) -> pd.DataFrame:
    # store the solutions in a dataframe
    sols_df = pd.DataFrame(sols, columns=["word", "value", "mixture"])

    # sort the solutions by value and get the top n
    sols_df = sols_df.sort_values(by="value", ascending=False)

    sols_df = sols_df.head(n)

    # add the word-to-words similarity in every row
    sols_df["similarities"] = sols_df["word"].apply(lambda w: M.T @ embds(w))

    # get the relative gain in similarity
    sols_df["gain"] = (1 - ((1 - sols_df["value"]) / (1 - sols_df["similarities"].apply(np.max))))

    # round the values
    sols_df["similarities"] = sols_df["similarities"].apply(lambda x: np.round(x, 3))
    sols_df["mixture"] = sols_df["mixture"].apply(lambda x: np.round((x / x.sum()), 2))

    # sort by gain
    return sols_df.sort_values(by="gain", ascending=False)


def get_closer_composable_meaning(words: List[str], embds: EmbeddingLibrary, n: int = 8):
    """
    Returns n words present in the embedding library that are closer to any combination of the words in the list.

    :param n:
    :param words:
    :param embds:
    :return:
    """

    M = embds(words)

    sols = []
    # iterate over all the words in the embedding library
    for i, word in enumerate(embds.itow):

        if word not in words:
            emb = embds.get_embedding(i)

            # get the projection of the embedding onto the cone generated by the words in the list
            value, mixture = cone_projection(M, emb, beta=0.0, upper_bound=0.9)
            sols.append((word, value, mixture))

    sols_df = compute_solution_measures(sols, M, embds, n)

    print("-------")
    print(f"Best combination of {words}.")
    print("word, \t\tvalue, \t\tmixture, \t\tgain")
    for _, row in sols_df.iterrows():
        print(f'{row["word"]}, \t\t{row["value"]:.3f}, \t\t{row["mixture"]}, \t\t{row["gain"]:.3f}')
    print("-------")

    embds.mask_input(words)

    mean_vector = M.mean(1).reshape(-1, 1)
    mean_vector /= np.linalg.norm(mean_vector)

    sims_to_mean = embds.M.T @ mean_vector

    most_similar = np.argsort(sims_to_mean, axis=0)[::-1][:n].reshape(-1)

    df_mean_vector = pd.DataFrame(
        data=zip(
            [embds.itow[x] for x in most_similar],
            [sims_to_mean[x, 0].round(3) for x in most_similar]
        ),
        columns=["word", "similarity"]
    )

    print(df_mean_vector)

    print("-------")


def main(words: Union[List[str] | None]):
    # get the embeddings of the common nouns

    common_nouns_library_path = "embedding_library/common_nouns.pkl"

    if os.path.exists(common_nouns_library_path):
        embds = EmbeddingLibrary.load(common_nouns_library_path)
    else:
        raise FileNotFoundError(f"Not found: {common_nouns_library_path}. Run 'build_update_embedding_library.py'")

    if words is None:
        # default words
        words = ["religion", "couple",
         "kingdom", "ruler", "mountain", "lava", "explosion", "summit"]

    get_closer_composable_meaning(words, embds, n=10)


if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(
        description='Find words whose meanings can be generated combining the input words.')
    parser.add_argument(
        'words', metavar='W', type=str, nargs='*', default=None,
        help='words to combine (separated by spaces)')

    args = parser.parse_args()

    ic(args.words)

    words_ = args.words if args.words else None
    ic(words_)

    main(words_)
