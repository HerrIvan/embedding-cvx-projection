"""
This script is a stub of class used to store and manipulate embeddings.

For any complex application, it is recommended to use a proper vector DB.
"""

import pickle
from typing import Union, List, Tuple

import numpy as np

from .embedders import Embedder

# whether to get query embeddings on-the-fly, were they not in the library
EMBED_OTF = True


class EmbeddingLibrary:
    """
    A class to store and manipulate embeddings.

    Essentially, a numpy 2D matrix with a dictionary to map words to column indexes with the added functionality
    to mask columns, retrieve embeddings from an endpoint, compute similarities, etc.

    The embeddings are stored in a matrix, where each column is an embedding vector.
    The columns are indexed by the words they represent.
    """

    embedder: Embedder

    _mask: Union[np.ndarray, None]
    _embedding_matrix: np.ndarray
    _itow: list[str] = []  # maps column indexes to words

    wtoi: dict[str, int] = {}  # maps words to column indexes

    _cache_coeffs: np.ndarray = None

    embed_otf: bool = EMBED_OTF

    @property
    def M(self) -> np.ndarray:
        if self._mask is not None:
            return self._embedding_matrix[:, self._mask]
        return self._embedding_matrix

    @property
    def itow(self) -> List[str]:
        if self._mask is not None:
            return list(np.array(self._itow)[self._mask])
        return self._itow

    def __init__(self,
                 embedder: Embedder,
                 initial_inputs: Union[List[str], str, None] = None):
        # Initialize attributes
        self.embedder = embedder
        self._mask = None
        self.wtoi = {}
        self._itow = []
        self._embedding_matrix = None

        # Fetch and add embeddings
        if initial_inputs:
            self.add_embeddings(initial_inputs)

    # Add embeddings
    def add_embeddings(self, inputs: Union[List[str], str], update_existing: bool = False):

        if isinstance(inputs, str):
            inputs = [inputs]

        if not update_existing:
            # remove inputs that are already in the embedding library
            inputs = [input_ for input_ in inputs if input_ not in self.wtoi]

        # check if there are any inputs left to add
        if len(inputs) == 0:
            return

        print(inputs)

        # get the embeddings of the inputs
        # the "embedder" should control the rate of requests to the API.
        embeddings = self.embedder(inputs)

        # loop over the input words and their corresponding embeddings
        for input_, embedding_vector in zip(inputs, embeddings):
            self.add_embedding(input_, embedding_vector=embedding_vector)

    def add_embedding(self,
                      input_: str,
                      embedding_vector: Union[List[float], None] = None,
                      update_existing: bool = False):
        """
        Add an embedding to the embedding library.
        :param input_:
        :param embedding_vector:
        :param update_existing:
        :return:
        """

        # if no embedding vector is provided, get it from the embedder.
        if embedding_vector is None:
            embedding_vector = self.embedder(input_)[0]

        # make it a column vector
        embedding_np = np.array(embedding_vector).reshape(-1, 1)

        # if the embedding matrix is empty, initialize it
        if self._embedding_matrix is None:
            print("initializing embedding matrix")
            self._embedding_matrix = embedding_np
            self.wtoi[input_] = 0
            self._itow.append(input_)
            return

        if input_ in self.wtoi:
            if update_existing:
                col = self.wtoi[input_]
                self._embedding_matrix[:, col] = embedding_np
        else:
            self._embedding_matrix = np.hstack([self._embedding_matrix, embedding_np])
            self.wtoi[input_] = self._embedding_matrix.shape[1] - 1
            self._itow.append(input_)

    # Remove embeddings
    def remove_embedding(self, input_: str):
        if input_ not in self.wtoi:
            raise ValueError(f"input {input_} not in embedding library")

        col = self.wtoi[input_]

        self._embedding_matrix = np.delete(self._embedding_matrix, col, axis=1)
        del self.wtoi[input_]

        # update the indexes of the words that come after the removed word
        for word, col in self.wtoi.items():
            if col > col:
                self.wtoi[word] -= 1

        del self._itow[col]

    def remove_embeddings(self, inputs: Union[List[str], str]):
        if isinstance(inputs, str):
            inputs = [inputs]

        for input_ in inputs:
            self.remove_embedding(input_)

    # Getters
    def get_embedding(self, input_: Union[List[str], str, int]) -> np.ndarray:

        if isinstance(input_, list):
            return np.array([self.get_embedding(word) for word in input_]).T

        if isinstance(input_, int):
            col = input_

        else:
            if input_ not in self.wtoi:

                if self.embed_otf:
                    # get the embedding on the fly, but don't add it to the library
                    print("input not in library, getting embedding on the fly", input_)
                    return np.array(self.embedder(input_)[0])
                else:
                    raise ValueError(f"input {input_} not in embedding library")

            col = self.wtoi[input_]

        return self._embedding_matrix[:, col]

    def __call__(self, input_: Union[List[str],str]) -> np.ndarray:
        """
        Convenience method to get the embedding(s) of the input(s).

        :param input_:
        :return:
        """
        return self.get_embedding(input_)

    # Masking
    def mask_input(self, input_: Union[List[str], List[int], str, int], reset_mask: bool = True):

        # initialize mask if it is not already, every column is unmasked
        if reset_mask or self._mask is None:
            self._mask = np.ones(self._embedding_matrix.shape[1], dtype=bool)

        if isinstance(input_, list):
            for it in input_:
                self.mask_input(it, reset_mask=False)
            return

        if isinstance(input_, str):
            if input_ not in self.wtoi:
                print(f"Can't be masked. Input '{input_}' not in embedding library")
                return

            col = self.wtoi[input_]
        elif isinstance(input_, int):
            col = input_
        else:
            raise TypeError(f"input_ '{input_}' must be either a string or an int")

        self._mask[col] = False

    def remove_mask(self):
        self._mask = None

    # Similarity
    def get_most_similar_ids(self, input_: Union[str, np.ndarray], n: int = 10) -> np.array:
        """
        Get the n most similar words to the input, which can be either a word or an embedding vector.
        :param input_:
        :param n:
        :return:
        """

        if isinstance(input_, str):
            input_ = self.get_embedding(input_)

        # cache the coefficients for later use ...
        self._cache_coeffs = input_.T @ self.M

        # get the indexes of the mixture sorted by value (descending)
        return np.argsort(self._cache_coeffs)[-n:][::-1]

    def get_most_similar(self, input_: Union[str, np.ndarray], n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the n most similar words to the input, which can be either a word or an embedding vector.
        Ftb this is only used for output in the console.

        :param input_:
        :param n:
        :return: list of tuples (word, similarity coefficient)
        """

        most_similar_ids = self.get_most_similar_ids(input_, n)

        return list(
            zip(
                [self._itow[i] for i in most_similar_ids],
                list(self._cache_coeffs[most_similar_ids])
            )
        )

    def unmask_most_similar(self, input_: Union[str, np.ndarray], n: int = 10, reset_mask: bool = True):
        """
        Unmasks the n most similar words to the input, which can be either a word or an embedding vector.
        In the usual use case (reset_mask=True), all other columns will be masked.

        :param input_:
        :param n:
        :param reset_mask:
        :return:
        """
        most_similar_ids = self.get_most_similar_ids(input_, n)

        if reset_mask:
            self._mask = np.zeros(self._embedding_matrix.shape[1], dtype=bool)

        self._mask[most_similar_ids] = True

    # Persistence
    def save(self, filename):
        with open(filename, 'wb') as f:
            d = self.__dict__.copy()
            # remove the cache coefficients and the mask before saving
            if '_cache_coeffs' in d:
                del d['_cache_coeffs']
            if '_mask' in d:
                del d['_mask']
            pickle.dump(d, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            attributes = pickle.load(f)
        obj = cls(attributes['embedder'])
        obj.__dict__.update(attributes)
        return obj
