import logging
from abc import ABC, abstractmethod
from typing import Union, List
import os

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder(ABC):
    def __call__(self, input_: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Convenience method to get the embedding of the input.

        :param input_:
        :return:
        """
        return self._embed(input_)

    @abstractmethod
    def _embed(self, input_: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        pass

    @abstractmethod
    def __str__(self):
        pass


class OpenAIEmbedder(Embedder):
    model_name: str

    @property
    def openai_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        return api_key

    def __init__(self, model_name):
        self.model_name = model_name

    def _embed(self, input_: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get the embeddings of the input.
        """
        import openai  # lazy import

        logger.info(f"Embedding input: {input_}")

        embeddings_response = openai.embeddings.create(
            model=self.model_name,
            input=input_
        )

        logger.info(f"Number of embeddings received: {len(embeddings_response.data)}")

        return [embedding.embedding for embedding in embeddings_response.data]

    def __str__(self):
        return f"OpenAIEmbedder using model: {self.model_name}"
