"""The module defines the FaissRetrieval class for managing named embeddings and performing similarity searches.

Classes:
    FaissRetrieval: Extends FaissSimilaritySearch and implements the BaseRetrieval interface, incorporating
                    model-driven encoding and FAISS-based vector storage and retrieval.

"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from nltkp.factory import ChainedRunnable
from nltkp.modules.indices import FaissSimilaritySearch
from nltkp.modules.rag.retriever import BaseRetrieval, SearchInput, SearchOutput
from nltkp.utils import LOGGER

if TYPE_CHECKING:
    from numpy import float32
    from numpy.typing import NDArray
    from torch import Tensor


class BaseFaissRetrieval(FaissSimilaritySearch, BaseRetrieval, ChainedRunnable[SearchInput, SearchOutput]):
    """Class for managing named embeddings and performing similarity searches with FAISS.

    This class extends FaissSimilaritySearch and implements the BaseRetrieval interface,
    incorporating model-driven encoding and FAISS-based vector storage and retrieval.
    """

    def __init__(self: BaseFaissRetrieval, config: SimpleNamespace) -> None:
        """Initialize the FaissRetrieval with a specific model and configuration settings.

        Args:
        ----
            model (Module): The model used for generating vector embeddings.
            config (SimpleNamespace): Configuration settings for the vector store, including:
                - embeddings: The file path where embeddings are stored.
                - model_type: The type of model to use.
                - index_type: The type of FAISS index to create.
                - pooling_modes: Optional dictionary specifying pooling modes for the model.

        """
        super().__init__(config=config)
        ChainedRunnable.__init__(self=self, func=self.search)
        self.config = config
        self.build()

    def build(self: BaseFaissRetrieval) -> None:
        """Build the FAISS index by loading embeddings and fitting the index.

        This method loads the embeddings and their corresponding names from the specified directory
        and fits the FAISS index with the loaded embeddings.

        Raises
        ------
            ValueError: If the directory is not defined in the configuration.
            TypeError: If the directory is not a string.

        """
        msg: str
        embeddings_dir: str | None = None

        if isinstance(self.config, SimpleNamespace):
            embeddings_dir = self.config.embeddings

        if isinstance(self.config, dict):
            embeddings_dir = self.config.get("directory", None)

        if embeddings_dir is None:
            msg = "Directory not defined at config of RetrievalConfig"
            raise ValueError(msg)

        if isinstance(embeddings_dir, str):
            LOGGER.info("Loading embeddings from directory: %s", embeddings_dir)
            self.embeddings, self.names = self.load_embeddings_with_names()
            self._fit(embeddings=self.embeddings)
            LOGGER.info("FAISS index built successfully with embeddings from: %s", embeddings_dir)
        else:
            msg = f"Directory type not a str, got {type(embeddings_dir)}"
            raise TypeError(msg)

    def add(self: BaseFaissRetrieval, embeddings: NDArray[float32]) -> None:
        """Add vector embeddings to the FAISS index.

        Args:
        ----
            embeddings (NDArray[float32]): The array of embeddings to be added to the index.

        """
        try:
            LOGGER.info("Adding embeddings to the FAISS index.")
            self.index.add(embeddings)  # type: ignore # noqa: PGH003
            LOGGER.info("Successfully added embeddings to the FAISS index.")

        except Exception as e:
            LOGGER.error("Error while adding embeddings: %s", e)
            raise

    def search(self: BaseFaissRetrieval, input_value: SearchInput) -> SearchOutput:
        """Perform a similarity search and return names, distances, and embeddings of the nearest neighbors.

        Args:
        ----
            input_value (SearchInput): The input parameters for the search, including:
                - top_k (int): The number of nearest neighbors to retrieve.
                - sentences (str | List[str]): The sentence or list of sentences to be encoded and searched.

        Returns:
        -------
            SearchOutput: An object containing the results and the query embedding.
                - results: A list of tuples containing the name, distance, and embedding of each nearest neighbor.
                - query_embedding: The embedding of the query sentence(s).

        """
        results: list[tuple[str, float, NDArray[float32]]]
        query_embedding: NDArray[float32] | Tensor
        try:
            top_k: int = input_value.top_k
            sentences: str | list[str] = input_value.sentences
            LOGGER.info("Starting similarity search with top_k: %d and sentences: %s", top_k, sentences)
            results, query_embedding = self._query(sentences=sentences, top_k=top_k)
        except KeyError as ke:
            LOGGER.error("Missing key in search parameters: %s", ke)
            raise
        except Exception as e:
            LOGGER.error("Error during similarity search: %s", e)
            raise
        else:
            LOGGER.info("Similarity search completed. Found %d neighbors.", len(results))
        return SearchOutput(results=results, query_embedding=query_embedding)

    def __repr__(self: BaseFaissRetrieval) -> str:
        """Provide an unambiguous string representation of the FaissRetrieval object.

        Returns
        -------
        str:
            A string representation of the FaissRetrieval instance, including model name, index type,
            and configuration details.

        """
        return f"FaissRetrieval(model_name={self.model_name}, index_type={self.index_type}, config={self.config})"
