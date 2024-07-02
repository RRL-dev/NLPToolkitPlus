"""The module defines the FaissRetrieval class, an extension of FaissSimilaritySearch.

The FaissRetrieval class integrates custom model-driven encoding with FAISS-based vector storage
and retrieval, making it ideal for applications requiring rapid and scalable retrieval of
similar items based on vector similarity.

Classes:
- FaissRetrieval: Manages named embeddings and performs similarity searches using FAISS.

Typical usage example:
config = RetrievalConfig(
    directory='path/to/embeddings',
    model_type='bert',
    index_type='IP',
    pooling_modes={}
    )
faiss_retrieval = FaissRetrieval(config)
embeddings = np.array([...])
faiss_retrieval.add(embeddings)
results = faiss_retrieval.search('sample query', top_k=10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nltkp.factory import ChainedRunnable
from nltkp.modules.indices import FaissSimilaritySearch
from nltkp.utils import LOGGER

from .base import BaseRetrieval, RetrievalConfig, SearchInput, SearchOutput

if TYPE_CHECKING:
    from numpy import float32
    from numpy.typing import NDArray
    from torch import Tensor


class FaissRetrieval(FaissSimilaritySearch, BaseRetrieval, ChainedRunnable[SearchInput, SearchOutput]):
    """Class for managing named embeddings and performing similarity searches with FAISS.

    This class extends FaissSimilaritySearch and implements the BaseRetrieval interface,
    incorporating model-driven encoding and FAISS-based vector storage and retrieval.
    """

    def __init__(self: FaissRetrieval, config: RetrievalConfig) -> None:
        """Initialize the FaissRetrieval with a specific model and configuration settings.

        Args:
        ----
            model (Module): The model used for generating vector embeddings.
            config (RetrievalConfig): Configuration settings for the vector store, including:
                - directory: The file path where embeddings are stored.
                - model_type: The type of model to use.
                - index_type: The type of FAISS index to create.
                - pooling_modes: Optional dictionary specifying pooling modes for the model.

        """
        super().__init__(config=config)
        ChainedRunnable.__init__(self=self, func=self.search)
        self.config = config
        self.build()

    def build(self: FaissRetrieval) -> None:
        """Build the FAISS index by loading embeddings and fitting the index.

        This method loads the embeddings and their corresponding names from the specified directory
        and fits the FAISS index with the loaded embeddings.

        Raises
        ------
            ValueError: If the directory is not defined in the configuration.

        """
        msg: str
        directory: str | None = None

        if isinstance(self.config, RetrievalConfig):
            directory = self.config.directory

        if isinstance(self.config, dict):
            directory = self.config.get("directory", None)

        if directory is None:
            msg = "Directory not defined at config of RetrievalConfig"
            raise ValueError(msg)

        if isinstance(directory, str):
            LOGGER.info("Loading embeddings from directory: %s", directory)
            self.embeddings, self.names = self.load_embeddings_with_names(directory=directory)
            self._fit(embeddings=self.embeddings)
            LOGGER.info("FAISS index built successfully with embeddings from: %s", directory)
        else:
            msg = f"Directory type not a str, got {type(directory)}"
            raise TypeError(msg)

    def add(self: FaissRetrieval, embeddings: NDArray[float32]) -> None:
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

    def search(self: FaissRetrieval, input_value: SearchInput) -> SearchOutput:
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

    def __repr__(self: FaissRetrieval) -> str:
        """Provide an unambiguous string representation of the FaissRetrieval object.

        Returns
        -------
        str:
            A string representation of the FaissRetrieval instance, including model name, index type,
            and configuration details.

        """
        return f"FaissRetrieval(model_name={self.model_name}, index_type={self.index_type}, config={self.config})"
