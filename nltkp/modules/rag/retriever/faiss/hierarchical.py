"""Module for performing hierarchical similarity searches using FAISS.

This module defines the FaissHierarchicalSearch class which extends BaseFaissRetrieval
to manage and perform similarity searches on hierarchical embeddings stored in a FAISS index.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from joblib import load
from numpy import vstack
from tqdm import tqdm

from nltkp.modules.rag.retriever import SearchInput, SearchOutput
from nltkp.utils import LOGGER

from .base import BaseFaissRetrieval

if TYPE_CHECKING:
    from numpy import float32
    from numpy.typing import NDArray
    from torch import Tensor


NDIM = 3


class FaissHierarchicalSearch(BaseFaissRetrieval):
    """Class for managing hierarchical embeddings and performing similarity searches with FAISS.

    This class extends BaseFaissRetrieval to add functionality for hierarchical
    category handling. It manages embeddings that are organized in a hierarchical
    folder structure and performs similarity searches within this hierarchy.
    """

    def __init__(self: FaissHierarchicalSearch, config: SimpleNamespace) -> None:
        """Initialize the FaissHierarchicalSearch with a specific model and configuration settings.

        Args:
        ----
            config (SimpleNamespace): Configuration settings for the vector store, including:
                - embeddings: The file path where embeddings are stored.
                - model_type: The type of model to use.
                - index_type: The type of FAISS index to create.
                - pooling_modes: Optional dictionary specifying pooling modes for the model.

        """
        super().__init__(config=config)
        self.build()

    def build(self: FaissHierarchicalSearch) -> None:
        """Build the FAISS index by loading hierarchical embeddings and fitting the index.

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
            msg = "Directory not defined in the config of RetrievalConfig"
            raise ValueError(msg)

        if isinstance(embeddings_dir, str):
            LOGGER.info("Loading hierarchical embeddings from directory: %s", embeddings_dir)
            self.embeddings, self.names = self.load_embeddings_with_hierarchy()
            self._fit(embeddings=self.embeddings)
            LOGGER.info("FAISS index built successfully with embeddings from: %s", embeddings_dir)
        else:
            msg = f"Directory type is not a string, got {type(embeddings_dir)}"
            raise TypeError(msg)

    def load_embeddings_with_hierarchy(
        self: FaissHierarchicalSearch,
    ) -> tuple[NDArray[float32], list[str]]:
        """Load hierarchical embeddings and their names from joblib files.

        Args:
        ----
            directory (str): The directory containing the embeddings.

        Returns:
        -------
            Tuple[NDArray[float32], List[str]]: A tuple containing the embeddings and their corresponding names.

        """
        names_list: list[str] = []
        embeddings_list: list[NDArray[float32]] = []
        embeddings_path = Path(self.config.embeddings)

        for embedding_file in tqdm(
            iterable=embeddings_path.glob(pattern="**/*.joblib"),
            desc="Loading hierarchical embeddings",
            unit="files",
        ):
            embedding: NDArray[float32] = load(embedding_file)
            if embedding.ndim == NDIM:  # Flatten if the embedding is unexpectedly an array of arrays
                embedding = embedding[0]
            embeddings_list.append(embedding)
            # Generate hierarchical name from the file path
            hierarchical_name: str = embedding_file.parent.as_posix()
            names_list.append(hierarchical_name)
        return vstack(tup=embeddings_list), names_list

    def search(self: FaissHierarchicalSearch, input_value: SearchInput) -> SearchOutput:
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
            LOGGER.info("Starting hierarchical similarity search with top_k: %d and sentences: %s", top_k, sentences)
            results, query_embedding = self._query(sentences=sentences, top_k=top_k)
        except KeyError as ke:
            LOGGER.error("Missing key in search parameters: %s", ke)
            raise
        except Exception as e:
            LOGGER.error("Error during hierarchical similarity search: %s", e)
            raise
        else:
            LOGGER.info("Hierarchical similarity search completed. Found %d neighbors.", len(results))
        return SearchOutput(results=results, query_embedding=query_embedding)

    def __repr__(self: FaissHierarchicalSearch) -> str:
        """Provide an unambiguous string representation of the FaissHierarchicalSearch object.

        Returns
        -------
        str:
            A string representation of the FaissHierarchicalSearch instance.

        """
        return (
            f"FaissHierarchicalSearch(model_name={self.model_name}, index_type={self.index_type}, config={self.config})"
        )
