"""Module for performing similarity searches using FAISS.

This module defines the FaissSimilaritySearch class which extends the BaseFaissAnn class
to manage and perform similarity searches on named embeddings stored in a FAISS index.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy import dtype, float32, int32, load, ndarray, vstack
from torch._tensor import Tensor
from tqdm import tqdm

from nltkp.models import BaseSentenceModel
from nltkp.utils import LOGGER

from .base import BaseFaissAnn

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from nltkp.modules.rag.retriever import RetrievalConfig


class FaissSimilaritySearch(BaseSentenceModel, BaseFaissAnn):
    """Class for managing named embeddings and performing similarity searches with FAISS.

    This class extends the functionality of BaseFaissAnn to include named embeddings,
    enabling similarity searches that return both distances and names of nearest neighbors.
    """

    def __init__(
        self: FaissSimilaritySearch,
        config: RetrievalConfig,
    ) -> None:
        """Initialize the FaissSimilaritySearch for embedding searching.

        Args:
        ----
            config (dict[str, Any]): Configuration dictionary containing settings for the model.
                - model_name (str): The name of the model to be used for generating embeddings.
                                    This name should correspond to a pre-trained model from Hugging Face's Transformers.
                - index_type (str): Specifies the type of FAISS index to create.
                                    'L2' for L2 distance (Euclidean), and 'IP' for inner product (cosine similarity).
                - pooling_modes (dict, optional): A dictionary specifying the pooling modes with sentence model.

        """
        BaseFaissAnn.__init__(self=self, index_type=config.index_type)
        BaseSentenceModel.__init__(self=self, model_name=config.model_type, pooling_modes=config.pooling_modes)
        self.config: RetrievalConfig = config
        self.names: list[str] = []

    def load_embeddings_with_names(
        self: FaissSimilaritySearch,
        directory: str,
    ) -> tuple[NDArray[float32], list[str]]:
        """Load embeddings and their names from pickle files."""
        LOGGER.info(
            "Start loading embeddings from: %s\n Using FaissSimilaritySearch",
            self.config.directory,
        )

        names_list: list[str] = []
        embeddings_list: list[NDArray[float32]] = []
        embeddings_path = Path(directory)
        # Adding a tqdm progress bar to the file processing loop
        for embedding_file in tqdm(
            iterable=embeddings_path.glob(pattern="*.pkl"),
            desc="Loading embeddings",
            unit="files",
        ):
            with embedding_file.open(mode="rb") as file:
                embedding: NDArray[float32] = load(file=file, allow_pickle=True)
                if (
                    embedding.ndim == 3  # noqa: PLR2004
                ):  # Flatten if the embedding is unexpectedly an array of arrays
                    embedding = embedding[0]
                embeddings_list.append(embedding)
                names_list.append(embedding_file.stem)
        return vstack(tup=embeddings_list), names_list

    def _encode_sentences(
        self: FaissSimilaritySearch,
        sentences: str | list[str],
    ) -> NDArray[float32] | Tensor:
        """Encode sentences to obtain their embeddings.

        Args:
        ----
            sentences (str | List[str]): The sentence or list of sentences to be encoded.

        Returns:
        -------
            NDArray[float32] | Tensor: The embeddings of the query sentence(s).

        """
        if isinstance(sentences, str):
            sentences = [sentences]
        embedding: Tensor | ndarray[Any, dtype[Any]] = self.encode(sentences=sentences, convert_to_numpy=True).squeeze()
        LOGGER.info("Encoded sentence to embedding of shape: %s", embedding.shape)
        return embedding

    def _search_index(
        self: FaissSimilaritySearch,
        top_k: int,
        embedding: NDArray[float32] | Tensor,
    ) -> tuple[NDArray[float32], NDArray[int32]]:
        """Search the FAISS index to find the nearest neighbors for the given embedding.

        Args:
        ----
            top_k (int): The number of nearest neighbors to retrieve.
            embedding (NDArray[float32] | Tensor): The embedding of the query sentence(s).

        Returns:
        -------
            tuple[NDArray[float32], NDArray[int32]]: Distances and indices of the nearest neighbors.

        """
        indices: NDArray[int32]
        distances: NDArray[float32]
        distances, indices = self.query(embedding=embedding, top_k=top_k)
        LOGGER.info("Found nearest neighbors with distances: %s\n Indices: %s", distances, indices)
        return distances, indices

    def _get_results_with_reconstructions(
        self: FaissSimilaritySearch,
        indices: NDArray[int32],
        distances: NDArray[float32],
    ) -> list[tuple[str, float, NDArray[float32]]]:
        """Get the results with names, distances, and reconstructed embeddings.

        Args:
        ----
            indices (NDArray[int32]): Indices of the nearest neighbors.
            distances (NDArray[float32]): Distances to the nearest neighbors.

        Returns:
        -------
            list[tuple[str, float, NDArray[float64]]] List of tuples containing the name, distance,
            and reconstructed embedding of each nearest neighbor.

        """
        results: list[tuple[str, float, NDArray[float32]]] = [
            (self.names[idx], float(dist), self.index.reconstruct(key=int(idx)))  # type: ignore  # noqa: PGH003
            for idx, dist in zip(indices[0], distances[0], strict=False)
        ]
        LOGGER.info(msg="Completed reconstruction of embeddings for the nearest neighbors.")
        return results

    def _query(
        self: FaissSimilaritySearch,
        top_k: int,
        sentences: str | list[str],
    ) -> tuple[list[tuple[str, float, NDArray[float32]]], NDArray[float32] | Tensor]:
        """Perform a similarity search and return names, distances, and embeddings of the nearest neighbors.

        Args:
        ----
            top_k (int): The number of nearest neighbors to retrieve.
            sentences (str | List[str]): The sentence or list of sentences to be encoded and searched.

        Returns:
        -------
            tuple[list[tuple[str, float, NDArray[float64]]], Union[NDArray[float32], Tensor]]:
                - A list of tuples containing the name, distance, and embedding of each nearest neighbor.
                - The embedding of the query sentence(s).

        """
        indices: NDArray[int32]
        distances: NDArray[float32]

        LOGGER.info("Performing query with top_k: %d\n Sentences: %s", top_k, sentences)
        embedding: NDArray[float32] | Tensor = self._encode_sentences(sentences=sentences)
        distances, indices = self._search_index(top_k=top_k, embedding=embedding)
        results: list[tuple[str, float, NDArray[float32]]] = self._get_results_with_reconstructions(
            indices=indices,
            distances=distances,
        )
        return results, embedding

    def __repr__(self: FaissSimilaritySearch) -> str:
        """Provide an unambiguous string representation of the FaissSimilaritySearch object.

        Returns
        -------
        str:
            A string representation of the FaissSimilaritySearch instance.

        """
        return (
            f"FaissSimilaritySearch(model_name={self.model_name}, index_type={self.index_type}, config={self.config})"
        )
