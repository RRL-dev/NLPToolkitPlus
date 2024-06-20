"""Module for performing similarity searches using FAISS.

This module defines the FaissSimilaritySearch class which extends the BaseFaissAnn class
to manage and perform similarity searches on named embeddings stored in a FAISS index.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy import dtype, float32, int32, load, ndarray, vstack
from tqdm import tqdm  # Import tqdm for the progress bar

from nltkp.models import BaseSentenceModel
from nltkp.utils import LOGGER

from .base import BaseFaissAnn

if TYPE_CHECKING:
    from numpy import ndarray
    from numpy.typing import NDArray
    from torch import Tensor


class FaissSimilaritySearch(BaseSentenceModel, BaseFaissAnn):
    """Class for managing named embeddings and performing similarity searches with FAISS.

    This class extends the functionality of BaseFaissAnn to include named embeddings,
    enabling similarity searches that return both distances and names of nearest neighbors.
    """

    def __init__(
        self: FaissSimilaritySearch,
        directory: str,
        model_name: str,
        index_type: str = "IP",
        pooling_modes: dict | None = None,
    ) -> None:
        """Initialize the FaissSimilaritySearch for embedding searching.

        Args:
        ----
        directory: str
            The directory path where embeddings are stored.
            This directory should contain binary files of embeddings,
            which will be loaded and indexed for similarity searching.

        model_name: str
            The name of the model to be used for generating embeddings.
            This name should correspond to a pre-trained model in Hugging Face's Transformers.

        index_type: str, default='L2'
            Specifies the type of FAISS index to create.
            'L2' for L2 distance (Euclidean), and 'IP' for inner product (cosine similarity).

        pooling_modes: dict, default={}
            A dictionary specifying the pooling modes to be used with the sentence model.

        """
        if pooling_modes is None:
            pooling_modes = {}  # Ensure default is not mutable

        BaseFaissAnn.__init__(self=self, index_type=index_type)
        BaseSentenceModel.__init__(self=self, model_name=model_name, pooling_modes=pooling_modes)
        self.directory: str = directory
        self.names: list[str] = []
        self.embeddings, self.names = self.load_embeddings_with_names(directory=directory)
        self._fit(embeddings=self.embeddings)

    def load_embeddings_with_names(
        self: FaissSimilaritySearch,
        directory: str,
    ) -> tuple[NDArray[float32], list[str]]:
        """Load embeddings and their names from pickle files."""
        LOGGER.info(
            "Start loading embeddings from: %s\n Using FaissSimilaritySearch",
            self.directory,
        )

        names_list: list[str] = []
        embeddings_list: list[ndarray] = []
        embeddings_path = Path(directory)
        # Adding a tqdm progress bar to the file processing loop
        for embedding_file in tqdm(
            iterable=embeddings_path.glob(pattern="*.pkl"),
            desc="Loading embeddings",
            unit="files",
        ):
            with embedding_file.open(mode="rb") as file:
                embedding: ndarray = load(file=file, allow_pickle=True)
                if (
                    embedding.ndim == 3  # noqa: PLR2004
                ):  # Flatten if the embedding is unexpectedly an array of arrays
                    embedding = embedding[0]
                embeddings_list.append(embedding)
                names_list.append(embedding_file.stem)
        return vstack(tup=embeddings_list), names_list

    def query_with_names(
        self: FaissSimilaritySearch,
        sentences: str | list[str],
        n_neighbors: int,
    ) -> list[tuple[str, float]]:
        """Perform a similarity search and return names and distances of the nearest neighbors."""
        LOGGER.info("Encode sentence: %s\n Using model: %s", sentences, self.model_name)

        if isinstance(sentences, str):
            sentences = [sentences]

        embedding: Tensor | ndarray[Any, dtype[Any]] = self.encode(
            sentences=sentences,
            convert_to_numpy=True,
        ).squeeze()
        LOGGER.info("Query embedding sentence of shape dim: %s", embedding.shape)

        indices: NDArray[int32]
        distances: NDArray[float32]

        distances, indices = self.query(embedding=embedding, n_neighbors=n_neighbors)
        # Skip the first result if it includes the query itself
        return [
            (self.names[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0], strict=False)
        ]
