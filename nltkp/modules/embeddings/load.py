"""Module provides functionality for loading embeddings from pickle files.

Classes:
    EmbeddingsLoaderConfig: Configuration class for specifying the path and dimensions of embeddings.

Functions:
    load_embeddings_for_hashes: Load embeddings for the given list of hashes and associate them with similarity scores.
"""

from pathlib import Path

from numpy import float32, load
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from nltkp.utils import LOGGER


class EmbeddingsLoaderConfig(BaseModel):
    """Configuration class for specifying the path and dimensions of embeddings.

    Attributes
    ----------
    num_dim : int
        Number of dimensions for the embeddings.
    path : str
        Path to the directory containing the embedding files.

    """

    path: str = Field(default="", description="Path to the directory containing the embedding files.")
    num_dim: int = Field(default=3, description="Number of dimensions for the embeddings.")


def load_embeddings_for_hashes(
    config: EmbeddingsLoaderConfig,
    path_with_scores: list[tuple[str, float]],
) -> list[tuple[NDArray[float32], float]]:
    """Load embeddings for the given list of hashes and associate them with their similarity scores.

    Args:
    ----
        config (EmbeddingsConfig): Configuration for embeddings.
        path_with_scores (list[tuple[str, float]]): List of tuples containing hash and similarity score.

    Returns:
    -------
        list[tuple[NDArray[float32], float]]: List of tuples containing embeddings and their similarity scores.

    """
    embeddings_list: list[tuple[NDArray[float32], float]] = []
    embeddings_path = Path(config.path)

    for hash_value, score in path_with_scores:
        embedding_file: Path = embeddings_path / f"{hash_value}.pkl"
        if embedding_file.exists():
            with embedding_file.open(mode="rb") as file:
                embedding: NDArray[float32] = load(file=file, allow_pickle=True)
                if embedding.ndim == config.num_dim:
                    embedding = embedding[0]
                embeddings_list.append((embedding, score))
        else:
            LOGGER.info(msg=f"Embedding file for hash {hash_value} not found.")
    return embeddings_list
