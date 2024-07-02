"""Approximate nearest neighborhood."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import dtype, ndarray
    from torch import Tensor


class BaseAnn(ABC):
    """Interface for Approximate Nearest Neighbor (ANN) algorithms.

    This abstract base class defines the required methods for fitting data
    and querying nearest neighbors in the embedding space.
    """

    @abstractmethod
    def _fit(self: BaseAnn, embeddings: ndarray) -> None:
        """Fit the data into ANN.

        Args:
        ----
            embeddings (ndarray): The data to fit into the ANN structure.

        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self: BaseAnn,
        top_k: int,
        embedding: Tensor | ndarray[Any, dtype[Any]],
    ) -> tuple[ndarray, ndarray]:
        """Search for nearest neighbors in the embedding pool.

        Args:
        ----
            top_k (int): The number of nearest neighbors to retrieve.
            embedding (ndarray): The embedding vector to query for nearest neighbors.

        Returns:
        -------
            Tuple[int, float]: A tuple contains the index of the neighbor and the distances.

        """
        raise NotImplementedError

    def __repr__(self: BaseAnn) -> str:
        """Provide an unambiguous string representation of the BaseAnn object."""
        return "BaseAnn"
