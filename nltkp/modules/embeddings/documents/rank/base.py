"""Module for defining the base class for ranking documents.

This module includes:
- BaseRanking: An abstract base class that ranks documents based on their embeddings and a query embedding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from numpy import float32
    from numpy.typing import NDArray

    from nltkp.modules.rag import SearchOutput


class MMRConfig(BaseModel):
    """Configuration for MMR re-ranking."""

    top_k: int = Field(default=5, description="Number of top documents to select.")
    lambda_: float = Field(default=0.5, description="Trade-off parameter between relevance and diversity.")


class RankInput(BaseModel):
    """Input parameters for MMRRanking.

    Attributes
    ----------
    query_embedding : NDArray[float32]
        The embedding of the query.
    results : list[tuple[str, float, NDArray[float32]]]
        List of tuples containing the document hash, similarity score, and document embeddings.

    """

    query_embedding: NDArray[float32]
    results: list[tuple[str, float, NDArray[float32]]]


class RankOutput(BaseModel):
    """Output of MMRRanking.

    Attributes
    ----------
    context : list[str]
        List of document hashes selected based on MMR ranking.

    """

    context: list[str]


class BaseRanking(ABC):
    """Abstract base class for a selector that ranks documents."""

    @abstractmethod
    def rank(self: BaseRanking, input_value: SearchOutput) -> RankOutput:
        """Select and rank documents based on embeddings.

        Args:
        ----
            input_value (RankInput): Input containing query_embedding and results.
                - query_embedding (NDArray[float32]): Embedding of the query.
                - results (list[tuple[str, float, NDArray[float32]]]): List of tuples containing the document hash,
                  similarity score, and document embeddings.

        Returns:
        -------
            RankOutput: Output containing ranked indices.
                - context (list[str]): List of document hashes selected based on MMR ranking.

        """
