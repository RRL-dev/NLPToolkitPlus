"""Module for defining the MMR-based document ranking.

This module includes:
- MMRConfig: Configuration class for MMR re-ranking.
- MMRRanking: Class that implements MMR to rank documents based on relevance and diversity.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy import float32
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor

from nltkp.factory import ChainedRunnable
from nltkp.modules.rag.retriever import SearchOutput
from nltkp.utils import LOGGER

from .base import BaseRanking, MMRConfig, RankOutput


class MMRRanking(BaseRanking, ChainedRunnable[SearchOutput, RankOutput]):
    """Class to apply Maximal Marginal Relevance (MMR) to re-rank documents based on relevance and diversity."""

    def __init__(
        self: MMRRanking,
        config: MMRConfig,
    ) -> None:
        """Initialize the MMRRanking with the given configuration, document embeddings, and query embedding.

        Args:
        ----
            config (MMRConfig): Configuration for MMR re-ranking.

        """
        super().__init__(func=self.rank)
        self.config: MMRConfig = config
        LOGGER.info("MMRRanking initialized with config: %s", config)

    def rank(self: MMRRanking, input_value: SearchOutput) -> RankOutput:
        """Apply Maximal Marginal Relevance (MMR) to re-rank documents based on relevance and diversity.

        Args:
        ----
            input_value (RankInput): Dictionary containing query_embedding and results.
                - "query_embedding" (NDArray[float32]): Embedding of the query.
                - "results" (list[tuple[str, float, NDArray[float32]]]): List of tuples containing the document hash,
                  similarity score, and document embeddings.

        Returns:
        -------
            RankOutput: Dictionary containing ranked document hashes.
                - "chunk_documents" (list[str]): List of document hashes selected based on MMR ranking.

        """
        try:
            results: list[tuple[str, float, NDArray[float32]]] = input_value.results
            query_embedding: NDArray[float32] | Tensor = input_value.query_embedding
            if isinstance(query_embedding, Tensor):
                query_embedding = cast(NDArray[float32], query_embedding.cpu().numpy())

        except KeyError as e:
            LOGGER.error("Missing key in inputs: %s", e)
            raise

        context_embedding: NDArray[float32] = np.vstack(tup=[embedding for _, _, embedding in results])
        top_k: int = self.config.top_k
        lambda_: float = self.config.lambda_

        LOGGER.info("Starting MMR ranking with top_k: %d and lambda: %.2f", top_k, lambda_)

        try:
            # Compute similarity between query and documents
            query_doc_similarity: NDArray[float32] = cosine_similarity(
                X=query_embedding.reshape(1, -1),
                Y=context_embedding,
            ).flatten()
            LOGGER.info(msg="Computed query-document similarity")

            # Compute pairwise similarity between documents
            pairwise_doc_similarity: NDArray[float32] = cosine_similarity(X=context_embedding)
            LOGGER.info(msg="Computed pairwise document similarity")

            selected: list[int] = []
            candidate_indices = list(range(len(context_embedding)))

            for _ in range(top_k):
                if not candidate_indices:
                    break

                if not selected:
                    # Select the most relevant document
                    next_selected: int = candidate_indices[np.argmax(a=query_doc_similarity[candidate_indices])]
                else:
                    # Compute MMR score for each candidate document
                    mmr_scores: list[float] = [
                        lambda_ * query_doc_similarity[idx]
                        - (1 - lambda_) * max(pairwise_doc_similarity[idx][selected])
                        for idx in candidate_indices
                    ]
                    next_selected = candidate_indices[np.argmax(a=mmr_scores)]

                selected.append(next_selected)
                candidate_indices.remove(next_selected)
                LOGGER.info("Selected document index: %d", next_selected)

            selected_files: list[str] = [results[idx][0] for idx in selected]
            LOGGER.info("MMR ranking completed. Selected document files: %s", selected_files)
            return RankOutput(context=selected_files)

        except Exception as e:
            LOGGER.error("Error during MMR ranking: %s", e)
            raise

    def __repr__(self: MMRRanking) -> str:
        """Provide an unambiguous string representation of the MMRRanking object."""
        return f"MMRRanking(config={self.config})"
