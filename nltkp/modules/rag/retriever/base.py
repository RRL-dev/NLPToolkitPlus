"""The module defines the components necessary for creating and managing vector storage, retrieval.

Classes:
    BaseRetrieval (ABC): Abstract base class for retrieval systems that utilize vector embeddings.
    RetrievalConfig (BaseModel): Configuration model for vector storage systems.

The IndexInterface class requires subclasses to implement methods for adding vectors,
searching vectors, and optionally building the index.
The BaseRetrieval class serves as a base for implementing specific retrieval systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy import float32  # noqa: TCH002
from numpy.typing import NDArray  # noqa: TCH002
from pydantic import BaseModel
from torch import Tensor  # noqa: TCH002

if TYPE_CHECKING:
    from types import SimpleNamespace

    from faiss.swigfaiss import IndexFlatIP, IndexFlatL2


class SearchInput(BaseModel):
    """A class representing the input for a search query.

    Attributes
    ----------
        top_k (int): The number of top results to retrieve from the search.
        sentences (str | list[str]): The search query or list of queries.

    """

    top_k: int
    sentences: str | list[str]


class SearchOutput(BaseModel):
    """A class representing the output of a search operation.

    Attributes
    ----------
    results : list[tuple[str, float, NDArray[float32]]]
        A list of tuples where each tuple represents a search result,
        typically containing information like document ID, relevance score, etc.
    query_embedding : NDArray[float32] | Tensor
        The numerical embedding of the query used in the search operation.

    """

    results: list[tuple[str, float, NDArray[float32]]]
    query_embedding: NDArray[float32] | Tensor

    class Config:
        """Pydantic configuration class to allow arbitrary types."""

        arbitrary_types_allowed = True


class BaseRetrieval(ABC):
    """Abstract base class for retrieval."""

    index: IndexFlatIP | IndexFlatL2

    def __init__(self: BaseRetrieval, config: SimpleNamespace) -> None:
        """Initialize the BaseRetrieval with a model and a configuration.

        Args:
        ----
            config (SimpleNamespace): Configuration settings for the vector store.

        """
        self.config: SimpleNamespace = config

    @abstractmethod
    def build(self: BaseRetrieval) -> None:
        """Abstract method to create an index search mechanism."""

    @abstractmethod
    def add(self: BaseRetrieval, embeddings: NDArray[float32]) -> None:
        """Add vectors to the index search.

        Args:
        ----
            embeddings (NDArray): The array of embeddings to add.

        """

    @abstractmethod
    def search(self: BaseRetrieval, input_value: SearchInput) -> SearchOutput:
        """Search for the top_k closest embeddings.

        Args:
        ----
            input_value (SearchInput): An instance of SearchInput containing the search parameters:
                - "top_k" (int): The number of nearest neighbors to retrieve.
                - "sentences" (str | list[str]): The sentence or list of sentences to be encoded and searched.

        Returns:
        -------
            SearchOutput: An instance of SearchOutput containing the results and query embedding:
                - "results" (list[tuple]): A list of tuples containing the name, distance,
                and embedding of each nearest neighbor.
                - "query_embedding" (NDArray[float32]): The embedding of the query sentence(s).

        """
