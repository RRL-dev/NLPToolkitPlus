"""Module for loading documents based on hash values using a chainable runnable approach.

This module defines a LoadDocuments class that takes hash values as input and retrieves the corresponding documents
from the specified directory. The class is designed to be used in a chainable workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nltkp.factory import ChainedRunnable
from nltkp.modules.embeddings import RankOutput
from nltkp.utils import LOGGER

if TYPE_CHECKING:
    from types import SimpleNamespace


class BaseLoadDocuments(ChainedRunnable[RankOutput, list[str]]):
    """Class to load documents based on hash values."""

    def __init__(self: BaseLoadDocuments, config: SimpleNamespace) -> None:
        """Initialize LoadDocuments with a configuration.

        Args:
        ----
            config (SimpleNamespace): Configuration containing paths and settings.

        """
        super().__init__(func=self.load_documents)
        self.config: SimpleNamespace = config

    def load_documents(self: BaseLoadDocuments, input_value: RankOutput) -> list[str]:
        """Load documents based on hash values provided in the input.

        Args:
        ----
            input_value (DocumentInput): Model containing the hash values of the documents to load.

        Returns:
        -------
            DocumentOutput: Model containing the loaded documents.

        """
        try:
            chunk_documents: list[str] = input_value.context
        except KeyError as e:
            LOGGER.error("Key error: %s", e)
            return [f"Key error: {e!s}"]

        else:
            documents: list[str] = [
                self._retrieve_document_from_hash(hash_value=hash_value) for hash_value in chunk_documents
            ]
            return documents

    def _retrieve_document_from_hash(self: BaseLoadDocuments, hash_value: str) -> str:
        """Retrieve a document's content based on its hash value.

        Args:
        ----
            hash_value (str): The hash value of the document.

        Returns:
        -------
            str: The content of the document if found, otherwise an error message.

        """
        document_path: Path = Path(self.config.chunks) / f"{hash_value}.txt"
        try:
            with document_path.open(mode="r", encoding="utf-8") as file:
                document: str = file.read()
            LOGGER.info("Loaded document for hash: %s", hash_value)
        except FileNotFoundError:
            LOGGER.error("Document not found for hash: %s", hash_value)
            document = f"Document not found for hash {hash_value}"
        return document
