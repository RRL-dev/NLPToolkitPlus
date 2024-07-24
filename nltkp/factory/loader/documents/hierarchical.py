"""Module for loading documents based on hash values using a chainable runnable approach.

This module defines a HierarchicalLoadDocuments class that takes hash values as input and,
retrieves the corresponding documents from a hierarchical directory structure.
The class is designed to be used in a chainable workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nltkp.utils import LOGGER

from .base import BaseLoadDocuments

if TYPE_CHECKING:
    from types import SimpleNamespace

    from nltkp.modules.embeddings import RankOutput


class HierarchicalLoadDocuments(BaseLoadDocuments):
    """Class to load documents based on hash values from a hierarchical structure."""

    def __init__(self: HierarchicalLoadDocuments, config: SimpleNamespace) -> None:
        """Initialize HierarchicalLoadDocuments with a configuration.

        Args:
        ----
            config (SimpleNamespace): Configuration containing paths and settings.

        """
        super().__init__(config=config)
        self.loaded_hashes: set[str] = set()  # To keep track of loaded hashes

    def load_documents(self: HierarchicalLoadDocuments, input_value: RankOutput) -> list[str]:
        """Load documents based on hash values provided in the input.

        Args:
        ----
            input_value (RankOutput): Model containing the hash values of the documents to load.

        Returns:
        -------
            List[str]: List containing the loaded documents.

        """
        try:
            chunk_documents: list[str] = input_value.context
        except KeyError as e:
            LOGGER.error("Key error: %s", e)
            return [f"Key error: {e!s}"]

        documents: list[str] = []
        for hash_value in chunk_documents:
            if hash_value not in self.loaded_hashes:
                document: str = self._retrieve_document_from_hash(hash_value=hash_value)
                if document:
                    documents.append(document)
                    self.loaded_hashes.add(hash_value)
        return documents

    def _retrieve_document_from_hash(self: HierarchicalLoadDocuments, hash_value: str) -> str:
        """Retrieve a document's content based on its hash value from a hierarchical structure.

        Args:
        ----
            hash_value (str): The hash value of the document.

        Returns:
        -------
            str: The content of the document if found, otherwise an error message.

        """
        chunks_path: str = self.config.chunks
        try:
            hash_relative_path: Path = Path(hash_value).relative_to(self.config.embeddings)
            document_directory: Path = Path(chunks_path) / hash_relative_path
        except ValueError:
            LOGGER.error("Invalid hash value: %s", hash_value)
            return f"Invalid hash value: {hash_value}"

        try:
            # Find the file starting with "original"
            document_path: Path = next(document_directory.glob(pattern="original*.txt"))
            with document_path.open(mode="r", encoding="utf-8") as file:
                document: str = file.read()
        except StopIteration:
            LOGGER.error("Document not found for hash: %s in path: %s", hash_value, document_directory)
            return f"Document not found for hash {hash_value}"
        except FileNotFoundError:
            LOGGER.error("Document file not found at path: %s", document_directory)
            return f"Document file not found at path {document_directory}"
        else:
            LOGGER.info("Loaded document for hash: %s from path: %s", hash_value, document_path)
            return document

    def __repr__(self: HierarchicalLoadDocuments) -> str:
        """Provide an unambiguous string representation of the HierarchicalLoadDocuments object.

        Returns
        -------
        str: A string representation of the HierarchicalLoadDocuments instance.

        """
        return f"HierarchicalLoadDocuments(config={self.config})"
