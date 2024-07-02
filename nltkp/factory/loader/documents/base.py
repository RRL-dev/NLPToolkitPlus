"""Module for loading documents based on hash values using ChainedRunnable.

This module defines the LoadDocuments class, which extends ChainedRunnable to allow for
chainable operations in the document retrieval process.

Classes:
    LoadDocuments: A class for loading documents based on their hash values.
"""

from __future__ import annotations

from pathlib import Path

from nltkp.factory import ChainedRunnable
from nltkp.modules.embeddings import RankOutput
from nltkp.utils import LOGGER


class LoadDocuments(ChainedRunnable[RankOutput, list[str]]):
    """A class for loading documents based on their hash values.

    This class extends ChainedRunnable to allow for chainable operations in the
    document retrieval process.

    Attributes
    ----------
    directory : str
        The directory where the document text files are stored.

    """

    def __init__(self: LoadDocuments, directory: str) -> None:
        """Initialize the LoadDocuments with a directory path.

        Args:
        ----
            directory (str): The directory where the document text files are stored.

        """
        super().__init__(func=self.load_documents)
        self.directory: str = directory

    def load_documents(self: LoadDocuments, input_value: RankOutput) -> list[str]:
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

    def _retrieve_document_from_hash(self: LoadDocuments, hash_value: str) -> str:
        """Retrieve a document's content based on its hash value.

        Args:
        ----
            hash_value (str): The hash value of the document.

        Returns:
        -------
            str: The content of the document if found, otherwise an error message.

        """
        document_path: Path = Path(self.directory) / f"{hash_value}.txt"
        try:
            with document_path.open(mode="r", encoding="utf-8") as file:
                document: str = file.read()
            LOGGER.info("Loaded document for hash: %s", hash_value)
        except FileNotFoundError:
            LOGGER.error("Document not found for hash: %s", hash_value)
            document = f"Document not found for hash {hash_value}"
        return document
