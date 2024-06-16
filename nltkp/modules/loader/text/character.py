"""Module for splitting text into chunks using different strategies.

This module provides classes for splitting text into chunks with options for recursive splitting
and handling separators, including options for whitespace stripping and metadata handling.
"""

from __future__ import annotations

import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

from .chunks import TextChunkProcessor
from .exception import TextFileError


class Document:
    """Class for storing content and metadata of a document, each document has a unique ID."""

    def __init__(self: Document, page_content: str, metadata: dict) -> None:
        """Initialize a new Document with content and metadata.

        Args:
        ----
            page_content (str): The content of the document.
            metadata (dict): The metadata associated with the document.

        """
        self.page_content: str = page_content
        self.metadata: dict[Any, Any] = metadata
        self.id_ = str(object=uuid.uuid4())  # Generate a unique ID for each document.

    @property
    def doc_id(self: Document) -> str:
        """Get the unique document ID."""
        return self.id_

    @property
    def hash(self: Document) -> str:
        """Generate a hash for the document based on its content and metadata."""
        doc_identity: str = str(object=self.page_content) + str(object=self.metadata)
        return sha256(string=doc_identity.encode(encoding="utf-8")).hexdigest()


class CharacterTextSplitter(TextChunkProcessor):
    """Class for splitting text by recursively looking at characters."""

    def __init__(
        self: CharacterTextSplitter,
        folder_path: str | None = None,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        is_separator_regex: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the text splitter with basic configuration.

        Args:
        ----
            folder_path (str | None): Base folder path for text processing.
            chunk_size (int): Maximum size of chunks.
            chunk_overlap (int): Overlap between chunks.
            is_separator_regex (bool): Flag to treat separators as regex.

        """
        super().__init__(chunk_size=chunk_size, is_separator_regex=is_separator_regex)
        self.folder_path: str | None = folder_path
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.is_separator_regex: bool = is_separator_regex
        self.separators: list[str] = [
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",
            "\uff0c",
            "\u3001",
            "\uff0e",
            "\u3002",
            "",
        ]

    def read_text_file(self: CharacterTextSplitter, file_path: str | Path) -> str:
        """Read a text file and return its content.

        Args:
        ----
            file_path (str | Path): The path to the text file.

        Returns:
        -------
            str: The content of the text file.

        Raises:
        ------
            TextFileError: If the file does not exist or is not a text file.

        """
        path = Path(file_path)
        if not path.exists():
            msg: str = f"The file {file_path} does not exist."
            raise TextFileError(message=msg)
        if path.suffix != ".txt":
            msg = f"The file {file_path} is not a text file."
            raise TextFileError(message=msg)

        return path.read_text(encoding="utf-8")

    def read_text_files_in_folder(self: CharacterTextSplitter) -> list[Document]:
        """Read all text files in a folder and return their content as a list of Documents.

        Args:
        ----
            folder_path (str): Path to the folder containing text files.

        Returns:
        -------
            list[Document]: A list of Documents containing the content of each file and metadata.

        """
        if self.folder_path is not None:
            folder = Path(self.folder_path)
            if not folder.exists() or not folder.is_dir():
                raise TextFileError(
                    message=f"The folder {self.folder_path} does not exist or is not a directory.",
                )
        else:
            raise TextFileError(
                message=f"The folder {self.folder_path} does not exist or is not a directory.",
            )

        documents: list[Document] = []
        for text_file in folder.glob(pattern="**/*.txt"):
            content: str = self.read_text_file(file_path=text_file)
            documents.extend(self.split_text(text=content, file_path=str(object=text_file)))
        return documents

    def split_text(
        self: CharacterTextSplitter,
        text: str,
        file_path: str,
    ) -> list[Document]:
        """Split text into chunks recursively and return them as Document objects with metadata.

        Args:
        ----
            text (str): Text to be split.
            file_path (str): Path of the file from which the text is read.

        Returns:
        -------
            list[Document]: List of Document objects containing chunks of text and their metadata.

        """
        return [
            Document(page_content=chunk, metadata={"index": i, "file_path": file_path})
            for i, chunk in enumerate(
                iterable=self._split_recursive(text=text, separators=self.separators),
            )
        ]
