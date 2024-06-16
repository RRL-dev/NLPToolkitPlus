"""Module for processing text into chunks using a variety of separator options.

This module includes the TextChunkProcessor class, which provides functionality
to recursively split text based on provided separators. It supports treating
separators as regular expressions and handles complex splitting strategies
that allow for nested or hierarchical text structures.
"""

from __future__ import annotations

from re import escape, search, split
from typing import Any


class TextChunkProcessor:
    """Handles the processing of text into chunks using a list of separators."""

    def __init__(self: TextChunkProcessor, chunk_size: int, is_separator_regex: bool) -> None:  # noqa: FBT001
        """Initialize the processor with settings for handling text chunks.

        Args:
        ----
            chunk_size (int): The maximum allowed size of each text chunk.
            is_separator_regex (bool): Flag indicating whether separators should be treated as regular expressions.

        """
        self.chunk_size: int = chunk_size
        self.is_separator_regex: bool = is_separator_regex

    def _split_recursive(
        self: TextChunkProcessor,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text into chunks using defined separators.

        Args:
        ----
            text (str): Text to be split.
            separators (list[str]): Separators used for splitting.

        Returns:
        -------
            list[str]: List of text chunks.

        """
        if not text:
            return []

        separator: str
        new_separators: list[str]

        separator, new_separators = self._get_separator_and_new_separators(
            text=text,
            separators=separators,
        )

        _separator: str = separator if self.is_separator_regex else escape(pattern=separator)
        splits: list[str | Any] = split(pattern=f"({_separator})", string=text)
        return self._process_splits(splits=splits, new_separators=new_separators)

    def _get_separator_and_new_separators(
        self: TextChunkProcessor,
        text: str,
        separators: list[str],
    ) -> tuple[str, list[str]]:
        """Determine the most appropriate separator and obtain new separators for further splitting.

        Args:
        ----
            text (str): Text to examine.
            separators (list[str]): List of potential separators.

        Returns:
        -------
            tuple[str, list[str]]: Chosen separator and the remaining list of separators.

        """
        for i, separator in enumerate(iterable=separators):
            _separator: str = separator if self.is_separator_regex else escape(pattern=separator)
            if search(pattern=_separator, string=text):
                return separator, separators[i + 1 :]
        return separators[-1], []

    def _process_splits(
        self: TextChunkProcessor, splits: list[str], new_separators: list[str]
    ) -> list[str]:
        """Process initial splits and further split them recursively as needed.

        Args:
        ----
            splits (list[str]): Initial list of text splits.
            new_separators (list[str]): Separators for further splitting.

        Returns:
        -------
            list[str]: Final list of processed text chunks.

        """
        final_chunks: list[str] = []
        current_chunk: list[str] = []
        total_length = 0
        separator_length: int = len(new_separators[0]) if new_separators else 0

        for _split in splits:
            split_length: int = len(_split)
            # Check if adding this split exceeds the chunk size, and process if it does
            if total_length + split_length + separator_length > self.chunk_size and current_chunk:
                final_chunks.append("".join(current_chunk))
                current_chunk = []
                total_length = 0

            current_chunk.append(_split)
            total_length += split_length

        # Add the last chunk if it's not empty
        if current_chunk:
            final_chunks.append("".join(current_chunk))

        return final_chunks
