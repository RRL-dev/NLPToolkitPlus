"""Module for handling text file related exceptions.

This module defines a custom exception class for handling specific errors that
can occur during text file operations, such as file not found or invalid format errors.
"""

from __future__ import annotations


class TextFileError(Exception):
    """Custom exception for errors related to text file operations."""

    def __init__(self: TextFileError, message: str) -> None:
        """Initialize the TextFileError with a specific error message.

        Args:
        ----
        message : str
            The message describing the error.

        """
        super().__init__(message)
