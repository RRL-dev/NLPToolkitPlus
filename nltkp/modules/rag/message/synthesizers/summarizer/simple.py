"""The module provides an implementation of the ContentSynthesizer base called SimpleSummarizer.

It returns the provided text as is, concatenated into a single string.
This class can be used in applications where simple concatenation of input text chunks is required without processing.
"""

from __future__ import annotations

from nltkp.factory import ChainedRunnable


class SimpleSummarizer(ChainedRunnable[list[str], str]):
    """A simple implementation of the ContentSynthesizer that returns the provided text as is."""

    def __init__(self: SimpleSummarizer) -> None:
        """Initialize the SimpleSummarizer with optional parameters."""
        ChainedRunnable.__init__(self=self, func=self.generate_response)

    @staticmethod
    def generate_response(inputs: list[str]) -> str:
        """Return the provided text chunks as a single string.

        Args:
        ----
            inputs (list[str]): List of text chunks to concatenate.

        Returns:
        -------
            str: The concatenated text returned as a single string.

        """
        return "\n".join(inputs)
