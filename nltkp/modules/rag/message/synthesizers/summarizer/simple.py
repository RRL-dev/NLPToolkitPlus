"""The module provides an implementation of the ContentSynthesizer base called SimpleSummarizer.

It returns the provided text as is, truncated to fit within
a specified maximum context window. This class can be used in applications where
simple concatenation and truncation of input text are required without further processing.
"""

from __future__ import annotations

from nltkp.modules.rag.message.synthesizers import ContentSynthesizer


class SimpleSummarizer(ContentSynthesizer):
    """A simple implementation of the ContentSynthesizer that returns the provided text as is."""

    def generate_response(
        self: SimpleSummarizer,
        context: list[str],
    ) -> str:
        """Return the provided text chunks as a single string, respecting the maximum context window.

        Args:
        ----
            context (list[str]): A list of text chunks to summarize or return.


        Returns:
        -------
            str: The text returned as a single concatenated string.

        """
        # Concatenate text ensuring it does not exceed the max_context_window
        full_text: str = "\n".join(context)
        return full_text[: self.max_context_window]
