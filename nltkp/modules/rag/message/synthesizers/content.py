"""The module defines the ContentSynthesizer abstract base class (ABC).

Classes:
    ContentSynthesizer (ABC): Provides a framework for synthesizing responses from textual context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ContentSynthesizer(ABC):
    """Abstract base class for synthesizing content based on a provided context window."""

    def __init__(self: ContentSynthesizer, max_context_window: int = 8000) -> None:
        """Initialize the synthesizer with the maximum context window for the language model.

        Args:
        ----
            max_context_window (int): The maximum number of tokens the language model can handle in a single request.

        """
        self.max_context_window: int = max_context_window

    @abstractmethod
    def generate_response(
        self: ContentSynthesizer,
        context: list[str],
    ) -> str:
        """Generate a response based on the provided context.

        Args:
        ----
            context (list[str]): Contextual information to help generate the response.

        Returns:
        -------
            str: The generated response.

        """
        msg = "This method needs to be overridden in subclasses."
        raise NotImplementedError(msg)
