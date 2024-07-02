"""The module defines the ContentSynthesizer base class using Pydantic BaseModel.

Classes:
    ContentSynthesizer (BaseModel): Provides a framework for synthesizing responses from textual context.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ContentSynthesizer(BaseModel):
    """Base class for synthesizing content based on a provided context window.

    Attributes
    ----------
    max_context_window : int
        Maximum number of tokens the language model can handle in a single request.
    context : list[str]
        Contextual information to help generate the response.

    """

    max_context_window: int = Field(
        default=8000,
        description="Maximum number of tokens the language model can handle in a single request.",
    )
    context: list[str] = Field(
        default_factory=list,
        description="Contextual information to help generate the response.",
    )

    class Config:
        """Pydantic configuration class to allow arbitrary types."""

        arbitrary_types_allowed = True

    @abstractmethod
    def generate_response(self: ContentSynthesizer, inputs: dict[str, Any]) -> dict[str, str]:
        """Generate a response based on the provided context.

        Args:
        ----
            inputs (dict[str, Any]): Inputs to the content synthesizer.

        Returns:
        -------
            dict[str, str]: The generated response.

        """
        msg = "This method needs to be overridden in subclasses."
        raise NotImplementedError(msg)
