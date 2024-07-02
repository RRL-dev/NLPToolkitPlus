"""Module for generating responses to user queries using a chat model.

This module defines the ChatGeneration class which uses a neural network model
to generate responses to user queries based on provided messages.

Classes:
    ChatGeneration: Generates responses using a configured chat model.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nltkp.factory import ChainedRunnable
from nltkp.utils import LOGGER

from .base import BaseGeneration


@dataclass
class ChatGeneration(BaseGeneration, ChainedRunnable[list[dict[str, str]], str]):
    """Class for generating responses to user queries using a chat model.

    Attributes
    ----------
    config : ConfigType
        Configuration settings for the generation system.
    instruction : BaseChatModel
        The neural network model used for generating responses.

    """

    def __post_init__(self: ChatGeneration) -> None:
        """Post-initialization to set up the chained runnable."""
        ChainedRunnable.__init__(self=self, func=self.generate)


    def generate(self: ChatGeneration, message: list[dict[str, str]]) -> str:
        """Generate a response to a query using the chat model.

        Args:
        ----
            message (list[dict[str, str]]): The list of messages to process and generate a response from.

        Returns:
        -------
            str: The generated response from the system.

        """
        generated_text: str = ""
        try:
            response = self.instruction(message)
            generated_text = self.extract_generated_text(response=response)

        except Exception as e:
            LOGGER.error("Error during invoke: %s", e)
            raise
        else:
            if not generated_text:
                LOGGER.error("Generated text is empty.")
                return "An error occurred during response generation."
            return generated_text

    def extract_generated_text(self: ChatGeneration, response: list[dict[str, Any]]) -> str:
        """Extract the generated text from the model response.

        Args:
        ----
            response (list[dict[str, Any]]): The response from the model.

        Returns:
        -------
            str: The extracted generated text.

        """
        for result in response:
            if "generated_text" in result:
                return result["generated_text"]
        return ""
