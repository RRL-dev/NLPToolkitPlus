"""The module defines the BaseGeneration abstract class and its configuration for interacting with pretrained models."""

from abc import ABC, abstractmethod
from typing import Any

from torch import inference_mode
from transformers.tokenization_utils_base import BatchEncoding


class BaseGeneration(ABC):
    """Abstract base class for generating text responses using a language model."""

    def __init__(self, pretrained_model_name: str) -> None:
        """Initialize the BaseGeneration with the given model name and configuration.

        Args:
        ----
            pretrained_model_name (str): The name of the pretrained model.

        """
        self.model_name: str = pretrained_model_name

    @abstractmethod
    @inference_mode()
    def _forward(self, input_info: BatchEncoding) -> None:
        """Perform a forward pass using the pre-trained language model to generate text.

        Args:
        ----
            input_info (BatchEncoding): Inputs containing token IDs and attention masks.

        """

    @abstractmethod
    def forward(self, messages: list[dict[Any, Any]]) -> list[dict[Any, Any]]:
        """Process text input to generate text responses using a pre-trained language model.

        Args:
        ----
            messages (list[dict[Any, Any]]): The text prompt to which the model generates a response.

        """
