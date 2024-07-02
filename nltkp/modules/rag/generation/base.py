"""The module defines an abstract base class for generation systems that utilize both retrieval mechanisms.

It provides a framework for combining vector-based retrieval systems with neural network models,
to generate contextually relevant responses based on user queries.

Classes:
    LLMConfig: Configuration class for specifying the parameters of the language model.
    GenerationConfig: Aggregated configuration class that holds settings for the entire generation system.
    BaseGeneration: Abstract base class that defines the core functionality for a text generation system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nltkp.models.chat import BaseChatModel


@dataclass
class BaseGeneration(ABC):
    """Abstract base class for a generation system that utilizes a retrieval mechanism and a language model generation.

    Attributes
    ----------
        config (ConfigType): Configuration settings for the generation system.
        instruction (BaseChatModel): The neural network model used for generating responses.

    """

    instruction: BaseChatModel = field(repr=False, compare=False)

    @abstractmethod
    def generate(self: BaseGeneration, message: list[dict[str, str]]) -> str:
        """Abstract method to generate a response based on the provided query.

        Args:
        ----
            message (str): The user query or input message.

        Returns:
        -------
            str: The generated response from the system.

        """
        msg: str = "Subclasses must implement this method to generate responses."
        raise NotImplementedError(msg)
