"""Module provides an abstract base class for sample generation using the nltkp.models utilities."""

from abc import ABC, abstractmethod

from transformers import GenerationConfig, PretrainedConfig


class BaseSampleGenerator(ABC):
    """Abstract base class for sample generation using the nltkp.models utilities."""

    def __init__(self, pretrained_model_name: str) -> None:
        """Initialize the BaseSampleGenerator with the given model name and generation parameters.

        Args:
        ----
            pretrained_model_name: The name of the pretrained model.

        """
        self.generation_config: GenerationConfig
        self._pretrained_model_name: str = pretrained_model_name

        self.model_config: PretrainedConfig = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path=self._pretrained_model_name,
        )

        self.set_llm()
        self.set_tokenizer()

    @abstractmethod
    def set_llm(self) -> None:
        """Set the language model for causal language modeling."""

    @abstractmethod
    def set_tokenizer(self) -> None:
        """Set the tokenizer."""

    @abstractmethod
    def set_criteria(self) -> None:
        """Set the stopping criteria and top-k logits based on the model configuration."""

    @abstractmethod
    def generate(self) -> None:
        """Generate samples based on model configurations."""

    @abstractmethod
    def sample(self) -> None:
        """Perform sampling using prepared model inputs."""
