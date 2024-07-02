"""The module defines the Sentence base model, which integrates the llm models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from numpy import dtype, ndarray, stack
from torch import Tensor, cat, device, inference_mode, nn
from transformers import AutoModel, AutoTokenizer

from nltkp.utils import LOGGER, set_device

from .pooling import BasePooling

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class BaseSentenceModel(nn.Module):
    """A neural network module integrating a pre-trained transformer model for sentence embeddings.

    The BaseSentenceModel class wraps a transformer model specified by `model_name`,
    managing both the model initialization and embedding generation processes.

    Attributes
    ----------
        model_name (str): Identifier for the pre-trained model from Hugging Face Transformers.
        _tokenizer (AutoTokenizer): Tokenizer corresponding to the transformer model.
        pooling (BasePooling): An instance of BasePooling that applies different pooling strategies.

    Methods
    -------
        encode(sentences, batch_size, convert_to_numpy, normalize_embeddings):
            Encodes provided sentences into embeddings using the transformer model and pooling.

    Example usage:
        model = BaseSentenceModel(model_name="multi-qa-mpnet-base-dot-v1")
        embeddings = model.encode(
            ["Hello, world!", "How are you?"],
            batch_size=2,
            convert_to_numpy=True
            )

    """

    def __init__(self: BaseSentenceModel, model_name: str, pooling_modes: dict[str, bool]) -> None:
        """Initialize the BaseSentenceModel with a model name and pooling modes.

        Args:
        ----
            model_name (str): Identifier for the pre-trained model from Hugging Face Transformers.
            pooling_modes (dict[str, bool]): Dictionary specifying pooling strategies.

        Raises:
        ------
            RuntimeError: If the model cannot be loaded due to an I/O error or configuration error.

        """
        super().__init__()
        LOGGER.info("Initializing BaseSentenceModel with model: %s", model_name)
        LOGGER.info("The pooling modes are: %s", pooling_modes)

        self.pooling = BasePooling(**pooling_modes)

        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None

        self.model_name: str = model_name
        self.register_model()

        self.device: device = set_device()
        self.to(device=self.device)

    @property
    def tokenizer(self: BaseSentenceModel) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Lazy load and cache the tokenizer used for preprocessing text.

        Returns
        -------
            PreTrainedTokenizer | PreTrainedTokenizerFast:
            A tokenizer instance compatible with the specified model.

        """
        if self._tokenizer is None:
            LOGGER.info("Loading tokenizer for model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
            )
        return self._tokenizer

    def register_model(self: BaseSentenceModel) -> None:
        """Register the neural network model from Hugging Face Transformers.

        This method dynamically loads a pre-trained model specified by `self.model_name`
        and registers it as a module in this PyTorch module. It allows the model to be
        part of this module's network graph for proper parameter management and GPU allocation.
        """
        try:
            # Attempt to load the model from the Hugging Face repository
            self.register_module(
                name=self.model_name,
                module=AutoModel.from_pretrained(
                    pretrained_model_name_or_path=self.model_name,
                ),
            )
            LOGGER.info("Model %s registered successfully.", self.model_name)

        except OSError as e:
            LOGGER.error("Failed to load the model due to I/O error: %s", e)
            msg = "Model could not be loaded due to an I/O error.\n"
            "Please check the model path and internet connection."
            raise RuntimeError(
                msg,
            ) from e

        except ValueError as e:
            LOGGER.error("Configuration error when loading the model: %s", e)
            msg = "Model configuration error. Please check the model name and other parameters."
            raise RuntimeError(
                msg,
            ) from e

    def forward(self: BaseSentenceModel, features: dict[str, Tensor]) -> Tensor:
        """Perform a forward pass using the registered model to compute embeddings.

        This method processes input features, extracts token embeddings, potentially fetches
        hidden states, and applies configured pooling operations to produce a embedding tensor.

        Args:
        ----
            features (dict[str, Tensor]): A dictionary containing input tensors.

        Returns:
        -------
            Tensor: A tensor containing pooled embeddings from the model's output.

        """
        output_states: Any = self.get_submodule(target=self.model_name)(
            **features,
            return_dict=False,
        )
        output_tokens: Any = output_states[0]
        features.update(token_embeddings=output_tokens, attention_mask=features["attention_mask"])

        if self.get_submodule(target=self.model_name).config.output_hidden_states:
            all_layer_idx: Literal[2, 1] = 2 if len(output_states) >= 3 else 1  # noqa: PLR2004
            hidden_states: Any = output_states[all_layer_idx]
            features.update(all_layer_embeddings=hidden_states)

        embeddings: list[Tensor] = self.pooling.apply_pooling(output_vectors=[], features=features)

        return cat(tensors=embeddings, dim=1)

    def encode(
        self: BaseSentenceModel,
        sentences: str | list[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,  # noqa: FBT001, FBT002
        normalize_embeddings: bool = True,  # noqa: FBT001, FBT002
    ) -> ndarray[Any, dtype[Any]] | Tensor:
        """Encode a list of sentences into embeddings using a pre-trained model.

        Args:
        ----
            sentences (str | list[str]): A single sentence or a list of sentences to encode.
            batch_size (int): The number of sentences to process in each batch.
            convert_to_numpy (bool): Flag to determine if the output should be converted to numpy arrays.
            device (str | None): The device to perform the computation on. Defaults to 'cpu'.
            normalize_embeddings (bool): Whether to L2-normalize the embeddings.

        Returns:
        -------
            list[Tensor] | ndarray | Tensor: The embeddings as a list of tensors, a single tensor, or a numpy array,
                                            depending on the `convert_to_numpy` flag.

        """
        LOGGER.info("Encoding sentences with model: %s", self.model_name)
        if isinstance(sentences, str):
            sentences = [sentences]

        self.eval()

        all_embeddings: list[Tensor] = []
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch: list[str] = sentences[start_index : start_index + batch_size]
            encoded_input: BatchEncoding = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            encoded_input = encoded_input.to(device=self.device)
            with inference_mode():
                embeddings: Tensor = self(encoded_input)

            if normalize_embeddings:
                embeddings = nn.functional.normalize(input=embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        if convert_to_numpy:
            # Convert all embeddings to NumPy arrays
            LOGGER.info("Converting embeddings to numpy arrays.")
            return stack(arrays=[emb.cpu().numpy() for emb in all_embeddings])
        # Concatenate all tensors to form a single tensor
        return cat(tensors=all_embeddings, dim=0)
