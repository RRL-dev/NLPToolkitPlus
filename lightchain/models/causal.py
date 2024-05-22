"""A simple module for loading models and tokenizers from Hugging Face."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.tokenization_utils_base import BatchEncoding

if TYPE_CHECKING:
    from torch import Tensor
    from transformers.models.auto.auto_factory import _BaseAutoModelClass
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from the environment
hf_token: str | None = os.getenv(key="HF_TOKEN")


class BaseModelForCausalLM:
    """A simple class to load a causal language model."""

    def __init__(
        self: BaseModelForCausalLM,
        pretrained_model_name_or_path: str,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the SimpleAutoModelForCausalLM class.

        Args:
        ----
            pretrained_model_name_or_path (str): The path or name of the pretrained model.
            **kwargs (Dict[str, Any]): Additional keyword arguments for the model configuration.

        """
        config: dict[str, Any] | None = kwargs.pop("config", None)
        trust_remote_code: dict[str, Any] | None = kwargs.pop("trust_remote_code", None)

        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        if kwargs.get("torch_dtype", None) == "auto":
            kwargs["torch_dtype"] = "auto"  # type: ignore  # noqa: PGH003
        if kwargs.get("quantization_config", None) is not None:
            kwargs["quantization_config"] = kwargs["quantization_config"]

        model_class: type[None | _BaseAutoModelClass | Any] = self._get_model_class(
            config=config,  # type: ignore  # noqa: PGH003
        )

        self.model = model_class.from_pretrained( # type: ignore  # noqa: PGH003
            pretrained_model_name_or_path,
            config=config,
            **kwargs,  # type: ignore  # noqa: PGH003
        )

    @classmethod
    def from_pretrained(
        cls: type[BaseModelForCausalLM],
        pretrained_model_name_or_path: str,
        *model_args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> BaseModelForCausalLM:
        """Load the pretrained model.

        Args:
        ----
            pretrained_model_name_or_path (str): The path or name of the pretrained model.
            *model_args: Additional arguments for the model.
            **kwargs (Dict[str, Any]): Additional keyword arguments for the model.

        Returns:
        -------
            SimpleAutoModelForCausalLM: An instance of the SimpleAutoModelForCausalLM class.

        """
        token: dict[str, Any] | str | None = kwargs.pop("use_auth_token", hf_token)
        if isinstance(token, str):
            kwargs["use_auth_token"] = token  # type: ignore  # noqa: PGH003
        return cls(pretrained_model_name_or_path, *model_args, **kwargs)

    def _get_model_class(self: BaseModelForCausalLM, config: PretrainedConfig) -> type[Any]:
        supported_models = MODEL_FOR_CAUSAL_LM_MAPPING[type(config)]
        if not isinstance(supported_models, list | tuple):
            return supported_models

        name_to_model: dict[str, type[Any]] = {model.__name__: model for model in supported_models}

        architectures: list[str] = getattr(config, "architectures", [])
        for arch in architectures:
            if arch in name_to_model:
                return name_to_model[arch]
            if f"TF{arch}" in name_to_model:
                return name_to_model[f"TF{arch}"]
            if f"Flax{arch}" in name_to_model:
                return name_to_model[f"Flax{arch}"]

        return supported_models[0]

    def __call__(
        self: BaseModelForCausalLM, input_ids: BatchEncoding | dict[str, Tensor]
    ) -> Tensor:
        """Call the model.

        Raises
        ------
            NotImplementedError: If model inference without PyTorch is not implemented.

        """
        if isinstance(input_ids, BatchEncoding):
            input_ids = input_ids.data  # Convert BatchEncoding to dict[str, Tensor]
        return self.model.generate(**input_ids)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    causal_model: BaseModelForCausalLM = BaseModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="google/gemma-2b",
    )

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/gemma-2b",
    )

    input_text = "Write me a poem about Machine Learning."
    input_ids: BatchEncoding = tokenizer(input_text, return_tensors="pt")

    outputs: Tensor = causal_model(input_ids)
