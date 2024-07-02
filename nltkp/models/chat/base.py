"""The module defines the BaseChatModel class for interacting with pre-trained language models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from torch import LongTensor, Tensor, bfloat16, device, inference_mode, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from nltkp.models.utils import clean_module_name, move_tensors_to_device
from nltkp.utils import LOGGER, set_device

from .postprocess import postprocess
from .preprocess import preprocess

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class BaseChatConfig(BaseModel):
    """Configuration settings for the neural network model used for generating responses.

    Attributes
    ----------
        model_name (str): Model name for generating responses, typically a pre-trained model identifier.
        temperature (float): Softens the next token probabilities; higher temperature results in more random completions.
        top_k (int): Filters the proposed tokens to only the top k before applying softmax, providing randomness.
        top_p (float): Nucleus sampling that selects the smallest set of tokens whose cumulative probability is above p.
        max_new_tokens (int): Maximum number of new tokens to generate.
        do_sample (bool): Whether or not to use sampling; set to true to have more diverse responses.

    """  # noqa: E501

    model_name: str = Field(default=..., description="Model name for generating responses.")
    temperature: float = Field(default=0.9, description="Temperature for response generation.")
    top_k: int = Field(default=50, description="Top K tokens to be considered in generation.")
    top_p: float = Field(default=0.95, description="Top P cumulative probability threshold in generation.")
    max_new_tokens: int = Field(default=256, description="Maximum new tokens to generate.")
    do_sample: bool = Field(default=True, description="Flag to determine if sampling is used.")


class BaseChatModel(nn.Module):
    """PyTorch module for integrating and utilizing Hugging Face's causal language models.

    This class handles the initialization, device allocation, and forward execution of models,
    ensuring that they are properly set up for inference tasks.
    """

    def __init__(self: BaseChatModel, config: BaseChatConfig) -> None:
        """Initialize a BaseChatModel instance with the specified model name.

        Args:
        ----
            config (str): The model configuration.

        The constructor initializes the tokenizer and loads the model,
        setting up the device configuration for running model inference.

        """
        super().__init__()
        self.config: BaseChatConfig = config
        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None

        self.model_name: str = clean_module_name(name=config.model_name)
        self.module_name: str = config.model_name
        self.register_model()

        self.device: device = set_device()
        self.to(device=self.device)
        self.eval()

    @property
    def tokenizer(self: BaseChatModel) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """Lazy load and cache the tokenizer used for preprocessing text.

        Returns
        -------
            PreTrainedTokenizer | PreTrainedTokenizerFast:
            A tokenizer instance compatible with the specified model.

        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.module_name,
            )
        return self._tokenizer

    def register_model(self: BaseChatModel) -> None:
        """Register the neural network model from Hugging Face Transformers.

        This method dynamically loads a pre-trained model specified by `self.model_name`
        and registers it as a module in this PyTorch module. It allows the model to be
        part of this module's network graph for proper parameter management and GPU allocation.
        """
        try:
            # Attempt to load the model from the Hugging Face repository
            self.register_module(
                name=self.model_name,
                module=AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.module_name,
                    torch_dtype=bfloat16,
                    device_map="auto",
                ),
            )
            LOGGER.info("Model %s registry successfully.", self.module_name)

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

    def forward_params(self: BaseChatModel) -> dict[str, Any]:
        """Preprocess the ChatConfig to generate model arguments.

        Returns
        -------
            dict[str, Any]: A dictionary containing the model arguments for generation.

        """
        return {
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
        }

    @inference_mode()
    def _forward(
        self: BaseChatModel,
        model_inputs: BatchEncoding,
        **forward_params: dict,
    ) -> dict:
        """Perform a sub forward pass using the pre-trained language model to generate text.

        Args:
        ----
            model_inputs (BatchEncoding): Inputs containing token IDs and attention masks.
            **forward_params (dict): Additional parameters for the model's generate method.

        Returns:
        -------
            dict: A dictionary containing the generated sequence,
            original input ids, and prompt text.

        """
        # Move model inputs to the appropriate device (GPU/CPU)
        model_inputs = move_tensors_to_device(batch_encoding=model_inputs, device=self.device)

        input_ids: LongTensor | None = model_inputs.get("input_ids", None)
        attention_mask: LongTensor | None = model_inputs.get("attention_mask", None)

        if isinstance(input_ids, Tensor):
            if input_ids.shape[1] == 0:
                LOGGER.warning("Empty input_ids received; returning empty output.")
                return {"generated_sequence": [], "input_ids": input_ids, "prompt_text": ""}

            generated_sequence: Tensor = self.get_submodule(target=self.model_name).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **forward_params,
            )

            input_batch_size = input_ids.shape[0]
            output_batch_size = generated_sequence.shape[0]
            generated_sequence = generated_sequence.view(
                input_batch_size,
                output_batch_size // input_batch_size,
                -1,
            )
        else:
            LOGGER.warning("Unsupported input type for input_ids; expected Tensor.")
            return {"generated_sequence": [], "input_ids": input_ids, "prompt_text": ""}

        prompt_text: str = model_inputs.get("prompt_text", "")
        return {
            "generated_sequence": generated_sequence.to("cpu"),
            "input_ids": input_ids,
            "prompt_text": prompt_text,
        }

    def forward(
        self: BaseChatModel,
        messages: list[dict[str, str]],
    ) -> list[dict[Any, Any]]:
        """Process text input to generate text responses using a pre-trained language model.

        This method orchestrates the preprocessing of the input text, passing the processed inputs
        to the model's forward pass, and postprocessing the outputs to generate human-readable text.

        Args:
        ----
            messages (list[dict[str, str]]): The text prompt to which the model generate a response.

        Returns:
        -------
            list[dict[Any, Any]]: A list of dictionaries containing the postprocessed model outputs,
                                which typically include generated text and other relevant data.

        Example:
        -------
            >>> model = BaseChatModel(model_name="gpt2")
            >>> response = model.forward("Hello, how are you?")
            >>> print(response)
            [{'generated_text': 'I am fine. Thank you!'}]

        """
        model_inputs: BatchEncoding = preprocess(
            prefix="",
            messages=messages,
            tokenizer=self.tokenizer,
        )

        model_outputs: dict[str, str | Tensor] = self._forward(
            model_inputs=model_inputs,
            **self.forward_params(),
        )
        return postprocess(tokenizer=self.tokenizer, model_outputs=model_outputs)
