"""The module defines the BaseChatModel class for interacting with pre-trained language models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


class BaseChatModel(nn.Module):
    """PyTorch module for integrating and utilizing Hugging Face's causal language models.

    This class handles the initialization, device allocation, and forward execution of models,
    ensuring that they are properly set up for inference tasks.
    """

    def __init__(self: BaseChatModel, model_name: str) -> None:
        """Initialize a BaseChatModel instance with the specified model name.

        Args:
        ----
            model_name (str): The name of the model to load.

        The constructor initializes the tokenizer and loads the model,
        setting up the device configuration for running model inference.

        """
        super().__init__()
        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None

        self.model_name: str = clean_module_name(name=model_name)
        self.module_name: str = model_name
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
        **forward_params: dict,
    ) -> list[dict[Any, Any]]:
        """Process text input to generate text responses using a pre-trained language model.

        This method orchestrates the preprocessing of the input text, passing the processed inputs
        to the model's forward pass, and postprocessing the outputs to generate human-readable text.

        Args:
        ----
            messages (list[dict[str, str]]): The text prompt to which the model generate a response.
            **forward_params (dict): Additional keyword arguments to the model's generate method.

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
            **forward_params,
        )
        return postprocess(tokenizer=self.tokenizer, model_outputs=model_outputs)
