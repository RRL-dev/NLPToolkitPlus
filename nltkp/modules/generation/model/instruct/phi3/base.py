"""The module defines the BasePhi3Instruct class for interacting with pre-trained language models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import Tensor, device, inference_mode
from transformers.tokenization_utils_base import BatchEncoding

from nltkp.models.decoder import Phi3Sample
from nltkp.models.utils import clean_module_name, move_tensors_to_device
from nltkp.modules.generation.model.base import BaseGeneration
from nltkp.modules.generation.model.postprocess import output_postprocess
from nltkp.modules.generation.model.preprocess import input_preprocess
from nltkp.modules.generation.params import SamplingParams
from nltkp.utils import LOGGER, set_device

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import BatchEncoding


class BasePhi3Instruct(Phi3Sample, BaseGeneration):
    """Base class for generating responses using a pre-trained language model."""

    model_params = SamplingParams()

    def __init__(self) -> None:
        """Initialize the BasePhi3Instruct with the given configuration.

        Args:
        ----
            config (Phi3InstructConfig): Configuration settings for the model.

        """
        super().__init__(pretrained_model_name=self.model_params.model_name)
        self.generation_config.update(**self.model_params.__dict__)

        self.model_name: str = clean_module_name(name=self.model_params.model_name)
        self.module_name: str = self.model_params.model_name

        self.device: device = set_device()
        self.to(device=self.device)
        self.eval()

        LOGGER.info(f"Initialized {self.__class__.__name__} with model {self.model_name} on device {self.device}.")

    @inference_mode()
    def _forward(
        self,
        input_info: BatchEncoding,
    ) -> None:
        """Handle the forward pass and generate text.

        Args:
        ----
            input_info (BatchEncoding): Encoded inputs for the model.

        """
        LOGGER.info(msg="Starting forward pass.")
        # Move model inputs to the appropriate device (GPU/CPU)
        input_info = move_tensors_to_device(batch_encoding=input_info, device=self.device)
        self.model_params.tokens_ids = input_info.input_ids
        self.model_params.attention_mask = input_info.attention_mask

        if isinstance(self.model_params.tokens_ids, Tensor):
            if self.model_params.tokens_ids.shape[1] == 0:
                LOGGER.warning(msg="Empty input_ids received; returning empty output.")

            LOGGER.info("Generating sequence.")
            self.generate()

            generated_sequence: Tensor = self.model_params.tokens_ids[:, input_info.input_ids.shape[1]:]
            self.model_params.output_ids = generated_sequence.to(device="cpu")
            self.model_params.prompt_text = input_info.prompt_text

            LOGGER.info(msg="Sequence generated successfully.")
        else:
            LOGGER.warning(msg="Unsupported input type for input_ids; expected Tensor.")

    def forward(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[Any, Any]]:
        """Process text input to generate text responses using a pre-trained language model.

        Args:
        ----
            messages (list[dict[str, str]]): The text prompt to which the model generates a response.

        Returns:
        -------
            list[dict[Any, Any]]: A list of dictionaries containing the postprocessed model outputs,
                                  which typically include generated text and other relevant data.

        """
        LOGGER.info(msg="Starting Phi3 instruct generation.")
        input_info: BatchEncoding = input_preprocess(
            prefix="",
            messages=messages,
            tokenizer=self.tokenizer,
        )

        LOGGER.info(msg="Input preprocessing completed.")

        self._forward(input_info=input_info)
        LOGGER.info(msg="Output postprocessing completed.")
        return output_postprocess(tokenizer=self.tokenizer, model_params=self.model_params)
