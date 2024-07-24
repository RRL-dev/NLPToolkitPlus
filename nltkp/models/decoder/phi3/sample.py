"""The module provides a base class for sample generation using the nltkp.models utilities."""

from typing import TYPE_CHECKING, Any, cast

from torch import FloatTensor, LongTensor, Tensor, argmax, cat, empty, float16, float32, inference_mode, long, multinomial, ones
from torch.nn import Module
from torch.nn.functional import softmax
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from nltkp.models.decoder.phi3.utils import prepare_inputs_for_generation
from nltkp.models.utils import clean_module_name
from nltkp.modules.generation.criteria import StoppingCriteriaModule, TopKLogits, prepare_maximum_generation_length
from nltkp.modules.generation.mode import BaseSampleGenerator
from nltkp.modules.generation.params import SamplingParams
from nltkp.modules.generation.utils import (
    get_initial_cache_position,
    has_unfinished_sequences,
    prepare_generation_config,
    update_model_params_for_generation,
)

if TYPE_CHECKING:
    from transformers.generation.stopping_criteria import StoppingCriteriaList
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .causal import Phi3ForCausalLM


class Phi3Sample(Module, BaseSampleGenerator):
    """Base class for sample generation using the nltkp.models utilities.

    This class provides methods to generate samples using top-k logits warping
    and stopping criteria.
    """

    def __init__(self, pretrained_model_name: str) -> None:
        """Initialize the BaseSampleGenerator with the given model name and generation parameters.

        Args:
        ----
            pretrained_model_name: The name of the pretrained model.
            generation_params: Parameters for generation, including top_k, filter_value, etc.

        """
        super().__init__()
        self.generation_config: GenerationConfig
        self._module_name: str = clean_module_name(name=pretrained_model_name)
        self._pretrained_model_name: str = pretrained_model_name

        self.model_config: PretrainedConfig = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path=self._pretrained_model_name,
        )

        self.generation_config = prepare_generation_config(
            pretrained_model_name=self._pretrained_model_name,
        )

        self.model_params = SamplingParams()

        self.set_llm()
        self.set_tokenizer()

    def set_llm(self) -> None:
        """Set the language model for causal language modeling.

        This method loads the pretrained model using the provided model name and configuration,
        and assigns it to the `model` attribute of the instance.

        """
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=float16)

        model = Phi3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._pretrained_model_name,
            config=self.model_config,
            quantization_config=bnb_config,
        )

        model = cast(Module, model)

        self.register_module(
            name=self._module_name,
            module=model,
        )

        self.eval()

    def set_tokenizer(self) -> None:
        """Set tokenizer.

        This method initializes the tokenizer if it hasn't been set already and returns it.
        """
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._pretrained_model_name,
        )

    def set_criteria(self) -> None:
        """Set the stopping criteria and top-k logits based on the model configuration."""
        self.stopping_criteria: StoppingCriteriaList = StoppingCriteriaModule(
            model_config=self.model_config,
        ).get_stopping_criteria(
            generation_config=self.generation_config,
            stopping_criteria=None,
            tokenizer=None,
        )

        self.top_k_logits = TopKLogits(
            top_k=self.generation_config.top_k,
        )

    def generate(
        self,
        sentence: str | None = None,
    ) -> None:
        """Generate samples based on model configurations.

        Args:
        ----
            sentence (str): The input sentence to generate samples from.
            model_params (SamplingParams): Parameters for the sampling process.

        """
        if self.model_params.tokens_ids is not None and sentence is not None:
            inputs: BatchEncoding = self.tokenizer(sentence, return_tensors="pt")
            self.model_params.tokens_ids = inputs.input_ids
            self.model_params.attention_mask = inputs.attention_mask

        self.generation_config = prepare_maximum_generation_length(
            model_params=self.model_params,
            generation_config=self.generation_config,
        )

        self.set_criteria()
        self.sample()

    @inference_mode()
    def sample(self) -> None:
        """Perform sampling using prepared model inputs.

        Args:
        ----
            model_params (SamplingParams): Parameters for the sampling process.

        Raises:
        ------
            TypeError: If input_ids is not provided or not a Tensor.

        """
        tokens_ids: Tensor | None = self.model_params.tokens_ids

        # Check if input_ids is not None and is a Tensor
        if not isinstance(tokens_ids, Tensor):
            msg = "input_ids must be provided and be a Tensor"
            raise TypeError(msg)

        batch_size: int = tokens_ids.shape[0]
        empty_scores: FloatTensor = cast(FloatTensor, empty(size=(0,), dtype=float32))

        this_peer_finished: bool | Tensor = False
        unfinished_sequences: Tensor = ones(batch_size, dtype=long, device=tokens_ids.device)
        has_eos_stopping_criteria: bool = any(hasattr(criteria, "eos_token_id") for criteria in self.stopping_criteria)

        get_initial_cache_position(model_params=self.model_params)

        while has_unfinished_sequences(
            this_peer_finished=this_peer_finished,
            synced_gpus=False,
            device=tokens_ids.device,
        ):
            logits: Tensor
            kv_cache: tuple[tuple[FloatTensor]] | None

            model_inputs: dict[str, Any] = prepare_inputs_for_generation(model_params=self.model_params)
            logits, kv_cache = self.get_submodule(target=self._module_name)(**model_inputs)
            if this_peer_finished:
                continue

            next_token_logits: Tensor = logits[:, -1, :]

            next_token_scores: Tensor = self.process_next_token_scores(next_token_logits=next_token_logits)

            next_tokens: Tensor = self.sample_next_tokens(next_token_scores=next_token_scores)

            if has_eos_stopping_criteria:
                next_tokens = self.adjust_for_eos_stopping_criteria(
                    next_tokens=next_tokens,
                    unfinished_sequences=unfinished_sequences,
                )

            tokens_ids = self.update_input_ids(tokens_ids=tokens_ids, next_tokens=next_tokens)
            self.model_params: SamplingParams = self.update_model_params(
                kv_cache=kv_cache,
                model_params=self.model_params,
                tokens_ids=tokens_ids,
            )

            tokens_ids = cast(LongTensor, tokens_ids)

            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(
                input_ids=tokens_ids,
                scores=empty_scores,
            )
            this_peer_finished = unfinished_sequences.max() == 0

    def process_next_token_scores(self, next_token_logits: Tensor) -> Tensor:
        """Process the next token scores based on the generation configuration.

        Args:
        ----
            next_token_logits (Tensor): The logits for the next token.

        Returns:
        -------
            Tensor: The processed next token scores.

        """
        if self.generation_config.do_sample:
            return self.top_k_logits(next_token_logits)
        return next_token_logits

    def sample_next_tokens(self, next_token_scores: Tensor) -> Tensor:
        """Sample the next tokens based on the generation configuration.

        Args:
        ----
            next_token_scores (Tensor): The scores for the next token.

        Returns:
        -------
            Tensor: The sampled next tokens.

        """
        if self.generation_config.do_sample:
            probs: Tensor = softmax(input=next_token_scores, dim=-1)
            return multinomial(input=probs, num_samples=1).squeeze(dim=1)
        return argmax(input=next_token_scores, dim=-1)

    def adjust_for_eos_stopping_criteria(self, next_tokens: Tensor, unfinished_sequences: Tensor) -> Tensor:
        """Adjust the next tokens for EOS stopping criteria.

        Args:
        ----
            next_tokens (Tensor): The next tokens to be generated.
            unfinished_sequences (Tensor): The tensor indicating which sequences are unfinished.

        Returns:
        -------
            Tensor: The adjusted next tokens.

        """
        return next_tokens * unfinished_sequences + self.generation_config.pad_token_id * (1 - unfinished_sequences)

    def update_input_ids(self, tokens_ids: Tensor, next_tokens: Tensor) -> Tensor:
        """Update the input IDs by appending the next tokens.

        Args:
        ----
            tokens_ids (Tensor): The current input IDs.
            next_tokens (Tensor): The next tokens to be added.

        Returns:
        -------
            Tensor: The updated input IDs.

        """
        return cat(tensors=[tokens_ids, next_tokens.unsqueeze(dim=1)], dim=1)

    def update_model_params(
        self,
        kv_cache: tuple[tuple[FloatTensor]] | None,
        model_params: SamplingParams,
        tokens_ids: Tensor,
    ) -> SamplingParams:
        """Update the model parameters for the next generation step.

        Args:
        ----
            kv_cache (tuple[tuple[FloatTensor]] | None): The key-value cache for the model.
            model_params (SamplingParams): The current model parameters.
            tokens_ids (Tensor): The updated input IDs.

        Returns:
        -------
            SamplingParams: The updated model parameters.

        """
        model_params = update_model_params_for_generation(kv_cache=kv_cache, model_params=model_params)
        model_params.tokens_ids = tokens_ids
        return model_params
