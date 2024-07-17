"""The module provides a StoppingCriteriaModule class for handling stopping criteria."""

from typing import TYPE_CHECKING, Any

from transformers import PreTrainedTokenizerBase
from transformers.generation.configuration_utils import GenerationConfig, PretrainedConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

from nltkp.modules.generation.params import SamplingParams
from nltkp.utils import LOGGER

if TYPE_CHECKING:
    from torch import Tensor


class StoppingCriteriaModule:
    """A module to handle stopping criteria in text generation processes.

    This class provides methods to generate a list of stopping criteria based on
    the generation configuration and merge custom stopping criteria with default ones.
    """

    def __init__(self, model_config: PretrainedConfig) -> None:
        """Initialize the StoppingCriteriaModule with the given configuration.

        Args:
        ----
            model_config: The configuration for the stopping criteria.

        """
        self.model_config: PretrainedConfig = model_config

    def get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: StoppingCriteriaList | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> StoppingCriteriaList:
        """Generate a StoppingCriteriaList based on the generation configuration.

        Args:
        ----
            generation_config: Configuration for the generation process.
            stopping_criteria: A list of custom stopping criteria.
            tokenizer: The tokenizer to be used for stop string criteria.

        Returns:
        -------
            A StoppingCriteriaList with the appropriate criteria based on the generation configuration.

        """
        criteria = StoppingCriteriaList()

        if generation_config.max_length is not None:
            max_position_embeddings: Any | None = getattr(self.model_config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                ),
            )

        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))

        if generation_config.stop_strings is not None:
            if tokenizer is None:
                msg = (
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
                raise ValueError(
                    msg,
                )
            criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))

        if generation_config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))

        return self._merge_criteria_processor_list(default_list=criteria, custom_list=stopping_criteria)

    def _merge_criteria_processor_list(
        self,
        default_list: LogitsProcessorList | StoppingCriteriaList,
        custom_list: LogitsProcessorList | StoppingCriteriaList | None,
    ) -> StoppingCriteriaList:
        """Merge the default and custom criteria processor lists.

        Args:
        ----
            default_list: The default list of criteria or logits processors.
            custom_list: A custom list of criteria or logits processors.

        Returns:
        -------
            A merged list of stopping criteria.

        """
        msg: str
        if isinstance(default_list, LogitsProcessorList) or (
            custom_list is not None and isinstance(custom_list, LogitsProcessorList)
        ):
            msg = "LogitsProcessorList is not compatible with StoppingCriteriaList."
            raise ValueError(msg)

        if custom_list is None or len(custom_list) == 0:
            return default_list

        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    msg = (
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `.generate()` instead of using a custom {object_type}."
                    )
                    raise ValueError(msg)
        default_list.extend(custom_list)
        return default_list


def prepare_maximum_generation_length(
    model_params: SamplingParams,
    generation_config: GenerationConfig,
) -> GenerationConfig:
    """Prepare max and min length in generation configs to avoid clashes between similar attributes.

    Args:
    ----
        model_params (SamplingParams): Model parameters which include input tensors and related settings.
        generation_config (GenerationConfig): Configuration for generation settings.

    Returns:
    -------
        GenerationConfig: Updated generation configuration with correct max length settings.

    """
    tokens_ids: Tensor | None = model_params.tokens_ids
    if tokens_ids is None:
        msg = "input_ids must be provided in model_params."
        raise ValueError(msg)

    input_ids_length: int = tokens_ids.shape[-1]
    has_default_max_length: bool = generation_config.max_length is not None

    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            LOGGER.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length

    return generation_config
