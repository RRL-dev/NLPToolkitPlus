from typing import TYPE_CHECKING, Any  # noqa: D100

from transformers.cache_utils import Cache

from nltkp.modules.generation.params import SamplingParams

if TYPE_CHECKING:
    from torch import FloatTensor, Tensor


def prepare_inputs_for_generation(model_params: SamplingParams) -> dict[str, Any]:
    """Prepare the inputs for the generation step based on caching and input adjustments.

    Args:
    ----
        model_params (SamplingParams): An instance of SamplingParams containing all necessary inputs and parameters.

    Returns:
    -------
        dict: A dictionary containing modified and prepared model inputs for generation.

    """
    kv_cache: tuple[tuple[FloatTensor]] | None = model_params.kv_cache
    use_cache: bool = model_params.use_cache
    tokens_ids: Tensor | None = model_params.tokens_ids
    position_ids: Tensor | None = model_params.position_ids
    inputs_embeds: Any | None = getattr(model_params, "inputs_embeds", None)
    attention_mask: Tensor | None = model_params.attention_mask

    if tokens_ids is None:
        msg = "input_ids must be provided in model_params."
        raise ValueError(msg)

    past_length: int | None = None
    if kv_cache is not None:
        if isinstance(kv_cache, Cache):
            cache_length: int = kv_cache.get_seq_length()
            past_length = kv_cache.seen_tokens
            max_cache_length: int | None = kv_cache.get_max_length()
        else:
            cache_length = past_length = kv_cache[0][0].shape[2]
            max_cache_length = None

        if past_length is not None:
            if attention_mask is not None and attention_mask.shape[1] > tokens_ids.shape[1]:
                tokens_ids = tokens_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < tokens_ids.shape[1]:
                tokens_ids = tokens_ids[:, past_length:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and (cache_length + tokens_ids.shape[1] > max_cache_length)
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if kv_cache:
            position_ids = position_ids[:, -tokens_ids.shape[1] :]

    model_inputs: dict[str, Any] = (
        {"inputs_embeds": inputs_embeds}
        if inputs_embeds is not None and kv_cache is None
        else {"input_ids": tokens_ids}
    )
    model_inputs.update(
        {
            "kv_cache": kv_cache,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        },
    )

    return model_inputs
