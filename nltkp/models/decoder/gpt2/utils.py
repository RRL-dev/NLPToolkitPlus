"""Module providing functionality to handle cache model outputs."""

from dataclasses import dataclass
from typing import Any

from torch import Tensor, arange, argmax, cat, long, ones, tensor


@dataclass
class CacheModelOutputs:
    """Cache for model's outputs that may also contain a past key/values.

    Attributes
    ----------
        last_hidden_state (Tensor | None):
            Sequence of hidden-states at the output of the last layer of the model, shaped as
            `(batch_size, sequence_length, hidden_size)`.
        past_key_values (tuple[tuple[Tensor, ...], ...] | None, optional):
            Tuple of tuples, each containing pre-computed hidden-states (keys and values) from the self-attention
            blocks used to speed up sequential decoding. This is returned when `use_cache=True`.
        logits (Tensor | None):
            The logits or raw scores output by the model's last layer, typically shaped as
            `(batch_size, sequence_length, vocab_size)`. These scores are passed through a softmax layer to
            obtain probability distributions over possible tokens for each position in the sequence.

    """  # noqa: E501

    logits: Tensor | None
    past_key_values: tuple[tuple[Tensor, ...], ...] | None = None
    last_hidden_state: Tensor | None = None


def update_model_kwargs_for_generation(
    cache_output: CacheModelOutputs,
    model_kwargs: dict[str, Any],
    num_new_tokens: int = 1,
) -> dict[str, Any]:
    """Update model keyword arguments for the next generation step.

    Args:
    ----
        cache_output (Any): The output from the model's forward pass containing past_key_values.
        model_kwargs (dict[str, Any]): The current model keyword arguments.
        num_new_tokens (int, optional): Number of new tokens generated. Defaults to 1.

    Returns:
        dict[str, Any]: Updated model keyword arguments.

    """
    # update past_key_values
    past_key_values: tuple[tuple[Tensor, ...], ...] | None = getattr(
        cache_output,
        "past_key_values",
        None,
    )

    model_kwargs["past_key_values"] = past_key_values

    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        if isinstance(attention_mask, Tensor):
            model_kwargs["attention_mask"] = cat(
                tensors=[
                    attention_mask,
                    attention_mask.new_ones(size=(attention_mask.shape[0], 1)),
                ],
                dim=-1,
            )

    if (
        model_kwargs.get("use_cache", True)
        and "cache_position" in model_kwargs
        and model_kwargs["cache_position"] is not None
    ) and isinstance(model_kwargs["cache_position"], Tensor):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

    return model_kwargs


def get_initial_cache_position(
    input_ids: Tensor,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Calculate `cache_position` for the pre-fill stages based on `input_ids` and optionally past length.

    Args:
    ----
        input_ids (Tensor): Input IDs.
        model_kwargs (dict): A dictionary of keyword arguments for the model.

    Returns:
    -------
        dict: Updated model keyword arguments with cache position information.

    """  # noqa: E501
    if not model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = None
        return model_kwargs

    past_length = 0
    if "past_key_values" in model_kwargs:
        past_length = model_kwargs["past_key_values"][0][0].shape[2]
    cur_len = model_kwargs.get("inputs_embeds", input_ids).shape[1]

    model_kwargs["cache_position"] = arange(
        start=past_length,
        end=cur_len,
        device=input_ids.device,
    )
    return model_kwargs


def initialize_sampling(input_ids: Tensor, vocab_size: int) -> tuple[Tensor, bool, Tensor]:
    """Initialize variables for the sampling process.

    Args:
    ----
        input_ids (Tensor): Input IDs.
        vocab_size (int): Vocabulary size.

    Returns:
    -------
        tuple[Tensor, bool, Tensor]: Initialized pad token ID, finish samples flag, and unfinished sequences tensor.

    """  # noqa: E501
    batch_size = input_ids.shape[0]
    pad_token_id: Tensor = tensor(
        data=vocab_size - 1,
        device=input_ids.device,
    )

    finish_samples: bool = False
    unfinished_sequences: Tensor = ones(
        batch_size,
        dtype=long,
        device=input_ids.device,
    )

    return pad_token_id, finish_samples, unfinished_sequences


def generate_next_token_and_update_inputs(
    cache_output: CacheModelOutputs,
    input_ids: Tensor,
    unfinished_sequences: Tensor,
    pad_token_id: Tensor,
    model_kwargs: dict[str, Any],
) -> tuple[Tensor, dict[str, Any]]:
    """Generate the next token and update the input IDs and model keyword arguments.

    Args:
    ----
        cache_output (CacheModelOutputs): The output from the model's forward pass.
        input_ids (Tensor): The current sequence of input IDs.
        unfinished_sequences (Tensor): Tensor indicating which sequences are still being generated.
        pad_token_id (Tensor): The padding token ID.
        model_kwargs (dict[str, Any]): The current model keyword arguments.
        update_model_kwargs_for_generation (Callable): Function to update model keyword arguments.

    Returns:
    -------
        tuple[Tensor, dict[str, Any]]: Updated input IDs and model keyword arguments.

    """
    if cache_output.logits is None:
        msg = "Logits from the forward pass are None."
        raise ValueError(msg)

    next_token_logits: Tensor = cache_output.logits[:, -1, :]
    next_tokens: Tensor = argmax(input=next_token_logits, dim=-1)
    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

    input_ids = cat(tensors=[input_ids, next_tokens[:, None]], dim=-1)
    model_kwargs = update_model_kwargs_for_generation(
        cache_output=cache_output,
        model_kwargs=model_kwargs,
    )

    return input_ids, model_kwargs
