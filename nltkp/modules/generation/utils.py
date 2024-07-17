from typing import Any  # noqa: D100, F401

import torch.distributed as dist
from torch import FloatTensor, Tensor, arange, cat, device, tensor
from transformers.generation.configuration_utils import GenerationConfig, PretrainedConfig

from nltkp.modules.generation.params import SamplingParams


def prepare_generation_config(
    generation_config: GenerationConfig | None = None,
    *,
    pretrained_model_name: str = "microsoft/Phi-3-mini-128k-instruct",
    model_config: PretrainedConfig | None = None,
) -> GenerationConfig:
    """Prepare the generation configuration.

    Args:
    ----
    generation_config : GenerationConfig | None
        The initial generation configuration.
    pretrained_model_name : str, optional
        The name of the pretrained model to load the configuration from.
    model_config : PretrainedConfig | None, None
        The model configuration to load the generation configuration from.
    **kwargs : Dict
        Additional keyword arguments to update the generation configuration.

    Returns:
    -------
    tuple[GenerationConfig, dict]
        The updated generation configuration and model keyword arguments.

    """
    if generation_config is None:
        if model_config is not None:
            generation_config = GenerationConfig.from_model_config(model_config=model_config)
        else:
            # Load from a pretrained model
            generation_config = GenerationConfig.from_pretrained(pretrained_model_name=pretrained_model_name)
    # Create a deep copy and update with additional kwargs
    # generation_config = copy.deepcopy(generation_config)  # noqa: ERA001

    return generation_config


def has_unfinished_sequences(*, this_peer_finished: bool | Tensor, synced_gpus: bool, device: device) -> bool:
    """Check if there are unfinished sequences across GPUs.

    Args:
    ----
        this_peer_finished (bool | Tensor): Indicates if the current peer has finished processing.
        synced_gpus (bool): If True, synchronize across GPUs to check the status of all peers.
        device (torch.device): The device to use for tensor operations.

    Returns:
    -------
        bool: True if there are unfinished sequences, False otherwise.

    """
    if isinstance(this_peer_finished, Tensor):
        this_peer_finished = bool(this_peer_finished)

    if synced_gpus:
        # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
        # The following logic allows an early break if all peers finished generating their sequence.
        this_peer_finished_flag: Tensor = tensor(data=0.0 if this_peer_finished else 1.0).to(device=device)
        # Send 0.0 if we finished, 1.0 otherwise
        dist.all_reduce(tensor=this_peer_finished_flag, op=dist.ReduceOp.SUM)
        # Did all peers finish? The reduced sum will be 0.0 then
        if this_peer_finished_flag.item() == 0.0:
            return False
    elif this_peer_finished:
        return False
    return True


def get_initial_cache_position(model_params: SamplingParams) -> SamplingParams:
    """Initialize the cache position for model inputs based on past key values and input length.

    Args:
    ----
        model_params (SamplingParams): Model parameters which include 'use_cache', 'past_key_values', 'input_ids',
                                       and 'inputs_embeds'.
        use_cache (bool): Flag indicating whether caching is enabled for the model.

    Returns:
    -------
        SamplingParams: Updated model_params with 'cache_position' set appropriately.

    Raises:
    ------
        ValueError: If 'input_ids' are required but not provided in model_params when 'inputs_embeds' are absent.

    """
    if not model_params.use_cache:
        model_params.cache_position = None
        return model_params

    # Ensure input_ids or inputs_embeds are provided
    if model_params.tokens_ids is None:
        msg = "Either 'tokens_ids' or 'inputs_embeds' must be provided in model_params."
        raise ValueError(msg)

    tokens_ids: Tensor = model_params.tokens_ids

    # Calculate past length from kv_cache
    past_length = 0
    if (
        model_params.kv_cache is not None
        and isinstance(model_params.kv_cache, tuple)
        and isinstance(model_params.kv_cache[0], tuple)
    ):
        past_length = model_params.kv_cache[0][0].shape[2]

    # Determine the current length from inputs_embeds or input_ids
    cur_len: int = model_params.tokens_ids.shape[1]

    # Set the range for cache position based on calculated lengths
    model_params.cache_position = arange(
        start=past_length,
        end=past_length + cur_len,
        device=tokens_ids.device if tokens_ids is not None else "cpu",
    )
    return model_params


def update_model_params_for_generation(
    kv_cache: tuple[tuple[FloatTensor]] | None,
    model_params: SamplingParams,
) -> SamplingParams:
    """Update model parameters for the next generation cycle based on key-value caches and input length.

    Args:
    ----
        kv_cache (tuple[tuple[FloatTensor]] | None): Cached past key values for sequence continuation.
        model_params (SamplingParams): Current generation parameters to be updated.

    Returns:
    -------
        SamplingParams: Updated model parameters for the next generation cycle.

    """
    # Safely update past key values if they exist
    if kv_cache is not None:
        model_params.kv_cache = kv_cache

    # Extend token_type_ids if present
    if model_params.token_type_ids is not None:
        last_token_type_ids: Tensor = model_params.token_type_ids[:, -1].unsqueeze(dim=-1)
        model_params.token_type_ids = cat(tensors=[model_params.token_type_ids, last_token_type_ids], dim=-1)

    # Update attention masks
    key_to_update = "attention_mask"
    attention_mask = getattr(model_params, key_to_update)
    if attention_mask is not None:
        last_mask: Tensor = attention_mask.new_ones(size=(attention_mask.shape[0], 1))
        attention_mask: Tensor = cat(tensors=[attention_mask, last_mask], dim=-1)
        setattr(model_params, key_to_update, attention_mask)

    # Update cache position for models using cache mechanism
    if model_params.use_cache and model_params.cache_position is not None:
        cache_position: Tensor = model_params.cache_position
        model_params.cache_position = cache_position[-1:] + 1

    # Remove input_ids to prevent duplication in next generation step
    model_params.tokens_ids = None

    return model_params
