"""Module for implements the GPT2Model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from .block import GPT2Block
from .utils import (
    CacheModelOutputs,
    generate_next_token_and_update_inputs,
    get_initial_cache_position,
    initialize_sampling,
)

if TYPE_CHECKING:
    from .config import Config


@dataclass
class ForwardParams:
    """Class for holding all the parameters required for the forward pass of the GPT2Model.

    This class encapsulates all input parameters, allowing for a cleaner function signature in the
    model's forward method. It handles default values and provides a structured way to manage
    the forward pass inputs.

    Attributes
    ----------
        use_cache (bool | None): Flag to determine if the current layer's past key and values should be returned,
            enabling incremental decoding.
        input_ids (Tensor | None): Indices of input sequence tokens in the vocabulary.
        return_dict (bool | None): Whether to return a dictionary of all important outputs or only the final tensor.
        position_ids (Tensor | None): Position indices corresponding to sequence tokens for positional encoding.
        inputs_embeds (Tensor | None): Precomputed token embeddings, bypassing the embedding lookup.
        attention_mask (Tensor | None): Mask to avoid attention on certain positions (usually padding).
        past_key_values (tuple[tuple[Tensor, ...], ...] | None): Tuples of past key and value tensors for each layer,
            used for incremental decoding.

    """  # noqa: E501

    use_cache: bool | None = None
    input_ids: Tensor | None = None
    return_dict: bool | None = None
    position_ids: Tensor | None = None
    inputs_embeds: Tensor | None = None
    attention_mask: Tensor | None = None
    past_key_values: tuple[tuple[Tensor, ...], ...] | None = None


class GPT2Model(nn.Module):
    """GPT-2 model as described by OpenAI which uses the transformer mechanism.

    The model comprises embedding layers for input tokens and positions, several transformer blocks (GPT2Block),
    and layer normalization at the end.

    Args:
    ----
        config (Config): A configuration object containing all settings for the model.

    """  # noqa: E501

    def __init__(self: GPT2Model, config: Config) -> None:
        """Initialize the GPT2 model with embedding layers and transformer blocks."""
        super().__init__()
        self.config: Config = config
        self.embed_dim: int = config.hidden_size

        self.wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=self.embed_dim)
        self.wpe = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=self.embed_dim,
        )

        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.h = nn.ModuleList(
            modules=[
                GPT2Block(config=config, layer_idx=i) for i in range(config.num_hidden_layers)
            ],
        )

        self.ln_f = nn.LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self: GPT2Model, params: ForwardParams) -> CacheModelOutputs:
        """Forward pass through the GPT-2 model to generate predictions."""
        use_cache: bool = (
            params.use_cache if params.use_cache is not None else self.config.use_cache
        )

        if params.input_ids is not None and params.inputs_embeds is not None:
            msg = "You cannot specify both input_ids and inputs_embeds at the same time"
            raise ValueError(msg)

        if params.input_ids is not None:
            input_shape: torch.Size = params.input_ids.size()
            params.input_ids = params.input_ids.view(-1, input_shape[-1])
            batch_size = params.input_ids.shape[0]
        elif params.inputs_embeds is not None:
            input_shape = params.inputs_embeds.size()[:-1]
            batch_size = params.inputs_embeds.shape[0]
        else:
            msg = "You have to specify either input_ids or inputs_embeds"
            raise ValueError(msg)

        device: torch.device = (
            params.input_ids.device if params.input_ids is not None else params.inputs_embeds.device  # type: ignore  # noqa: PGH003
        )

        if params.past_key_values is None:
            params.past_key_values = tuple([None] * len(self.h))  # type: ignore  # noqa: PGH003

        if params.position_ids is None:
            params.position_ids = torch.arange(
                start=0,
                end=input_shape[-1],
                dtype=torch.long,
                device=device,
            ).unsqueeze(dim=0)

        if params.attention_mask is not None:
            params.attention_mask = params.attention_mask.view(batch_size, -1)
            params.attention_mask = params.attention_mask[:, None, None, :]
            # Ensure attention_mask is floating point for calculation involving torch.finfo
            params.attention_mask = params.attention_mask.to(dtype=torch.float32)
            attention_mask_finfo = torch.finfo(params.attention_mask.dtype)
            params.attention_mask = (1.0 - params.attention_mask) * attention_mask_finfo.min

        if params.inputs_embeds is None:
            params.inputs_embeds = self.wte(params.input_ids)

        position_embeds: Tensor = self.wpe(params.position_ids)
        hidden_states: Tensor = params.inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(dim=-1),)

        presents: tuple[()] | None = () if use_cache else None
        for block, layer_past in zip(self.h, params.past_key_values, strict=False): # type: ignore  # noqa: PGH003
            outputs: tuple[Tensor, ...] | Tensor = block(
                hidden_states,
                layer_past,
                params.attention_mask,
                use_cache,
            )

            hidden_states = outputs[0]
            if use_cache and isinstance(presents, tuple):
                presents = presents + (outputs[1],)  # type: ignore # noqa: RUF005, PGH003

        return CacheModelOutputs(
            logits=None,
            past_key_values=presents,
            last_hidden_state=self.ln_f(hidden_states).view(output_shape),
        )


class GPT2LMHeadModel(GPT2Model):
    """Extends the GPT2Model with a linear layer for language modeling tasks.

    This model appends a language modeling head to the GPT-2 model, which can be used
    to generate predictions directly from the output embeddings of the GPT-2 model.

    Args:
    ----
        config (Config): The configuration information containing model dimensions and settings.

    Attributes:
    ----------
        lm_head (nn.Linear): The linear layer that maps hidden states to vocabulary size predictions.

    """  # noqa: E501

    def __init__(self: GPT2LMHeadModel, config: Config) -> None:
        """Initialize the GPT2 model with a language modeling head."""
        super().__init__(config=config)
        self.lm_head = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.wte.weight

    def prepare_inputs_for_generation(
        self: GPT2LMHeadModel,
        input_ids: Tensor,
        past_key_values: tuple[tuple[Tensor, ...], ...] | None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs: Tensor | bool,
    ) -> dict[str, Tensor | tuple[tuple[Tensor, ...], ...] | None | bool]:
        """Prepare the inputs required for a generation step in the model.

        Args:
        ----
            input_ids (Tensor): Input IDs.
            past_key_values (Optional[tuple[tuple[Tensor, ...], ...]]): Past key and value tensors.
            inputs_embeds (Optional[Tensor]): Precomputed token embeddings.
            **kwargs: Additional keyword arguments containing settings like attention_mask and token_type_ids.

        Returns:
        -------
            dict[str, Tensor | tuple[tuple[Tensor, ...], ...] | None | bool]: Prepared inputs for the model generation.

        """  # noqa: E501
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            remove_prefix_length = (
                past_length if input_ids.shape[1] > past_length else input_ids.shape[1] - 1
            )
            input_ids = input_ids[:, remove_prefix_length:]

        attention_mask: Tensor | None = kwargs.get("attention_mask", None)  # type: ignore  # noqa: PGH003
        position_ids = None
        if isinstance(attention_mask, Tensor):
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(mask=attention_mask == 0, value=1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs: dict[str, Tensor | tuple[tuple[Tensor, ...], ...] | None | bool] = {
            "input_ids": input_ids,
        }

        if inputs_embeds is not None and not past_key_values:
            model_inputs["inputs_embeds"] = inputs_embeds

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", self.config.use_cache)
                if isinstance(kwargs.get("use_cache"), bool)
                else self.config.use_cache,
                "position_ids": position_ids,
                "attention_mask": attention_mask if isinstance(attention_mask, Tensor) else None,
            },
        )

        return model_inputs

    def forward(self: GPT2LMHeadModel, params: ForwardParams) -> CacheModelOutputs:
        """Forward pass through the GPT2 model and language modeling head."""
        cache_output: CacheModelOutputs = super().forward(params=params)
        logits: Tensor = self.lm_head(cache_output.last_hidden_state)
        return CacheModelOutputs(
            logits=logits,
            past_key_values=cache_output.past_key_values,
            last_hidden_state=cache_output.last_hidden_state,
        )

    def sample(
        self: GPT2LMHeadModel,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Generate a sequence of tokens given input IDs and an optional attention mask.

        Args:
        ----
            input_ids (Tensor): Input IDs.
            attention_mask (Tensor, optional): Attention mask. Defaults to None.

        Returns:
        -------
            Tensor: Generated sequence of token IDs.

        """
        model_kwargs: dict[str, Any] = {}

        if attention_mask is None:
            default_attention_mask: Tensor = torch.ones(
                size=input_ids.shape[:2],
                dtype=torch.long,
                device=input_ids.device,
            )
            model_kwargs["attention_mask"] = default_attention_mask
        else:
            model_kwargs["attention_mask"] = attention_mask

        model_kwargs["use_cache"] = self.config.use_cache

        model_kwargs = get_initial_cache_position(
            input_ids=input_ids,
            model_kwargs=model_kwargs,
        )

        pad_token_id: Tensor
        finish_samples: bool
        unfinished_sequences: Tensor

        pad_token_id, finish_samples, unfinished_sequences = initialize_sampling(
            input_ids=input_ids,
            vocab_size=self.config.vocab_size,
        )

        while not finish_samples:
            model_inputs: dict[str, Tensor | tuple[tuple[Tensor, ...], ...] | bool | None] = (
                self.prepare_inputs_for_generation(
                    input_ids=input_ids,
                    **model_kwargs,
                )
            )

            forward_params = ForwardParams(
                input_ids=model_inputs["input_ids"]
                if isinstance(model_inputs["input_ids"], Tensor)
                else None,
                past_key_values=model_inputs["past_key_values"]
                if isinstance(model_inputs["past_key_values"], tuple)
                else None,
                attention_mask=model_inputs["attention_mask"]
                if isinstance(model_inputs["attention_mask"], Tensor)
                else None,
                position_ids=model_inputs["position_ids"]
                if isinstance(model_inputs["position_ids"], Tensor)
                else None,
                use_cache=model_inputs["use_cache"]
                if isinstance(model_inputs["use_cache"], bool)
                else None,
            )

            cache_output: CacheModelOutputs = self.forward(params=forward_params)

            if cache_output.logits is None:
                msg = "Logits from the forward pass are None."
                raise ValueError(msg)

            input_ids, model_kwargs = generate_next_token_and_update_inputs(
                cache_output=cache_output,
                input_ids=input_ids,
                unfinished_sequences=unfinished_sequences,
                pad_token_id=pad_token_id,
                model_kwargs=model_kwargs,
            )

            stopping_criteria: bool = (
                input_ids.shape[1] > self.config.max_token_length
                or torch.tensor(data=[self.config.vocab_size - 1], device=input_ids.device)
                in input_ids
            )

            unfinished_sequences = unfinished_sequences & (~stopping_criteria)
            finish_samples = unfinished_sequences.max().item() == 0

        return input_ids
