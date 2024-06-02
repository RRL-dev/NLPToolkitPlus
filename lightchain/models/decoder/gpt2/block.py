"""The module provides the implementation of a GPT-2 Transformer Block."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from .attention import GPT2Attention
from .mlp import GPT2MLP

if TYPE_CHECKING:
    from .config import Config


class GPT2Block(nn.Module):
    """Implementation of the GPT-2 Transformer Block.

    The block includes layer normalization, an attention mechanism, and a two-layer feed-forward network
    with a GELU activation in between. It supports operations such as caching for efficient sequence generation.

    Args:
    ----
        config (Config): Configuration object with model dimensions and settings.
        layer_idx (int | None): Index of this layer within the model; used for layer-specific behaviors.

    """  # noqa: E501

    def __init__(self: GPT2Block, config: Config, layer_idx: int | None = None) -> None:
        """Initialize the GPT-2 Transformer Block with necessary components.

        Args:
        ----
            config (Config): Configuration object containing necessary model parameters.
            layer_idx (int | None): Layer index for possible layer-specific configurations.

        """
        super().__init__()
        hidden_size: int = config.hidden_size
        inner_dim: int = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(normalized_shape=hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(normalized_shape=hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(intermediate_size=inner_dim, config=config)

    def forward(
        self: GPT2Block,
        hidden_states: Tensor | None,
        layer_past: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[Tensor, ...] | Tensor:
        """Perform the forward pass of the GPT-2 Transformer Block.

        Args:
        ----
            hidden_states (Tensor | None): Input hidden states to the layer.
            layer_past (tuple[Tensor, Tensor] | None): Optional tuple of past key and value tensors.
            attention_mask (Tensor | None): Optional tensor for attention masking.
            use_cache (bool): If True, cache the current key and value tensors.

        Returns:
        -------
            Tensor | tuple[Tensor, ...]: Output of the block, optionally including cached tensors.

        """
        if hidden_states is None:
            msg = "hidden_states cannot be None"
            raise ValueError(msg)

        residual: Tensor = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

        attn_output: Tensor = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs: Tensor = attn_outputs[1:]  # Extract additional outputs if they exist

        # Residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states: Tensor = self.mlp(hidden_states)

        # Another residual connection
        hidden_states = feed_forward_hidden_states + residual

        # Handling outputs based on whether caching is used
        if use_cache:
            return (hidden_states, *outputs)
        return hidden_states
