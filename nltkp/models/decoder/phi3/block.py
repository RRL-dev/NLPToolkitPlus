"""Root Mean Square (RMS) Normalization for the Phi3 model.

This module defines the RMS normalization used in the Phi3 model, which is equivalent to the T5LayerNorm.
"""

from typing import Any

import torch
from torch import LongTensor, Tensor, mean, nn, ones, rsqrt
from transformers.cache_utils import DynamicCache

from .attention import Phi3Attention
from .mlp import Phi3Config, Phi3MLP


class Phi3RMSNorm(nn.Module):
    """Root Mean Square (RMS) Normalization for the Phi3 model."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize the Phi3RMSNorm module.

        Args:
        ----
            hidden_size (int): The size of the hidden layer.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

        """
        super().__init__()
        self.weight = nn.Parameter(data=ones(hidden_size))
        self.variance_epsilon: float = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Perform the forward pass of the RMS normalization.

        Args:
        ----
            hidden_states (Tensor): The input hidden states.

        Returns:
        -------
            Tensor: The normalized hidden states.

        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype=torch.float32)
        variance: Tensor = mean(input=hidden_states.pow(exponent=2), dim=-1, keepdim=True)
        hidden_states *= rsqrt(input=variance + self.variance_epsilon)
        return self.weight * hidden_states.to(dtype=input_dtype)


class Phi3DecoderLayer(nn.Module):
    """Decoder layer for the Phi3 model."""

    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        """Initialize the Phi3DecoderLayer module.

        Args:
        ----
            config (Phi3Config): Configuration object containing model hyperparameters.
            layer_idx (int): Index of the layer.

        """
        super().__init__()

        self.config: Phi3Config = config
        self.self_attn = Phi3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Phi3MLP(config=config)
        self.input_layernorm = Phi3RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(p=config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(p=config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        kv_cache: DynamicCache | None = None,
        position_ids: LongTensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[Tensor, DynamicCache | None] | tuple[Any, None]:
        """Perform the forward pass of the decoder layer.

        Args:
        ----
            hidden_states (Tensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            kv_cache (DynamicCache | None): Cached past key and value projection states.
            position_ids (LongTensor | None): Indices of positions of each input sequence tokens.
            attention_mask (Tensor | None): Attention mask of size `(batch, 1, tgt_len, src_len)`.
            use_cache (bool, optional): If set to `True`, `past_key_values` key value states are returned.

        Returns:
        -------
            Tuple[FloatTensor, Optional[Tuple[FloatTensor, FloatTensor]]]: The hidden states, attention weights, and
            past key values if `use_cache` is True.

        """
        residual: Tensor = hidden_states
        normed_hidden_states: Tensor = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs: Tensor
        present_key_value: DynamicCache | None

        attn_outputs, present_key_value = self.self_attn(
            kv_cache=kv_cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            hidden_states=normed_hidden_states,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(normed_hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        if use_cache:
            outputs: tuple[Tensor, DynamicCache | None] = (hidden_states, present_key_value)
            return outputs

        return hidden_states, None
