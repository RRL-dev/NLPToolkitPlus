"""Attention mechanism for the Phi3 model.

This module defines the multi-headed attention mechanism used in the Phi3 model,
including the key-value cache management and rotary positional embedding.
"""

from logging import Logger
from math import sqrt

from torch import LongTensor, Tensor, float32, matmul, nn
from transformers import logging
from transformers.cache_utils import Cache
from transformers.models.phi3.configuration_phi3 import Phi3Config

from nltkp.models.utils import create_general_causal_mask

from .embedding import Phi3RotaryEmbedding, apply_rotary_pos_emb

logger: Logger = logging.get_logger(name=__name__)


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat the key-value states to match the number of attention heads."""
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeats=n_rep, dim=1)


class Phi3Attention(nn.Module):
    """Multi-headed attention mechanism for the Phi3 model."""

    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        """Initialize the Phi3Attention module.

        Args:
        ----
            config (Phi3Config): Configuration object containing model hyperparameters.
            layer_idx (int): Index of the layer.

        """
        super().__init__()
        self.config: Phi3Config = config
        self.head_dim: int = config.hidden_size // config.num_attention_heads
        self.layer_idx: int = layer_idx
        self.num_heads: int = config.num_attention_heads
        self.num_key_value_heads: int = config.num_key_value_heads
        self.num_key_value_groups: int = self.num_heads // self.num_key_value_heads
        self._init_rope()

        if (self.head_dim * self.num_heads) != config.hidden_size:
            msg = "hidden_size must be divisible by num_heads."
            raise ValueError(msg)

        op_size: int = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(in_features=self.num_heads * self.head_dim, out_features=config.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(in_features=config.hidden_size, out_features=op_size, bias=False)
        self.attention_dropout: float = config.attention_dropout

    def _init_rope(self) -> None:
        self.rotary_emb = Phi3RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,  # type: ignore  # noqa: PGH003
        )

    def forward(
        self,
        position_ids: LongTensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        kv_cache: Cache | None = None,
    ) -> tuple[Tensor, Cache | None]:
        """Perform the forward pass of the attention mechanism.

        Args:
        ----
            hidden_states (Tensor): The input hidden states.
            kv_cache (Cache | None): The cache for past key-value states. Defaults to None.
            attention_mask (Tensor | None): The attention mask. Defaults to None.
            position_ids (LongTensor | None): The position IDs. Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.

        Returns:
        -------
            tuple[Tensor, Tensor | None, Cache | None]: The attention output,
            attention weights (if output_attentions is True), and updated past_key_value.

        """
        logger.warning(msg="You are not running the flash-attention implementation, expect numerical differences.")

        batch_size: int
        query_size: int

        batch_size, query_size, _ = hidden_states.size()

        key_states: Tensor
        query_states: Tensor
        value_states: Tensor

        query_states, key_states, value_states = self._project_qkv(hidden_states=hidden_states)
        query_states, key_states, value_states = self._reshape_qkv(
            query_size=query_size,
            batch_size=batch_size,
            key_states=key_states,
            query_states=query_states,
            value_states=value_states,
        )

        cos: LongTensor
        sin: LongTensor

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query=query_states, key=key_states, cos=cos, sin=sin)

        if kv_cache is not None and isinstance(kv_cache, Cache):
            key_states, value_states = kv_cache.update(
                layer_idx=self.layer_idx,
                key_states=key_states,
                value_states=value_states,
                cache_kwargs={"sin": sin, "cos": cos},
            )

        key_states = repeat_kv(hidden_states=key_states, n_rep=self.num_key_value_groups)
        value_states = repeat_kv(hidden_states=value_states, n_rep=self.num_key_value_groups)

        attn_output: Tensor

        attn_output = self._compute_attention(
            key_states=key_states,
            value_states=value_states,
            query_states=query_states,
            attention_mask=attention_mask,
        )

        attn_output = (
            attn_output.transpose(dim0=1, dim1=2)
            .contiguous()
            .view(batch_size, query_size, self.num_heads * self.head_dim)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, kv_cache

    def _project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project hidden states to query, key, and value states.

        Args:
        ----
            hidden_states (Tensor): The input hidden states.

        Returns:
        -------
            Tuple[Tensor, Tensor, Tensor]: The query, key, and value states.

        """
        qkv: Tensor = self.qkv_proj(hidden_states)
        query_pos: int = self.num_heads * self.head_dim
        key_states: Tensor = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states: Tensor = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
        query_states: Tensor = qkv[..., :query_pos]
        return query_states, key_states, value_states

    def _reshape_qkv(  # noqa: PLR0913
        self,
        query_size: int,
        batch_size: int,
        key_states: Tensor,
        query_states: Tensor,
        value_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Reshape query, key, and value states.

        Args:
        ----
            query_states (Tensor): The query states.
            key_states (Tensor): The key states.
            value_states (Tensor): The value states.
            batch_size (int): The batch size.
            query_size (int): The query size.

        Returns:
        -------
            Tuple[Tensor, Tensor, Tensor]: The reshaped query, key, and value states.

        """
        query_states = query_states.view(batch_size, query_size, self.num_heads, self.head_dim).transpose(
            dim0=1,
            dim1=2,
        )
        key_states = key_states.view(batch_size, query_size, self.num_key_value_heads, self.head_dim).transpose(
            dim0=1,
            dim1=2,
        )
        value_states = value_states.view(batch_size, query_size, self.num_key_value_heads, self.head_dim).transpose(
            dim0=1,
            dim1=2,
        )
        return query_states, key_states, value_states

    def _compute_attention(
        self,
        key_states: Tensor,
        value_states: Tensor,
        query_states: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Compute attention weights and output.

        Args:
        ----
            query_states (Tensor): The query states.
            key_states (Tensor): The key states.
            value_states (Tensor): The value states.
            attention_mask (Optional[Tensor]): The attention mask.
            kv_cache (Cache | None): The cache for past key-value states. Defaults to None.

        Returns:
        -------
            Tensor: The attention output.

        """
        attn_weights: Tensor = matmul(input=query_states, other=key_states.transpose(dim0=2, dim1=3)) / sqrt(
            self.head_dim,
        )  # attn_weights: [batch_size, num_heads, seq_len, total_seq_len]

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(input=attn_weights, dim=-1, dtype=float32).to(dtype=value_states.dtype)
        attn_weights = nn.functional.dropout(input=attn_weights, p=self.attention_dropout, training=self.training)
        attn_output: Tensor = matmul(input=attn_weights, other=value_states)
        return attn_output
