"""The module provides the GPT2Attention class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .conv import Conv1D

if TYPE_CHECKING:
    from .config import Config


class GPT2Attention(nn.Module):
    """GPT-2 Attention layer as defined by OpenAI.

    This layer implements the attention mechanism used by the GPT-2 model. It takes a configuration object
    to set up layer-specific parameters and options that control the behavior of the attention mechanism.

    Args:
    ----
        config (Config): The configuration object containing settings like number of attention heads,
                         dimensionality sizes, dropout rates, and other model-specific parameters.
        layer_idx (int, optional): The index of the layer within the overall model architecture. This can
                                   be used for layer-specific behaviors or initializations.

    """  # noqa: E501

    def __init__(self: GPT2Attention, config: Config, layer_idx: int | None = None) -> None:
        """Initialize the GPT2Attention layer.

        This constructor initializes the attention layer by setting up weights, biases, and other necessary
        components based on the provided configuration. It also prepares buffers and dropout layers that
        are essential for the attention operations within GPT-2.

        Args:
        ----
            config (Config): Configuration object providing necessary parameters like hidden size,
                             number of attention heads, and dropout probabilities.
            layer_idx (int | None, optional): Optional index indicating the position of this layer within
                                              the model. Defaults to None, if not specified.

        """  # noqa: E501
        super().__init__()
        self.config: Config = config
        max_positions: int = config.max_position_embeddings
        self.register_buffer(
            name="bias",
            tensor=torch.tril(
                input=torch.ones(size=(max_positions, max_positions), dtype=torch.bool),
            ).view(
                1,
                1,
                max_positions,
                max_positions,
            ),
            persistent=False,
        )

        self.embed_dim: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.embed_dim // self.num_heads
        self.split_size: int = self.embed_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            msg: str = f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."  # noqa: E501
            raise ValueError(
                msg,
            )

        self.layer_idx: int | None = layer_idx
        self.scale_attn_weights: bool = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx: bool = config.scale_attn_by_inverse_layer_idx

        self.c_attn = Conv1D(nf=3 * self.embed_dim, nx=self.embed_dim)
        self.c_proj = Conv1D(nf=self.embed_dim, nx=self.embed_dim)
        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)
        self.is_causal = True

    def _split_heads(
        self: GPT2Attention,
        tensor: Tensor,
        num_heads: int,
        attn_head_size: int,
    ) -> Tensor:
        """Split hidden_size dim into attn_head_size and num_heads.

        Args:
        ----
            tensor (Tensor): The input tensor.
            num_heads (int): The number of attention heads.
            attn_head_size (int): The size of each attention head.

        Returns:
        -------
            Tensor: The reshaped tensor.

        """
        new_shape: tuple[int, ...] = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(size=new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(
        self: GPT2Attention,
        tensor: Tensor,
        num_heads: int,
        attn_head_size: int,
    ) -> Tensor:
        """Merge attn_head_size dim and num_attn_heads dim into hidden_size.

        Args:
        ----
            tensor (Tensor): The input tensor.
            num_heads (int): The number of attention heads.
            attn_head_size (int): The size of each attention head.

        Returns:
        -------
            Tensor: The reshaped tensor.

        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape: tuple[int, ...] = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(size=new_shape)

    def attn(
        self: GPT2Attention,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask_details: dict[str, Tensor | None],
    ) -> Tensor:
        """Calculate the attention scores and apply the attention and head masks.

        Args:
        ----
            query (Tensor): The query tensor of shape (batch_size, num_heads, seq_length, head_dim).
            key (Tensor): The key tensor, transposed to shape (batch_size, num_heads, head_dim, seq_length).
            value (Tensor): The value tensor of shape (batch_size, num_heads, seq_length, head_dim).
            mask_details (Dict[str, Optional[Tensor]]): Dictionary containing optional tensors:
                'attention_mask' (Tensor | None): Mask to avoid attention to unwanted positions, typically future positions.
                'head_mask' (Tensor | None): Mask to nullify attention at specific heads.

        Returns:
        -------
            Tensor: A Tensor containing:
                attn_output (Tensor): The output from attention application to value tensor.

        The attention scores are scaled, masked, and then applied to the value tensor. Masks are applied
        according to the specifications in `mask_details`. This method supports mixed precision by casting
        tensors to the necessary data types.

        """  # noqa: E501
        attn_weights: Tensor = torch.matmul(input=query, other=key.transpose(dim0=-1, dim1=-2))
        if self.scale_attn_weights:
            attn_weights /= torch.sqrt(
                input=torch.tensor(data=value.size(dim=-1), device=attn_weights.device),
            )

        query_length, key_length = query.size(dim=-2), key.size(dim=-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min

        mask_value = torch.full(
            size=[],
            fill_value=mask_value,  # type: ignore  # noqa: PGH003
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )  # type: ignore  # noqa: PGH003

        attn_weights = torch.where(
            condition=causal_mask,
            input=attn_weights.to(dtype=attn_weights.dtype),
            other=mask_value,
        )

        if mask_details.get("attention_mask") is not None:
            attn_weights += mask_details["attention_mask"]

        attn_weights = torch.nn.functional.softmax(input=attn_weights, dim=-1)

        if mask_details.get("head_mask") is not None:
            attn_weights *= mask_details["head_mask"]

        attn_weights = self.attn_dropout(attn_weights.type(dtype=value.dtype))
        attn_output: Tensor = torch.matmul(input=attn_weights, other=value)
        return attn_output

    def forward(
        self: GPT2Attention,
        hidden_states: Tensor,
        layer_past: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Perform the forward pass of the GPT2Attention layer.

        Args:
        ----
            hidden_states (Tensor): Input hidden states to the attention layer.
            layer_past (Optional[Tuple[Tensor, Tensor]], optional): Past key and value tensors for efficient decoding.
            attention_mask (Optional[Tensor], optional): Mask to prevent attention to future tokens.
            use_cache (bool, optional): Returns current key and value tensors if True.

        Returns:
        -------
            Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]: Output tensor and optionally the current key and value tensors.

        """  # noqa: E501
        key: Tensor
        value: Tensor
        query: Tensor

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query, key, value = (
            self._split_heads(tensor=t, num_heads=self.num_heads, attn_head_size=self.head_dim)
            for t in (query, key, value)
        )

        if layer_past is not None:
            key, value = (
                torch.cat(tensors=[past, t], dim=-2)
                for past, t in zip(layer_past, (key, value), strict=False)
            )

        present: tuple[Tensor, Tensor] | None = (key, value) if use_cache else None

        attn_output: Tensor = self.attn(
            query=query,
            key=key,
            value=value,
            mask_details={"attention_mask": attention_mask, "head_mask": None},
        )

        attn_output = self._merge_heads(
            tensor=attn_output,
            num_heads=self.num_heads,
            attn_head_size=self.head_dim,
        )

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present
