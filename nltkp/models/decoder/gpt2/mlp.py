"""The module defines the GPT2MLP class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .activation import NewGELUActivation
from .conv import Conv1D

if TYPE_CHECKING:
    from .config import Config


class GPT2MLP(nn.Module):
    """Feed-forward MLP layer for GPT-2 model as used in Transformer architectures.

    This class encapsulates a two-layer feed-forward neural network with a GELU activation in between.
    It is typically used within the transformer blocks of models like GPT-2.

    Args:
    ----
        intermediate_size (int): The size of the intermediate layer (dimensionality of the first linear transformation).
        config (Config): Configuration object containing model dimensions and probabilities for dropout.

    """  # noqa: E501

    def __init__(self: GPT2MLP, intermediate_size: int, config: Config) -> None:
        """Initialize the GPT2MLP module with necessary layers and configurations.

        This constructor sets up two convolutional layers with a GELU activation in between. It is designed to
        project the input tensor from hidden size to intermediate size and back to hidden size, with dropout
        applied after the final projection.

        Args:
        ----
            intermediate_size (int): The size of the intermediate (feed-forward) layer.
            config (Config): A configuration object containing model-specific settings such as hidden_size and dropout probabilities.

        """  # noqa: E501
        super().__init__()
        embed_dim: int = config.hidden_size
        self.c_fc = Conv1D(nf=intermediate_size, nx=embed_dim)
        self.c_proj = Conv1D(nf=embed_dim, nx=intermediate_size)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def forward(self: GPT2MLP, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through the MLP block.

        Args:
        ----
            hidden_states (torch.FloatTensor): Input feature map from the previous layer or the self-attention module.

        Returns:
        -------
            torch.FloatTensor: Output feature map after processing through two linear projections separated by GELU activation and dropout.

        """  # noqa: E501
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)
