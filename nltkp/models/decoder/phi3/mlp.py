"""Multi-layer Perceptron (MLP) for the Phi3 model.

This module defines the MLP used in the Phi3 model, including the gating mechanism
and activation function.
"""

from torch import FloatTensor, Tensor, nn
from transformers.models.phi3.configuration_phi3 import Phi3Config


class Phi3MLP(nn.Module):
    """Multi-layer Perceptron (MLP) for the Phi3 model."""

    def __init__(self, config: Phi3Config) -> None:
        """Initialize the Phi3MLP module.

        Args:
        ----
            config (Phi3Config): Configuration object containing model hyperparameters.

        """
        super().__init__()

        self.config: Phi3Config = config
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)

        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: FloatTensor) -> FloatTensor:
        """Perform the forward pass of the MLP.

        Args:
        ----
            hidden_states (FloatTensor): The input hidden states.

        Returns:
        -------
            FloatTensor: The output hidden states after applying the MLP.

        """
        gate: Tensor
        up_states: Tensor = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(chunks=2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)
