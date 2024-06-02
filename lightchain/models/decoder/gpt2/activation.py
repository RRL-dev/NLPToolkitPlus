"""GPT2 activations."""

from __future__ import annotations

import math

import torch


class NewGELUActivation(torch.nn.Module):
    """Implementation of the GELU activation function.

    The GELU (Gaussian Error Linear Unit) activation function is used in several prominent NLP models,
    including Google's BERT and OpenAI's GPT. This function is described in the Gaussian Error Linear Units
    paper (https://arxiv.org/abs/1606.08415).

    This particular implementation follows the exact formulation used in the Google BERT repository.
    """

    def forward(self: NewGELUActivation, input: torch.Tensor) -> torch.Tensor:
        """Apply the GELU activation function to the input Tensor.

        Args:
        ----
            input (torch.Tensor): The input tensor to which the GELU activation function will be applied.

        Returns:
        -------
            torch.Tensor: The resulting tensor after applying the GELU activation function.

        """
        cdf: torch.Tensor = 0.5 * (
            1.0
            + torch.tanh(
                input=math.sqrt(2 / math.pi)
                * (input + 0.044715 * torch.pow(input=input, exponent=3)),
            )
        )
        return input * cdf
