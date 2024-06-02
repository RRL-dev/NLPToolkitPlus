"""Conv1d for GPT2 model."""

from __future__ import annotations

from torch import Tensor, addmm, empty, nn, zeros


class Conv1D(nn.Module):
    """1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
    ----
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.

    """

    def __init__(self: Conv1D, nf: int, nx: int) -> None:
        """Initialize the Conv1D layer.

        Args:
        ----
            nf (int): The number of output features.
            nx (int): The number of input features.

        """
        super().__init__()
        self.nf: int = nf
        self.weight = nn.Parameter(data=empty(nx, nf))
        self.bias = nn.Parameter(data=zeros(size=(nf,)))
        nn.init.normal_(tensor=self.weight, std=0.02)

    def forward(self: Conv1D, x: Tensor) -> Tensor:
        """Perform the forward pass of the Conv1D layer.

        Args:
        ----
            x (Tensor): The input tensor.

        Returns:
        -------
            Tensor: The output tensor.

        """
        size_out: tuple[int, ...] = x.size()[:-1] + (self.nf,)
        x = addmm(input=self.bias, mat1=x.view(-1, x.size(dim=-1)), mat2=self.weight)
        return x.view(size=size_out)
