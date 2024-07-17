"""The module defines the BasePooling class and associated functions.

Token pooling strategies are used to aggregate or transform sets of token embeddings,
into a single representation,
which can be beneficial in various natural language processing tasks to reduce dimensionality,
and capture important features of the data.

The module provides several pooling methods such as:
- CLS token pooling: Uses the embedding of the CLS token as the representation.
- Maximum token pooling: Selects the maximum value over all tokens.
- Mean token pooling: Calculates the arithmetic mean of tokens.
- Mean square root token pooling: Calculates the square root of the mean of tokens.
- Weighted mean token pooling: Applies a weighted mean operation on tokens.

These pooling strategies can be configured dynamically when an instance of BasePooling is created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch._tensor import Tensor

from .func import pooling_funcs

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


class BasePooling:
    """A class for applying various token pooling operations based on configuration flags.

    Attributes:
    ----------
    pooling_functions (Sequence[Callable]): A list of pooling function callables.

    Methods:
    -------
    __init__(self, **pooling_modes: bool):
        Initializes the Pooling instance with pooling modes.

    apply_pooling(self, output_vectors: List[torch.Tensor], features: Dict[str, Any],
        token_embeddings: torch.Tensor) -> None:
        Applies the selected pooling functions to the given token embeddings.

    Args:
    ----
        **pooling_modes (bool): Arbitrary keyword arguments where keys are the mode.

    Example usage:
    --------------
        pooling_instance = Pooling(
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=True
            )
        output_vectors = []
        features = {'attention_mask': torch.tensor([...])}
        token_embeddings = torch.randn(10, 768)
        pooling_instance.apply_pooling(output_vectors, features, token_embeddings)

    """

    def __init__(self: BasePooling, **pooling_modes: bool) -> None:
        """Initialize the BasePooling instance with specified embedding dimension and pooling modes.

        Args:
        ----
            **pooling_modes (bool): Arbitrary keyword arguments where keys are the mode names
                and values are booleans indicating whether to activate the corresponding mode.

        """
        self.pooling_functions: list[Callable[[list[Tensor], dict[str, Any]], None]] = []
        self.pooling_modes: dict[str, bool] = pooling_modes

        # Append the correct functions based on configuration
        for mode, func in pooling_funcs.items():
            if pooling_modes.get(f"{mode}", False):
                self.pooling_functions.append(func)  # type: ignore  # noqa: PGH003

    def apply_pooling(
        self: BasePooling,
        output_vectors: list,
        features: dict[str, Any],
    ) -> list[Tensor]:
        """Apply the configured pooling functions to the given token embeddings.

        Args:
        ----
            output_vectors (list[torch.Tensor]): The list to which the results of pooling operations.
            features (dict[str, Any]): A dictionary containing relevant features such as the attention mask.

        Returns:
        -------
            List[Tensor]: The list containing all the pooled results.

        """
        for func in self.pooling_functions:
            func(output_vectors, features)
        return output_vectors

    def __repr__(self: BasePooling) -> str:
        """Represent the BasePooling instance showing active pooling modes and their status."""
        modes_status = ", ".join(f"{mode}={status}" for mode, status in self.pooling_modes.items())
        return f"{self.__class__.__name__}({modes_status})"
