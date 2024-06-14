"""Pooling functionality."""

from collections.abc import Callable
from typing import Any

import torch


def get_token_embeddings(features: dict[str, Any]) -> torch.Tensor:
    """Retrieve 'token_embeddings' from the given features dictionary.

    Args:
    ----
        features (Dict[str, Any]): Dictionary containing features including 'token_embeddings'.

    Returns:
    -------
        torch.Tensor: The token embeddings tensor extracted from the features.

    Raises:
    ------
        ValueError: If 'token_embeddings' is not present in the features dictionary.

    """
    token_embeddings: torch.Tensor | None = features.get("token_embeddings")
    if token_embeddings is None:
        msg = "token_embeddings not provided in features"
        raise ValueError(msg)
    return token_embeddings


def cls_token_vector(
    output_vectors: list[torch.Tensor],
    features: dict[str, Any],
) -> list[torch.Tensor]:
    """Append the CLS token's embedding from the provided token embeddings to the output vectors.

    Args:
    ----
        output_vectors (List[torch.Tensor]): List where the CLS token vector will be appended.
        features (dict[str, Any]): Dictionary containing features. 'token_embeddings'.

    Returns:
    -------
        List[torch.Tensor]: The list with the appended CLS token vector.

    """
    token_embeddings: torch.Tensor = get_token_embeddings(features=features)

    cls_token: torch.Tensor = token_embeddings[:, 0]  # Take the first token by default
    output_vectors.append(cls_token)
    return output_vectors


def max_token_vector(
    output_vectors: list[torch.Tensor],
    features: dict[str, Any],
) -> list[torch.Tensor]:
    """Compute the maximum token vector for each sequence, masked by the attention mask.

    Args:
    ----
        output_vectors (list[torch.Tensor]): List where the maximum token vector will be appended.
        features (dict[str, Any]): Dictionary containing features including 'attention_mask'.

    Returns:
    -------
        list[torch.Tensor]: Updated list of output vectors with the maximum token vector appended.

    Note:
    ----
        The function masks out irrelevant tokens by setting them to a very large negative value before taking the maximum.

    """  # noqa: E501
    token_embeddings: torch.Tensor = get_token_embeddings(features=features)
    attention_mask: torch.Tensor = features["attention_mask"]

    input_mask_expanded: torch.Tensor = (
        attention_mask.unsqueeze(dim=-1)
        .expand(size=token_embeddings.size())
        .to(dtype=token_embeddings.dtype)
    )
    # Create a copy to avoid modifying the original embeddings if necessary
    modified_embeddings: torch.Tensor = token_embeddings.clone()
    modified_embeddings[
        input_mask_expanded == 0
    ] = -1e9  # Mask out the tokens by setting to a large negative value

    max_over_time: torch.Tensor = torch.max(input=modified_embeddings, dim=1)[0]
    output_vectors.append(max_over_time)
    return output_vectors


def sum_embeddings_mask(
    features: dict[str, Any],
    sum_embeddings: torch.Tensor,
    input_mask_expanded: torch.Tensor,
) -> torch.Tensor:
    """Compute a sum mask for embeddings, optionally using weights.

    Args:
    ----
        features (dict[str, Any]): Dictionary containing optional 'token_weights_sum' which
                                   modifies the sum based on per-token weights.
        sum_embeddings (torch.Tensor): The embeddings tensor for which the sum mask is calculated.
        input_mask_expanded (torch.Tensor): The expanded input mask tensor.

    Returns:
    -------
        torch.Tensor: The clamped sum mask tensor to ensure it does not include zero values, avoiding
                      division by zero in subsequent operations.

    """  # noqa: E501
    if "token_weights_sum" in features:
        weights_sum: torch.Tensor = features["token_weights_sum"]
        if weights_sum.dim() == 1:
            weights_sum = weights_sum.unsqueeze(dim=-1)  # Ensure it can be expanded properly
        sum_mask: torch.Tensor = weights_sum.expand_as(other=sum_embeddings)
    else:
        sum_mask = input_mask_expanded.sum(dim=1)

    return torch.clamp(input=sum_mask, min=1e-9)


def mean_token_vector(
    func: str,
    output_vectors: list[torch.Tensor],
    features: dict[str, Any],
) -> list[torch.Tensor]:
    """Calculate either the arithmetic mean or the square-root mean of token embeddings.

    The function uses caching to improve performance by avoiding redundant calculations.
    This is particularly useful in scenarios where multiple calls are made.

    Args:
    ----
        func (str): Specifies the operation to perform; must be 'mean' or 'mean_sqrt'.
        output_vectors (list[torch.Tensor]): List to which the result will be appended.
        features (dict[str, Any]): Dictionary containing features including the 'attention_mask'.

    Returns:
    -------
        list[torch.Tensor]: Updated list of output vectors with the new computed mean vector appended.

    Raises:
    ------
        ValueError: If the specified function is not recognized.

    """  # noqa: E501
    if func not in ["mean", "mean_sqrt"]:
        msg: str = f"Function not in ['mean', 'mean_sqrt'], got {func}"
        raise ValueError(msg)

    token_embeddings: torch.Tensor = get_token_embeddings(features=features)
    attention_mask: torch.Tensor = features["attention_mask"]

    input_mask_expanded: torch.Tensor = (
        attention_mask.unsqueeze(dim=-1)
        .expand_as(other=token_embeddings)
        .to(dtype=token_embeddings.dtype)
    )

    sum_mask: torch.Tensor
    sum_embeddings: torch.Tensor

    sum_embeddings = torch.sum(
        input=token_embeddings * input_mask_expanded,
        dim=1,
    )

    sum_mask = sum_embeddings_mask(
        features=features,
        sum_embeddings=sum_embeddings,
        input_mask_expanded=input_mask_expanded,
    )

    if func == "mean":
        output_vectors.append(sum_embeddings / sum_mask)
    if func == "mean_sqrt":
        safe_denominator: torch.Tensor = torch.sqrt(input=sum_mask + 1e-9)
        output_vectors.append(sum_embeddings / safe_denominator)
    return output_vectors


def weighted_mean_token_vector(
    output_vectors: list[torch.Tensor],
    features: dict[str, Any],
) -> list[torch.Tensor]:
    """Calculate the weighted mean of token vectors using attention masks and linearly increasing weights.

    Args:
    ----
        output_vectors (List[torch.Tensor]): List where the weighted mean token vector will be appended.
        features (dict[str, Any]): Dictionary containing features, including 'attention_mask'.

    Returns:
    -------
        List[torch.Tensor]: The list with the appended weighted mean token vector.

    """  # noqa: E501
    token_embeddings: torch.Tensor = get_token_embeddings(features=features)
    attention_mask: torch.Tensor = features["attention_mask"]

    input_mask_expanded: torch.Tensor = (
        attention_mask.unsqueeze(dim=-1)
        .expand(size=token_embeddings.size())
        .to(dtype=token_embeddings.dtype)
    )
    # token_embeddings shape: bs, seq, hidden_dim
    weights: torch.Tensor = (
        torch.arange(start=1, end=token_embeddings.shape[1] + 1)
        .unsqueeze(dim=0)
        .unsqueeze(dim=-1)
        .expand(size=token_embeddings.size())
        .to(dtype=token_embeddings.dtype)
        .to(device=token_embeddings.device)
    )

    # Replacing assert with a proper check and exception
    if not (weights.shape == token_embeddings.shape == input_mask_expanded.shape):
        msg = "Mismatch in shapes of weights, token_embeddings, and input_mask_expanded."
        raise ValueError(msg)

    input_mask_expanded *= weights

    sum_embeddings: torch.Tensor = torch.sum(input=token_embeddings * input_mask_expanded, dim=1)

    sum_mask: torch.Tensor = sum_embeddings_mask(
        features=features,
        sum_embeddings=sum_embeddings,
        input_mask_expanded=input_mask_expanded,
    )

    output_vectors.append(sum_embeddings / sum_mask)
    return output_vectors


pooling_funcs: dict[str, Callable[..., list[torch.Tensor]]] = {
    "cls": cls_token_vector,
    "max": max_token_vector,
    "mean": lambda output_vectors, features: mean_token_vector(
        func="mean",
        output_vectors=output_vectors,
        features=features,
    ),
    "mean_sqrt": lambda output_vectors, features: mean_token_vector(
        func="mean_sqrt",
        output_vectors=output_vectors,
        features=features,
    ),
    "weighted_mean": weighted_mean_token_vector,
}
