"""The module provides functions to load weights into a PyTorch model from a state dictionary."""

from typing import Any

from torch.nn import Module


def transform_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Transform state dictionary keys to match model parameters."""
    old_keys: list[str] = []
    new_keys: list[str] = []

    for key in state_dict:
        new_key: str | None = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys, strict=False):
        state_dict[new_key] = state_dict.pop(old_key)

    return state_dict


def load_module_state(
    module: Module,
    state_dict: dict[str, Any],
    metadata: dict[str, Any] | None,
    prefix: str = "",
) -> None:
    """Recursively load state dictionary into module."""
    {} if metadata is None else metadata.get(prefix[:-1], {})
    module.load_state_dict(
        state_dict={k: v for k, v in state_dict.items() if k.startswith(prefix)},
        strict=False,
    )
    for name, child in module.named_children():
        if child is not None:
            load_module_state(
                module=child,
                state_dict=state_dict,
                metadata=metadata,
                prefix=prefix + name + ".",
            )


def load_weight(model: Module, state_dict: dict[str, Any]) -> Module:
    """Load weights into the given model.

    Args:
    ----
        model (Module): The model into which weights are to be loaded.
        state_dict (dict[str, Any]): The state dictionary containing model weights.

    Returns:
    -------
        Module: The model with loaded weights.

    """
    state_dict = transform_keys(state_dict=state_dict)
    metadata: dict[str, Any] | None = getattr(state_dict, "_metadata", None)
    state_dict_copy: dict[str, Any] = state_dict.copy()

    if metadata is not None:
        state_dict_copy["_metadata"] = metadata

    start_model: Module = model
    if hasattr(model, "transformer") and all(not s.startswith("transformer.") for s in state_dict):
        start_model = model.transformer

    load_module_state(module=start_model, state_dict=state_dict_copy, metadata=metadata)

    if hasattr(model, "set_tied"):
        model.set_tied()

    return model
