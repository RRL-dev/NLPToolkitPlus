"""Model utils."""

from __future__ import annotations

import re

from torch import Tensor, device
from transformers.tokenization_utils_base import BatchEncoding


def clean_module_name(name: str) -> str:
    """Clean module name by removing any non-alphanumeric characters.

    This function is useful for standardizing module names to a format that does not
    contain special characters and is in CamelCase.

    Args:
    ----
        name (str): The original module name which may contain special characters like
                    dashes, periods, or underscores.

    Returns:
    -------
        str: A cleaned-up version of the module name with CamelCase formatting and
             no special characters.

    """
    # Step 1: Replace any non-alphanumeric characters with a space for easy word boundary detection.
    cleaned: str = re.sub(pattern=r"[^a-zA-Z0-9]", repl=" ", string=name)

    # Step 2: Split the string into words and capitalize each word
    words: list[str] = cleaned.split()
    capitalized_words: list[str] = [word.capitalize() for word in words]

    # Step 3: Join all the words without spaces to form the final string
    final_name: str = "".join(capitalized_words)
    return final_name


def move_tensors_to_device(batch_encoding: BatchEncoding, device: device) -> BatchEncoding:
    """Move all tensor objects within a BatchEncoding to the specified device.

    Args:
    ----
        batch_encoding (BatchEncoding): The BatchEncoding object containing the input data.
        device (device): The torch device (e.g., cuda or cpu) to which tensors should be moved.

    Returns:
    -------
        BatchEncoding: The BatchEncoding object with tensors moved to the specified device.

    """
    # Create a new dictionary to hold the updated data
    updated_data: dict[str, str | Tensor] = {}

    # Iterate over all items in the original BatchEncoding data
    for key, value in batch_encoding.data.items():
        # Check if the value is an instance of Tensor and move it to the device
        if isinstance(value, Tensor):
            updated_data[key] = value.to(device=device)
        else:
            # If the value is not a tensor, just carry it over unchanged
            updated_data[key] = value

    # Create a new BatchEncoding with the updated data
    return BatchEncoding(data=updated_data)
