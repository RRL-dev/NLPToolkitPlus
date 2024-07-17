"""Model utils."""

import re

from torch import Size, Tensor, device, dtype, float16, full, triu, zeros
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


def create_upper_triangular_causal_mask(
    attn_weights_shape: Size,
    dtype: dtype = float16,
    device: device = "cuda",  # type: ignore  # noqa: PGH003
    inf_value: float = -65504.0,
) -> Tensor:
    """Create an upper triangular causal mask with the same shape as the attention weights.

    Args:
    ----
        attn_weights_shape (torch.Size): The shape of the attention weights tensor.
        dtype (torch.dtype): The data type of the mask. Default is torch.float16.
        device (torch.device): The device to create the mask on. Default is 'cuda'.
        inf_value (float): The value to use for masking (e.g., -infinity). Default is -65504.0.

    Returns:
    -------
        torch.Tensor: An upper triangular causal mask with the same shape as the attention weights.

    """
    batch_size, num_heads, seq_len, _ = attn_weights_shape
    mask: Tensor = triu(
        input=full(size=(seq_len, seq_len), fill_value=inf_value, dtype=dtype, device=device),
        diagonal=1,
    )
    mask = mask.unsqueeze(dim=0).unsqueeze(dim=0)  # Shape [1, 1, seq_len, seq_len]
    mask = mask.expand(batch_size, num_heads, -1, -1)  # Shape [batch_size, num_heads, seq_len, seq_len]
    mask.masked_fill_(mask=mask == inf_value, value=0.0)  # Set diagonal and lower triangular to 0
    return mask + triu(
        input=full(size=(seq_len, seq_len), fill_value=inf_value, dtype=dtype, device=device),
        diagonal=1,
    )


def create_general_causal_mask(
    attn_weights_shape: Size,
    dtype: dtype = float16,
    device: device = "cuda",  # type: ignore  # noqa: PGH003
    inf_value: float = -65504.0,
) -> Tensor:
    """Create a causal mask that only affects the last seq_len elements in the last dimension of the attention weights.

    Args:
    ----
        attn_weights_shape (torch.Size): The shape of the attention weights tensor.
        dtype (torch.dtype): The data type of the mask. Default is torch.float16.
        device (torch.device): The device to create the mask on. Default is 'cuda'.
        inf_value (float): The value to use for masking (e.g., -infinity). Default is -65504.0.

    Returns:
    -------
        torch.Tensor: A causal mask with the same shape as the attention weights.

    """
    seq_len: int
    num_heads: int
    batch_size: int
    total_seq_len: int
    batch_size, num_heads, seq_len, total_seq_len = attn_weights_shape

    # Create a mask of zeros
    mask: Tensor = zeros(size=(seq_len, total_seq_len), dtype=dtype, device=device)

    # Mask the last seq_len elements
    for i in range(seq_len):
        mask[i, total_seq_len - seq_len + i + 1 :] = inf_value

    return mask.unsqueeze(dim=0).unsqueeze(dim=0).expand(batch_size, num_heads, -1, -1)
