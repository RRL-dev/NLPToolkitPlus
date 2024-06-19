"""The module provides functionality for processing the output of nlp models.

It defines:
- `ReturnType`: an Enum to specify the desired format of the processing results.
- `postprocess`: a function to convert model outputs into human-readable forms or structured data.

The functions are designed to be flexible, supporting different return types and configurations,
allowing for easy integration into NLP data pipelines.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from torch import Tensor

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class ReturnType(Enum):
    """An enumeration to specify the type of data to be returned by the postprocess function.

    Attributes
    ----------
        TENSORS (int): Return the generated token IDs as lists.
        NEW_TEXT (int): Return the text generated after the prompt text.
        FULL_TEXT (int): Return the full text including the prompt text and the generated text.

    """

    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


def postprocess(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model_outputs: dict,
    return_type: ReturnType = ReturnType.FULL_TEXT,
    *,
    clean_up_tokenization_spaces: bool = True,
) -> list[dict]:
    """Process the output from a model based on the specified return type.

    Args:
    ----
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to decode ids.
        model_outputs (dict): Dictionary containing model output data.
        return_type (ReturnType): Specifies the type of return value desired.
        clean_up_tokenization_spaces (bool): Whether to clean up tokenization spaces.

    Returns:
    -------
        List[dict]: A list of dictionaries containing processed records based on the return type.

    Raises:
    ------
        ValueError: If an unknown return_type is provided.

    """
    input_ids: Any | None = model_outputs.get("input_ids")
    prompt_text: Any | None = model_outputs.get("prompt_text")
    generated_sequence: Any | None = model_outputs.get("generated_sequence")

    # Proper tensor checking
    if generated_sequence is None or generated_sequence.nelement() == 0:
        msg = "Empty or invalid 'generated_sequence' provided."
        raise ValueError(msg)
    if input_ids is None or input_ids.nelement() == 0:
        msg = "Empty or invalid 'input_ids' provided."
        raise ValueError(msg)

    # Convert tensor to list if necessary
    if hasattr(generated_sequence, "numpy"):
        generated_sequence = [seq.numpy().tolist() for seq in generated_sequence]
    elif isinstance(generated_sequence[0], list):
        generated_sequence = [seq.tolist() for seq in generated_sequence]

    records: list[dict] = []

    for sequence in generated_sequence[0]:
        if return_type == ReturnType.TENSORS:
            record: dict[str, Any] = {"generated_token_ids": sequence}
        elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
            text: str = tokenizer.decode(
                token_ids=sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            prompt_length: int = len(
                tokenizer.decode(
                    token_ids=input_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                ),
            )
            if return_type == ReturnType.FULL_TEXT:
                all_text: str = prompt_text + text[prompt_length:]  # type: ignore  # noqa: PGH003
                record = {"generated_text": all_text}
            else:
                new_text: str = text[prompt_length:]
                record = {"generated_text": new_text}
        else:
            msg = f"Unsupported return_type: {return_type}"
            raise ValueError(msg)

        records.append(record)

    return records
