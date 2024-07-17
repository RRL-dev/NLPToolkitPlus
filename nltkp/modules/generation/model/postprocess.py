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

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    from nltkp.modules.generation.params import SamplingParams


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


def output_postprocess(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model_params: SamplingParams,
    return_type: ReturnType = ReturnType.NEW_TEXT,
    *,
    clean_up_tokenization_spaces: bool = True,
) -> list[dict]:
    """Post-process the model output to generate text or token IDs.

    Args:
    ----
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer used for decoding token IDs.
        model_params (SamplingParams): The parameters containing the output from the model.
        return_type (ReturnType, optional): The type of return value, either NEW_TEXT, FULL_TEXT, or TENSORS.
        clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces in the generated text.

    Returns:
    -------
        List[dict]: A list of dictionaries containing the processed model outputs.

    Raises:
    ------
        ValueError: If the generated sequence or input IDs are empty or invalid,
        or if an unsupported return_type is provided.

    """
    output_ids: Any = model_params.output_ids
    prompt_text: Any = model_params.prompt_text

    # Proper tensor checking
    if output_ids is None or output_ids.nelement() == 0:
        msg = "Empty or invalid 'generated_sequence' provided."
        raise ValueError(msg)

    # Convert tensor to list if necessary
    if hasattr(output_ids, "numpy"):
        output_ids = [seq.numpy().tolist() for seq in output_ids]
    elif isinstance(output_ids[0], list):
        output_ids = [seq.tolist() for seq in output_ids]

    records: list[dict] = []

    for sequence in output_ids:
        if return_type == ReturnType.TENSORS:
            record: dict[str, Any] = {"generated_token_ids": sequence}
        elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
            text: str = tokenizer.decode(
                token_ids=sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

            record = {"generated_text": text, "prompt_text": prompt_text}
        else:
            msg: str = f"Unsupported return_type: {return_type}"
            raise ValueError(msg)

        records.append(record)

    return records
