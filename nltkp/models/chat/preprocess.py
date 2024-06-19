"""Module for preprocessing text using tokenizers from the transformers library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers.tokenization_utils_base import BatchEncoding

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class PreprocessConfig:
    """Configuration class for preprocessing text input using a tokenizer.

    Attributes
    ----------
        padding (bool): Whether to pad the text to `max_length`.
        add_special_tokens (bool): Whether to add special tokens to the text.

    """

    def __init__(
        self: PreprocessConfig,
        *,
        padding: bool = False,
        add_special_tokens: bool = False,
    ) -> None:
        """Initialize the configuration used for preprocessing text.

        Args:
        ----
            padding (bool): Whether to enable padding to `max_length`.
            add_special_tokens (bool): Whether to add special tokens to the text.

        """
        self.padding: bool = padding
        self.add_special_tokens: bool = add_special_tokens


def preprocess(
    prefix: str,
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int | None = None,
    config: PreprocessConfig | None = None,
) -> BatchEncoding:
    """Prepare text input for processing by a transformer model tokenizer.

    Args:
    ----
        prefix (str): A prefix to prepend to the prompt text.
        messages (list[dict[str, str]]): The main text to be tokenized.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer to use.
        max_length (int | None): The maximum length of the tokenized output.
        config (PreprocessConfig | None): Configuration for padding and special tokens.
            If None, uses default settings.

    Returns:
    -------
        BatchEncoding: A batch encoding with the tokenized text data.

    """
    if config is None:
        config = PreprocessConfig()

    prompt_text: str | list[int] | list[str] | list[list[int]] | BatchEncoding = (
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    inputs: BatchEncoding = tokenizer(
        prefix + prompt_text,  # type: ignore  # noqa: PGH003
        padding=config.padding,
        max_length=max_length,
        add_special_tokens=config.add_special_tokens,
        return_tensors="pt",
    )

    inputs["prompt_text"] = prompt_text
    return inputs
