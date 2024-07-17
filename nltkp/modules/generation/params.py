"""The module defines the SamplingParams model for use in generating model samples with various configuration options.

The SamplingParams model includes fields for input tensors, configuration options,
and other parameters needed for sampling from a model.
"""

from pydantic import BaseModel, ConfigDict, Field
from torch import FloatTensor, Tensor


class SamplingParams(BaseModel):
    """Parameters for sampling from a model.

    Attributes
    ----------
    do_sample : bool
        Whether to use sampling; use greedy decoding otherwise.
    tokens_ids : Tensor | None
        The input tensor containing token IDs.
    use_cache : bool
        Whether to use the cache.
    position_ids : Tensor | None
        The tensor containing position IDs.
    attention_mask : Tensor | None
        The attention mask tensor.
    cache_position : Tensor | None
        The tensor containing cache positions.
    token_type_ids : Tensor | None
        The tensor containing token type IDs.
    max_new_tokens : int
        The maximum number of new tokens to generate.
    kv_cache : tuple[tuple[FloatTensor]] | None
        The past key values tensor.
    output_ids : Tensor | None
        The output tensor containing generated token IDs.
    prompt_text : str | None
        The prompt text for generating samples.
    top_k : int
        Top K tokens to be considered in generation.
    model_name : str
        Model name for generating responses.

    """

    kv_cache: tuple[tuple[FloatTensor]] | None = Field(default=None, description="The past key values tensor.")
    use_cache: bool = Field(default=True, description="Whether to use the cache.")
    do_sample: bool = Field(default=True, description="Flag to determine if sampling is used.")
    tokens_ids: Tensor | None = Field(default=None, description="The input tensor containing token IDs.")
    output_ids: Tensor | None = Field(default=None, description="The output tensor containing generated token IDs.")
    prompt_text: str | None = Field(default=None, description="The prompt text for generating samples.")
    position_ids: Tensor | None = Field(default=None, description="The tensor containing position IDs.")
    attention_mask: Tensor | None = Field(default=None, description="The attention mask tensor.")
    cache_position: Tensor | None = Field(default=None, description="The tensor containing cache positions.")
    token_type_ids: Tensor | None = Field(default=None, description="The tensor containing token type IDs.")
    max_new_tokens: int = Field(default=512, description="Maximum new tokens to generate.")
    top_k: int = Field(default=50, description="Top K tokens to be considered in generation.")
    model_name: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct",
        description="Model name for generating responses.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
