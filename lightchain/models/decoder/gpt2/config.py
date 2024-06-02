"""Main config for GPT2 model."""


class Config:
    """Configuration for the GPT-2 model based on the provided GPT2Config settings."""

    # Basic model settings
    vocab_size: int = 50257  # The size of the vocabulary.
    hidden_size: int = 768  # Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers: int = 12  # Number of hidden layers in the transformer encoder.
    num_attention_heads: int = 12  # Number of attention heads for each attention layer in the transformer encoder.  # noqa: E501
    max_position_embeddings: int = 1024  # The maximum length of the input sequences.

    # Dropout and attention settings
    embd_pdrop: float = 0.1  # Embedding layer dropout probability.
    attn_pdrop: float = 0.1  # Attention dropout probability.
    resid_pdrop: float = 0.1  # Residual connection dropout probability.

    # Additional layer configurations
    n_inner: int | None = (
        None  # The dimensionality of the "inner" feed-forward layers. `None` if not using.  # noqa: E501
    )

    layer_norm_epsilon: float = 1e-5  # Epsilon parameter for the layer normalization components.

    # Scaling and projection settings
    scale_attn_weights: bool = (
        True  # Whether to scale attention weights by inverse square root of depth.
    )

    scale_attn_by_inverse_layer_idx: bool = False  # Scale attention weights by inverse layer index.
    reorder_and_upcast_attn: bool = (
        False  # Whether to reorder and upcast the attention during computation.
    )

    # Output handling
    use_cache: bool = True  # Whether to use caching of past key values for faster generation.
    use_return_dict: bool = False  # Whether to return output in the form of a dictionary or plain tensors.  # noqa: E501

    max_token_length: int = 20
