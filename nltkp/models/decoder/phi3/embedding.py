"""Module for Phi3 Rotary Embedding."""

from torch import Tensor, arange, cat, nn


class Phi3RotaryEmbedding(nn.Module):
    """Rotary positional embedding for the Phi3 model."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000) -> None:
        """Initialize the Phi3RotaryEmbedding module.

        Args:
        ----
            dim (int): Dimension of the embeddings.
            max_position_embeddings (int, optional): Maximum number of position embeddings. Defaults to 2048.
            base (int, optional): Base for computing the frequency. Defaults to 10000.

        """
        super().__init__()
        self.dim: int = dim
        self.max_position_embeddings: int = max_position_embeddings
        self.base: int = base
        self.inv_freq: Tensor = 1.0 / (self.base ** (arange(start=0, end=self.dim, step=2).float() / self.dim))

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass to compute rotary positional embeddings.

        Args:
        ----
            x (torch.Tensor): Input tensor with shape [batch_size, num_heads, seq_len, head_dim].
            position_ids (torch.Tensor): Position ids with shape [seq_len].

        Returns:
        -------
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing cos and sin tensors.

        """
        inv_freq_expanded: Tensor = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device=position_ids.device)
        )
        position_ids_expanded: Tensor = position_ids[:, None, :].float()
        freqs: Tensor = (inv_freq_expanded @ position_ids_expanded).transpose(dim0=1, dim1=2)
        emb: Tensor = cat(tensors=(freqs, freqs), dim=-1)
        cos: Tensor = emb.cos()
        sin: Tensor = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the dimensions of the input tensor."""
    x1: Tensor
    x2: Tensor
    half_size: int = x.size(dim=-1) // 2
    x1, x2 = x[..., :half_size], x[..., half_size:]
    return cat(tensors=(-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    """Apply Rotary Position Embedding to the query and key tensors.

    Args:
    ----
        query (Tensor): The query tensor.
        key (Tensor): The key tensor.
        cos (Tensor): The cosine part of the rotary embedding.
        sin (Tensor): The sine part of the rotary embedding.
        unsqueeze_dim (int, optional): The dimension along which to unsqueeze cos and sin. Defaults to 1.

    Returns:
    -------
        tuple[Tensor, Tensor]: The query and key tensors rotated using the Rotary Position Embedding.

    """
    cos = cos.unsqueeze(dim=unsqueeze_dim)
    sin = sin.unsqueeze(dim=unsqueeze_dim)
    q_embed: Tensor = (query * cos) + (rotate_half(x=query) * sin)
    k_embed: Tensor = (key * cos) + (rotate_half(x=key) * sin)
    return q_embed, k_embed
