"""The module defines the Phi3Model class, which is a Transformer decoder consisting of multiple Phi3DecoderLayer.

The Phi3Model class is designed for sequence-to-sequence tasks,
utilizing self-attention mechanisms and layer normalization.
"""

from logging import Logger

from torch import FloatTensor, Tensor, nn
from transformers import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.phi3.configuration_phi3 import Phi3Config

from .attention import Phi3Attention
from .block import Phi3DecoderLayer, Phi3RMSNorm

logger: Logger = logging.get_logger(name=__name__)


class Phi3Model(nn.Module):
    """Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`].

    Args:
    ----
        config: Phi3Config

    """

    def __init__(self, config: Phi3Config) -> None:
        """Initialize the Phi3Model with the given configuration.

        Args:
        ----
            config (Phi3Config): Configuration for the model.

        """
        super().__init__()
        self.config: Phi3Config = config
        self.vocab_size: int = config.vocab_size
        self.padding_idx: int = config.pad_token_id

        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.embed_dropout = nn.Dropout(p=config.embd_pdrop)
        self.layers = nn.ModuleList(
            modules=[Phi3DecoderLayer(config=config, layer_idx=i) for i in range(config.num_hidden_layers)],
        )
        self._attn_implementation = Phi3Attention
        self.norm = Phi3RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        kv_cache: tuple[tuple[FloatTensor]] | DynamicCache,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[tuple[FloatTensor]] | DynamicCache] | tuple[Tensor, None]:
        """Perform the forward pass of the model.

        Args:
        ----
            kv_cache (Union[Tuple[Tuple[FloatTensor]], DynamicCache]): Cached past key values.
            input_ids (Tensor): The input ids.
            position_ids (Tensor): The position ids.
            attention_mask (Tensor | None, optional): The attention mask. Defaults to None.
            use_cache (bool, optional): Whether to use cache. Defaults to True.

        Raises:
        ------
            ValueError: If input_ids is not provided.

        Returns:
        -------
            tuple[Tensor, tuple[tuple[Tensor], tuple[Tensor]]] | tuple[Tensor, None]: The output of the model.

        """
        inputs_embeds: Tensor = self.embed_tokens(input_ids)
        batch_size, seq_length = input_ids.shape[:2]

        seq_length: int

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache: bool = not isinstance(kv_cache, Cache)
            if use_legacy_cache:
                kv_cache = DynamicCache.from_legacy_cache(past_key_values=kv_cache)
            past_key_values_length: int = kv_cache.get_usable_length(new_seq_length=seq_length)

        position_ids = position_ids.view(-1, seq_length).long()

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

        hidden_states: Tensor = inputs_embeds
        for layer in self.layers:
            layer_outputs: tuple[Tensor, DynamicCache] = layer(
                kv_cache=kv_cache,
                use_cache=use_cache,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if use_cache:
            kv_cache = kv_cache.to_legacy_cache()  # type: ignore  # noqa: PGH003
            return hidden_states, kv_cache
        return hidden_states, None
