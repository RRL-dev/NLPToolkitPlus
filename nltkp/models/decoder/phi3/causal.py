"""Phi3 Model for Causal Language Modeling."""

from torch import FloatTensor, Tensor, nn
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

from .model import Phi3Config, Phi3Model


class Phi3ForCausalLM(Phi3PreTrainedModel):
    """Phi3 Model for Causal Language Modeling."""

    def __init__(self, config: Phi3Config) -> None:
        """Initialize the Phi3ForCausalLM model.

        Args:
        ----
            config (Phi3Config): Configuration for the model.

        """
        super().__init__(config=config)
        self.model = Phi3Model(config=config)
        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)

        self.init_weights()

    def forward(  # noqa: PLR0913
        self,
        input_ids: Tensor,
        kv_cache: tuple[tuple[FloatTensor]] | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[tuple[FloatTensor]] | None]:
        """Perform the forward pass of the model.

        Args:
        ----
            input_ids (Tensor): The input ids.
            kv_cache (tuple[tuple[FloatTensor]] | None): The past key values. Defaults to None.
            position_ids (Tensor | None): The position ids. Defaults to None.
            attention_mask (Tensor | None): The attention mask. Defaults to None.
            use_cache (bool, optional): Whether to use cache. Defaults to True.

        Returns:
        -------
            Tuple[Tensor, Optional[Tuple[Tuple[FloatTensor], Tuple[FloatTensor]]]]: The output of the model.

        """
        outputs: tuple[Tensor, tuple[tuple[FloatTensor]]] | tuple[Tensor, None]
        outputs = self.model(
            kv_cache=kv_cache,
            use_cache=use_cache,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        kv_cache = outputs[1]
        hidden_states: Tensor = outputs[0]
        logits: Tensor = self.lm_head(hidden_states)
        return logits, kv_cache
