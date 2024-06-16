"""The module defines the SentenceMPNet class, which integrates the MPNet model."""

from __future__ import annotations

from typing import Any, Literal

import torch
from numpy import dtype, ndarray, stack
from torch import Tensor, cat, inference_mode, nn
from transformers import MPNetModel, MPNetTokenizerFast

from lightchain.utils import set_device

from .pooling import BasePooling


class SentenceMPNet(nn.Module):
    """A class encapsulating the MPNet model along with a pooling layer and tokenizer."""

    def __init__(self: SentenceMPNet, pooling_modes: dict) -> None:
        """Initialize the SentenceMPNet model with specified model path or name."""
        super().__init__()

        self.tokenizer = MPNetTokenizerFast.from_pretrained(
            pretrained_model_name_or_path="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        )

        self.pooling = BasePooling(**pooling_modes)

        self.model_name = "multi-qa-mpnet-base-dot-v1"

        self.register_module(
            name=self.model_name,
            module=MPNetModel.from_pretrained(
                pretrained_model_name_or_path=f"sentence-transformers/{self.model_name}",
            ),  # type: ignore  # noqa: PGH003
        )

        self.device: torch.device = set_device()
        self.to(device=self.device)

    def forward(self: SentenceMPNet, features: dict[str, Tensor]) -> Tensor:
        """Forward pass of the model to compute the embeddings."""
        output_states: Any = self.get_submodule(target=self.model_name)(
            **features,
            return_dict=False,
        )
        output_tokens: Any = output_states[0]
        features.update(token_embeddings=output_tokens, attention_mask=features["attention_mask"])

        if self.get_submodule(target=self.model_name).config.output_hidden_states:
            all_layer_idx: Literal[2, 1] = 2 if len(output_states) >= 3 else 1  # noqa: PLR2004
            hidden_states = output_states[all_layer_idx]
            features.update(all_layer_embeddings=hidden_states)

        embeddings: list[Tensor] = self.pooling.apply_pooling(output_vectors=[], features=features)

        return torch.cat(tensors=embeddings, dim=1)

    def encode(
        self: SentenceMPNet,
        sentences: str | list[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,  # noqa: FBT001, FBT002
        normalize_embeddings: bool = True,  # noqa: FBT001, FBT002
    ) -> ndarray[Any, dtype[Any]] | Tensor:
        """Encode a list of sentences into embeddings using a pre-trained model.

        Args:
        ----
            sentences (str | list[str]): A single sentence or a list of sentences to encode.
            batch_size (int): The number of sentences to process in each batch.
            convert_to_numpy (bool): Flag to determine if the output should be converted to numpy arrays.
            device (str | None): The device to perform the computation on. Defaults to 'cpu'.
            normalize_embeddings (bool): Whether to L2-normalize the embeddings.

        Returns:
        -------
            list[Tensor] | ndarray | Tensor: The embeddings as a list of tensors, a single tensor, or a numpy array,
                                            depending on the `convert_to_numpy` flag.

        """  # noqa: E501
        if isinstance(sentences, str):
            sentences = [sentences]

        self.eval()

        all_embeddings: list[Tensor] = []
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch: list[str] = sentences[start_index : start_index + batch_size]
            encoded_input: dict[Any, Any] = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}
            with inference_mode():
                embeddings: Tensor = self(encoded_input)

            if normalize_embeddings:
                embeddings = nn.functional.normalize(input=embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        if convert_to_numpy:
            # Convert all embeddings to NumPy arrays
            return stack(arrays=[emb.cpu().numpy() for emb in all_embeddings])
        # Concatenate all tensors to form a single tensor
        return cat(tensors=all_embeddings, dim=0)


if __name__ == "__main__":
    # Example usage:
    # Initialize the SentenceMPNet module with the model path
    sentence_mpnet = SentenceMPNet(pooling_modes={"mean": True})

    sentence_mpnet.eval()
    query = "How big is London?"
    context: list[str] = [
        "London is known for its finacial district",
        "London has 9,787,426 inhabitants at the 2011 census",
        "The United Kingdom is the fourth largest exporter of goods in the world",
    ]

    query_embedding: ndarray[Any, dtype[Any]] | Tensor = sentence_mpnet.encode(
        sentences=query,
        convert_to_numpy=False,
        normalize_embeddings=True,
    )
    context_embedding: ndarray[Any, dtype[Any]] | Tensor = sentence_mpnet.encode(
        sentences=context,
        convert_to_numpy=False,
        normalize_embeddings=True,
    )
