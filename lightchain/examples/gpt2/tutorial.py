"""The module demonstrates how to use the GPT-2 model with custom configuration."""

from typing import Any

from torch import Tensor, load
from torch.nn.modules.module import Module
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from lightchain.models.decoder import Config, GPT2LMHeadModel, load_weight
from lightchain.utils import LOGGER, set_global_seed

set_global_seed(seed=42)

# Initialize model configuration and model instance
config = Config()
model = GPT2LMHeadModel(config=config)

# Load tokenizer
tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="gpt2",
)

# Load model weights
state_dict: dict[str, Any] = load(f="resources/weights/gpt2-pytorch_model.bin", map_location="cpu")
model = load_weight(model=model, state_dict=state_dict)

# Prepare input text
text = "A machine learning model"
input_ids: Tensor = tokenizer.encode(text=text, return_tensors="pt")  # type: ignore  # noqa: PGH003

# Generate text using the GPT-2 model
output: Tensor = model.sample(input_ids=input_ids)
LOGGER.info("Generated text: %s", tokenizer.decode(token_ids=output[0]))
