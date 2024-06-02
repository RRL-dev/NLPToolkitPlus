"""The module use matplotlib with seaborn for visualization."""

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn.modules.module import Module
from transformers import BatchEncoding, GPT2Tokenizer

from lightchain.models import Config, ForwardParams, GPT2LMHeadModel, load_weight

if TYPE_CHECKING:
    from numpy import ndarray


def plot_attention_heads(
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
    tokens: list[str],
    layer: int,
    heads: list[int],
) -> None:
    """Visualize the attention heads using past key values.

    Args:
    ----
        past_key_values (List[Tuple[Tensor, Tensor]]): The past key values containing the attention.
        tokens (List[str]): The tokens in the input text.
        layer (int): The layer number to visualize.
        heads (List[int]): The attention heads to visualize.

    """
    for head in heads:
        plt.figure(figsize=(10, 10))
        key: torch.Tensor = past_key_values[layer][0][0, head].detach().cpu().numpy()
        value: torch.Tensor = past_key_values[layer][1][0, head].detach().cpu().numpy()
        attention: torch.Tensor | ndarray = torch.matmul(
            input=torch.tensor(data=key),
            other=torch.tensor(data=value).transpose(dim0=-1, dim1=-2),
        ).numpy()

        # Normalize the attention weights
        attention = (attention - attention.min()) / (attention.max() - attention.min())

        sns.heatmap(data=attention, xticklabels=tokens, yticklabels=tokens, cmap="RdPu")
        plt.title(label=f"Layer {layer+1}, Head {head+1}")
        plt.xlabel(xlabel="Query Position")
        plt.ylabel(ylabel="Key Position")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()


# Tokenizer and model setup
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="gpt2")
config = Config()  # Ensure this is properly defined elsewhere in your application
model = GPT2LMHeadModel(config=config)

# Load pretrained weights
state_dict: OrderedDict = torch.load(
    f="/home/roni/Downloads/gpt2-pytorch_model.bin",
    map_location="cpu",
)

model: Module = load_weight(
    model=model,
    state_dict=state_dict,
)  # Ensure this function is correctly implemented to return GPT2LMHeadModel

# Example usage
input_text = "An example of machine learning"
model_inputs: BatchEncoding = tokenizer(input_text, return_tensors="pt")
input_ids: torch.Tensor | Any = model_inputs["input_ids"]

# Ensure attention_mask and position_ids are generated
attention_mask: torch.Tensor = model_inputs.get("attention_mask", torch.ones_like(input=input_ids))
position_ids: torch.Tensor = torch.arange(end=input_ids.size(-1)).unsqueeze(dim=0)

# Prepare inputs
forward_params = ForwardParams(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
)

model.eval()
with torch.inference_mode():
    outputs: Any = model(forward_params)

# Extract past key values for visualization
past_key_values = outputs.past_key_values
tokens: Any = tokenizer.convert_ids_to_tokens(input_ids[0])

# Visualize specific attention heads
plot_attention_heads(past_key_values=past_key_values, tokens=tokens, layer=0, heads=[0, 1, 2])
