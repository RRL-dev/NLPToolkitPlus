"""Module for loading and configuring LLAMA settings.

This module provides functionality to load and configure settings for LLAMA from a YAML file.
The settings are loaded into a `SimpleNamespace` for easy access.

Functions:
    load_yaml: Function to load a YAML file and return its contents as a dictionary.

Constants:
    FILE: The path to the current file.
    TINY_LLAMA_DICT: The dictionary containing the loaded TINY_LLAMA settings.
    TINY_LLAMA_CFG: A SimpleNamespace object containing the TINY_LLAMA settings for easy access.

"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nltkp.utils import load_yaml

FILE: Path = Path(__file__).resolve()

TINY_LLAMA_DICT: Any | dict[Any, Any] = load_yaml(file_path=(FILE.parent / "tiny_llama.yaml").as_posix())
TINY_LLAMA_CFG = SimpleNamespace(**TINY_LLAMA_DICT)
