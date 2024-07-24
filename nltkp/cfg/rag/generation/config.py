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

PHI3_INST_DICT: Any | dict[Any, Any] = load_yaml(file_path=(FILE.parent / "phi3_instruct.yaml").as_posix())
PHI3_INST_CFG = SimpleNamespace(**PHI3_INST_DICT)

COT_DICT: Any | dict[Any, Any] = load_yaml(file_path=(FILE.parent / "cot.yaml").as_posix())
COT_CFG = SimpleNamespace(**COT_DICT)
