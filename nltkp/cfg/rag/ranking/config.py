"""Module for managing ranking configuration settings.

This module is responsible for loading configuration settings for the Maximal Marginal Relevance (MMR) from a YAML file.
The settings are used across the application to configure MMR behavior in document ranking processes.

Functions:
    load_yaml: Loads YAML configuration files.

Constants:
    FILE: Represents the path to this module file.
    MMR_DICT: Stores the loaded MMR configuration settings as a dictionary.
    MMR_CFG: A SimpleNamespace object for easy attribute-style access to MMR configuration settings.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nltkp.utils import load_yaml

FILE: Path = Path(__file__).resolve()


MMR_DICT: Any | dict[Any, Any] = load_yaml(file_path=(FILE.parent / "mmr.yaml").as_posix())
MMR_CFG = SimpleNamespace(**MMR_DICT)
