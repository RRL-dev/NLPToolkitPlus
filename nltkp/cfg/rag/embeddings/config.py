"""Module for managing embeddings configuration settings.

This module is responsible for loading configuration settings for embeddings from a YAML file.
The settings are used across the application to configure the behavior of embeddings in various processes.

Functions:
    load_yaml: Loads YAML configuration files.

Constants:
    FILE: Represents the path to this module file.
    EMBEDDINGS_DICT: Stores the loaded embeddings configuration settings as a dictionary.
    EMBEDDINGS_CFG: A SimpleNamespace object for easy attribute-style access to embeddings configuration settings.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nltkp.utils import load_yaml

FILE: Path = Path(__file__).resolve()


EMBEDDINGS_DICT: Any | dict[Any, Any] = load_yaml(file_path=(FILE.parent / "base.yaml").as_posix())
EMBEDDINGS_CFG = SimpleNamespace(**EMBEDDINGS_DICT)
