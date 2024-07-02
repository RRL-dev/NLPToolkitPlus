"""Module for managing FAISS configuration settings.

This module is responsible for loading configuration settings for FAISS from a YAML file.
The settings are used across the application to configure FAISS behavior in vector search and retrieval processes.

Functions:
    load_yaml: Loads YAML configuration files.

Constants:
    FILE: Represents the path to this module file.
    FAISS_DICT: Stores the loaded FAISS configuration settings as a dictionary.
    FAISS_CFG: A SimpleNamespace object for easy attribute-style access to FAISS configuration settings.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nltkp.utils import load_yaml

FILE: Path = Path(__file__).resolve()


FAISS_DICT: Any | dict[Any, Any] = load_yaml(file_path=(FILE.parent / "faiss.yaml").as_posix())
FAISS_CFG = SimpleNamespace(**FAISS_DICT)
