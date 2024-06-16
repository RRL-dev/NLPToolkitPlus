from __future__ import annotations

import logging
from os import getenv
from pathlib import Path
from sys import stdout
from types import SimpleNamespace
from typing import Any, ClassVar

from .cuda import set_device
from .path import search_files, suffix
from .seed import set_global_seed
from .serialization import load_yaml

__all__: list[str] = ["search_files", "suffix", "set_device", "set_global_seed"]

FILE: Path = Path(__file__).resolve()
ROOT: Path = FILE.parents[1]

DATASET_CFG_PATH: Path = ROOT / "cfg/dataset/base.yaml"
DATASET_CFG_DICT: Any | dict[Any, Any] = load_yaml(file_path=DATASET_CFG_PATH.as_posix())
DATASET_CFG = SimpleNamespace(**DATASET_CFG_DICT)

EMBEDDINGS_CFG_PATH: Path = ROOT / "cfg/embeddings/documents.yaml"
EMBEDDINGS_CFG_DICT: Any | dict[Any, Any] = load_yaml(file_path=EMBEDDINGS_CFG_PATH.as_posix())
EMBEDDINGS_CFG = SimpleNamespace(**EMBEDDINGS_CFG_DICT)


class CustomFormatter(logging.Formatter):
    """Custom formatter to add colors and timestamps to logs."""

    grey: str = "\x1b[38;21m"
    green: str = "\x1b[32;21m"
    yellow: str = "\x1b[33;21m"
    red: str = "\x1b[31;21m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    formats: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + "%(asctime)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(message)s" + reset,
    }

    def format(self: CustomFormatter, record: logging.LogRecord) -> str:
        """Format the log record with color and timestamp.

        Args:
        ----
            record (logging.LogRecord): The log record to format.

        Returns:
        -------
            str: The formatted log message.

        """
        log_fmt: str | None = self.formats.get(record.levelno)
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_fmt, datefmt=datefmt)
        return formatter.format(record=record)


def set_logging(name="LOGGING_NAME", verbose=True) -> logging.Logger:  # noqa: ANN001, FBT002
    """Sets up logging with color and timestamps, and UTF-8 encoding."""  # noqa: D401
    level: int = logging.INFO if verbose else logging.ERROR

    # Configure the console (stdout) encoding to UTF-8, with checks for compatibility
    formatter = CustomFormatter()  # Use custom formatter with color and timestamps

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(stream=stdout)
    stream_handler.setFormatter(fmt=formatter)
    stream_handler.setLevel(level=level)

    # Set up the logger
    logger: logging.Logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    logger.addHandler(hdlr=stream_handler)
    logger.propagate = False
    return logger


VERBOSE: bool = getenv(key="VERBOSE", default="true").lower() == "true"
LOGGER: logging.Logger = set_logging(name="leLogger", verbose=VERBOSE)
