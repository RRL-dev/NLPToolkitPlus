"""The module contains the FilesReader class for reading and caching files based on configuration."""

from __future__ import annotations

from os.path import exists, splitext
from typing import TYPE_CHECKING, Any

from pandas.core.frame import DataFrame

from nltkp.utils import LOGGER

from .format import read_csv_format, read_xlsx_format

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import SimpleNamespace

    from pandas.core.frame import DataFrame


def suffix(name: str) -> str:
    """Get the file extension."""
    return splitext(p=name)[1]  # noqa: PTH122


def read_file(file_path: str | None, file_format: str | None = "csv") -> DataFrame:
    """Read tabular data.

    Args:
    ----
        file_path (str): path of csv or xlsx file.
        file_format (str): the file format, csv or xlsx. Defaults to 'csv'.

    Returns:
    -------
        DataFrame: Two-dimensional, size-mutable, potentially heterogeneous tabular data

    """
    if file_path is None or not exists(path=file_path):  # noqa: PTH110
        msg = "file path not defined"
        raise ValueError(msg)

    if file_format is None:
        file_format = "csv"
        LOGGER.warning(msg="file format is None, set as default to csv")

    file_reader: dict[str, Callable | Any] = {
        "csv": read_csv_format,
        "xlsx": read_xlsx_format,
    }

    if file_format not in file_reader:
        msg: str = f"file format {file_format} not defined as csv or xlsx"
        raise ValueError(msg)

    if file_path.endswith("csv"):
        return file_reader[file_format](file_path)

    if file_path.endswith("xlsx"):
        return file_reader[file_format](file_format)

    msg = f"file path: {file_path} not type of csv or xlsx"
    raise ValueError(msg)


class FilesReader:
    """Class for reading and caching files based on configuration."""

    def __init__(self: FilesReader, config: SimpleNamespace) -> None:
        """Initialize FilesReader with the dataset configuration and cache all files.

        Args:
        ----
            config (SimpleNamespace): Configuration specifying the dataset.

        """
        self.config: SimpleNamespace = config
        self.cache: dict[str, DataFrame] = {}
        self.file_readers: dict[str, Callable[[str], DataFrame]] = {
            "csv": read_csv_format,
            "xlsx": read_xlsx_format,
        }
        self._cache_all_files()

    def _cache_all_files(self: FilesReader) -> None:
        """Cache all files specified in the configuration."""
        file_paths: dict[str, str]
        for file_type, file_paths in self.config.dataset["type"].items():
            for file_key, file_path in file_paths.items():
                self._cache_file(file_type=file_type, file_key=file_key, file_path=file_path)

    def _cache_file(self: FilesReader, file_type: str, file_key: str, file_path: str) -> None:
        """Cache a single file specified in the configuration.

        Args:
        ----
            file_type (str): The type of file (e.g., "csv", "xlsx").
            file_key (str): A unique identifier for the file.
            file_path (str): The path to the file.

        """
        if not exists(path=file_path):  # noqa: PTH110
            LOGGER.warning(msg=f"File path {file_path} does not exist. Skipping.")
            return

        if file_type not in self.file_readers:
            LOGGER.warning(msg=f"File type {file_type} not supported. Skipping {file_path}.")
            return

        LOGGER.info(msg=f"Caching file: {file_path}")
        df: DataFrame = self.file_readers[file_type](file_path)
        if df.empty:
            LOGGER.warning(msg=f"File {file_path} is empty or invalid. Skipping.")
        else:
            self.cache[file_key] = df

    def get_from_cache(self: FilesReader, file_key: str) -> DataFrame | None:
        """Retrieve a file from cache by its key.

        Args:
        ----
            file_key (str): The key of the file to retrieve from cache.

        Returns:
        -------
            Optional[DataFrame]: The cached DataFrame or None if not in cache.

        """
        return self.cache.get(file_key)

    def clear_cache(self: FilesReader) -> None:
        """Clear the cache."""
        self.cache.clear()
        LOGGER.info(msg="Cache cleared.")

    def get_all_cached(self: FilesReader) -> dict[str, DataFrame]:
        """Get all cached DataFrames.

        Returns
        -------
            dict[str, DataFrame]: Dictionary with file names as keys and cached DataFrames as values.

        """
        return self.cache
