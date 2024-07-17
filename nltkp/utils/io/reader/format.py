"""The module provides utilities for reading and caching files using pandas, with support for CSV and XLSX formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, final
from zipfile import BadZipFile

from pandas import DataFrame
from pandas.io.excel import read_excel
from pandas.io.parsers import read_csv

from nltkp.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Callable


@final
class DataFormat:
    """Data format for pandas reader."""

    def __init__(self: DataFormat, func: Callable[[str], Any]) -> None:
        """Initialize the DataFormat with a function.

        Args:
        ----
            func (Callable[[str], Any]): The function to be decorated.

        """
        super().__init__()
        self._func: Callable[[str], Any] = func

    def __call__(self: DataFormat, file_path: str) -> DataFrame:
        """Read file decorator.

        Args:
        ----
            file_path (str): The path to the file.

        Returns:
        -------
            Any: The result of the decorated function.

        Raises:
        ------
            TypeError: If file_path is not a string.
            NotImplementedError: If the file extension is not supported.

        """
        if not isinstance(file_path, str):
            msg: str = f"file path need to be str, got {type(file_path)}"
            raise TypeError(msg)

        if file_path.endswith(".csv") and "csv" in self._func.__name__:
            return self._func(file_path)

        if file_path.endswith(".xlsx") or file_path.endswith(".xls") and "xlsx" in self._func.__name__:
            return self._func(file_path)

        msg = f"pandas cannot handle file type {file_path} with the function: {self._func.__name__}"
        raise NotImplementedError(
            msg,
        )

    def __repr__(self: DataFormat) -> str:
        """Return a string representation of the DataFormat instance."""
        return "DataFormat"


@DataFormat
def read_csv_format(file_path: str) -> DataFrame:
    """Read csv with format checking.

    Args:
    ----
        file_path (str): Path of csv file.

    Returns:
    -------
        DataFrame: The read DataFrame.

    """
    return read_csv(filepath_or_buffer=file_path, index_col=0)


@DataFormat
def read_xlsx_format(file_path: str) -> DataFrame:
    """Read xlsx file with format checking.

    Args:
    ----
        file_path (str): Path of xlsx file.

    Returns:
    -------
        DataFrame: The read DataFrame or an empty DataFrame if an error occurs.

    """
    try:
        return read_excel(io=file_path, engine="openpyxl")
    except (ValueError, BadZipFile) as e:
        LOGGER.error(msg=f"Error reading XLSX file {file_path}: {e}")
        return DataFrame()


if __name__ == "__main__":
    data: DataFrame = read_csv_format(file_path=".../out.csv")
