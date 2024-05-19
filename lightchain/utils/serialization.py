"""Basic functionality for serialization."""

from pickle import HIGHEST_PROTOCOL, dump, load
from typing import TypeVar

from yaml import safe_load

from superchain.utils.path import suffix

T = TypeVar("T")


def load_pickle(filepath: str) -> T: # type: ignore  # noqa: PGH003
    """Load a pickle file and return its contents.

    Args:
    ----
        filepath (str): The path to the pickle file.

    Returns:
    -------
        T: The object loaded from the pickle file.

    Raises:
    ------
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.

    """
    with open(file=filepath, mode="rb") as file:  # noqa: PTH123
        data: T = load(file=file)  # noqa: S301
    return data


def dump_pickle(*, data: object, filepath: str, mode: str = "wb") -> None:
    """Dump object to pickle.

    Args:
    ----
        data (object): object to save.
        filepath (str): pkl path or directory.
        mode (str, optional): write binary. Defaults to "wb".

    """
    with open(  # pylint: disable=W1514  # noqa: PTH123
        file=filepath,
        mode=mode,
    ) as file:
        dump(obj=data, file=file, protocol=HIGHEST_PROTOCOL)


def load_yaml(file_path: str) -> dict[str, str | int | float | bool | None | list | dict] | dict:
    """Load a YAML file and return its contents as a dictionary.

    Args:
    ----
        file_path (str): The path to the YAML file.

    Returns:
    -------
        Union[Dict[str, Union[str, int, float, bool, None, list, dict]], dict]: The contents of the YAML file as a dictionary.

    Raises:
    ------
        AssertionError: If the file is not a YAML file.

    """  # noqa: E501
    assert suffix(name=file_path) == ".yaml", f"file {file_path} is not a YAML file"  # noqa: S101

    with open(file=file_path, encoding="utf-8") as obj:  # noqa: PTH123
        file_content: str = obj.read()

    data: dict[str, str | int | float | bool | None | list | dict] = (
        safe_load(stream=file_content) or {}
    )
    return data
