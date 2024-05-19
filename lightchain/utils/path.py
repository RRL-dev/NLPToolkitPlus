"""The module contains utility functions for working with file paths."""

from collections.abc import Generator
from fnmatch import fnmatch
from os import listdir, rename, walk
from os.path import exists, isdir, join
from typing import Any


def suffix(name: str) -> str:
    """Return the suffix (file extension) of a given file name.

    Args:
    ----
        name (str): The name of the file.

    Returns:
    -------
        str: The file extension or an empty string if none is found.

    Raises:
    ------
        TypeError: If the name is not a string.

    """
    if not isinstance(name, str):
        msg = "name is not type of string"
        raise TypeError(msg)

    value: int = name.rfind(".")
    if 0 < value < len(name) - 1:
        return name[value:]
    return ""


def search_files(
    *, dir_path: str, pattern: str = "*.wav",
) -> Generator[str, Any, None]:
    """Recursively search for files matching the given pattern in a directory.

    Args:
    ----
        dir_path (str): The directory path to search within.
        pattern (str, optional): The file pattern to search for. Defaults to '*.wav'.

    Yields:
    ------
        Generator[str, Any, None]: A generator of file paths.

    Raises:
    ------
        NotADirectoryError: If the provided directory path does not exist.

    """
    if not isdir(s=dir_path) and exists(path=dir_path):  # noqa: PTH112, PTH110
        yield dir_path

    for root, _, files in walk(dir_path):
        for file_name in files:
            file_path: str = join(root, file_name)  # noqa: PTH118

            if fnmatch(name=file_name, pat=pattern):
                yield file_path


def copy_files_top_folder(dir_path: str) -> None:
    """Copy all files in the directory to the top-level directory.

    Args:
    ----
        dir_path (str): The directory path to copy files from.

    """
    files: Generator[str, Any, None] = search_files(dir_path=dir_path)

    for file in files:
        audio_file: str = file.split(sep="/")[-1]
        audio_file = join(dir_path, audio_file)  # noqa: PTH118
        rename(src=file, dst=audio_file)  # noqa: PTH104


def join_files_in_dir(dir_path: str) -> list[str]:
    """Join all files in the directory path to their absolute paths.

    Args:
    ----
        dir_path (str): The directory path.

    Raises:
    ------
        NotADirectoryError: If the directory does not exist.

    Returns:
    -------
        list[str]: List of files with absolute paths in the directory.

    """
    if exists(path=dir_path):  # noqa: PTH110
        files: list[str] = [join(dir_path, file) for file in listdir(path=dir_path)]  # noqa: PTH118
        return files
    msg: str = f"dir path: {dir_path} does not exist"
    raise NotADirectoryError(msg)
