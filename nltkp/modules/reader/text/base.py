"""Module for splitting text into chunks using different strategies.

This module is designed to facilitate the splitting of text files into manageable chunks.

Classes:
    Document: Stores content and metadata of a document,
    providing a unique ID and hashing functionality.
    TextChunkProcessor: Base class for handling the processing of text into chunks.
    RecursiveCharacterTextSplitter: Provides methods to split text recursively based on separators,
    with support for regex patterns.
    FolderTextSplitter: Extends RecursiveCharacterTextSplitter to specifically handle the reading,
    and splitting of all text files within a specified folder.

The module supports flexible configuration for text splitting, including the option to maintain,
or remove separators post-splitting and to handle separators as regular expressions.
This makes the module versatile for different text processing tasks,
such as preparing data for natural language processing applications or simply segmenting text files.

Examples of use:
    - Splitting an entire directory of text files into chunks for batch processing.
    - Extracting and managing metadata associated with each chunk, such as file origins and chunks.
    - Handling large text documents that need to be broken down into smaller segments.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from transformers.models.auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nltkp.modules.reader.utils import Document

from .exception import TextFileError

if TYPE_CHECKING:
    from collections.abc import Callable, Sized

    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class TextSplitterConfig(BaseModel):
    """Configuration model for FolderTextSplitter to handle various settings for text splitting.

    Attributes
    ----------
        separators (list[str] | None): Custom separators to use for text splitting, if any.
        model_name (str | None): Name of the tokenizer model to use for more advanced processing.
        folder_path (str | None): Path to the directory containing text files.
        keep_separator (bool): Flag to decide whether to retain separators in the output.
        is_separator_regex (bool): Indicates whether separators are regular expressions.
        extra_args (dict[str, Any]): Additional arguments to pass to the text splitter.

    """

    separators: list[str] | None = ["\n", ".", " "]
    model_name: str | None = None
    folder_path: str | None
    keep_separator: bool = True
    is_separator_regex: bool = False
    extra_args: dict[str, Any] = {}  # To handle any additional kwargs

    class Config:
        """Configuration settings for Pydantic models.

        Attributes
        ----------
            extra (str): Set to 'allow' to permit additional fields during model initialization,
                         which are not explicitly declared in the model. This provides flexibility
                         in accepting additional parameters without causing validation errors.

        """

        extra: str = "allow"


class BaseTextSplitter(RecursiveCharacterTextSplitter):
    """Extends RecursiveCharacterTextSplitter to handle text file splitting in a folder."""

    def __init__(self: BaseTextSplitter, config: TextSplitterConfig) -> None:
        """Initialize the FolderTextSplitter with specific configuration for text splitting."""
        super().__init__(
            separators=config.separators,
            keep_separator=config.keep_separator,
            is_separator_regex=config.is_separator_regex,
            **config.extra_args,
        )

        self.folder_path: str | None = config.folder_path
        self._load_tokenizer(model_name=config.model_name) if config.model_name else None

        self._length_function: Callable[..., int] | Callable[[Sized], int] = (
            self._create_length_function()
        )

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            self._chunk_size: int = self.tokenizer.model_max_length
            self._chunk_overlap = int(self._chunk_size / 4)

    def _load_tokenizer(self: BaseTextSplitter, model_name: str) -> None:
        """Load a tokenizer based on the provided model name."""
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        )

    def _create_length_function(
        self: BaseTextSplitter,
    ) -> Callable[..., int] | Callable[[Sized], int]:
        """Create a length function using the tokenizer to calculate text length.

        Returns
        -------
            A function that takes a string and returns its length as calculated by the tokenizer.

        """
        if isinstance(self.tokenizer, PreTrainedTokenizerBase) and self.tokenizer is not None:
            return lambda text: len(self.tokenizer.encode(text=text))
        return len

    @staticmethod
    def read_text_file(file_path: str | Path) -> str:
        """Read a text file and return its content.

        Args:
        ----
            file_path (str | Path): The path to the text file.

        Returns:
        -------
            str: The content of the text file.

        Raises:
        ------
            TextFileError: If the file does not exist or is not a text file.

        """
        path = Path(file_path)
        if not path.exists():
            msg: str = f"The file {file_path} does not exist."
            raise TextFileError(message=msg)
        if path.suffix != ".txt":
            msg = f"The file {file_path} is not a text file."
            raise TextFileError(message=msg)

        return path.read_text(encoding="utf-8")

    @staticmethod
    def add_documents(
        chunks: list[str],
        file_path: str | Path,
    ) -> list[Document]:
        """Create Document objects for each chunk of text.

        Args:
        ----
            chunks (list[str]): Chunks of text to be converted into Document objects.
            file_path (str | Path): Path of the original text file.

        Returns:
        -------
            list[Document]: A list of Document objects.

        """
        documents: list[Document] = []
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()
        for index, chunk in enumerate(iterable=chunks):
            documents.append(
                Document(page_content=chunk, metadata={"index": index, "file_path": file_path}),
            )
        return documents

    def read_text_files_in_folder(self: BaseTextSplitter) -> list[Document]:
        """Read all text files in a specified folder and return a list of Documents.

        Returns
        -------
            list[Document]: A list of Documents containing the content of each file and metadata.

        Raises
        ------
            TextFileError: If the specified folder path does not exist or is not a directory.

        """
        msg: str
        if self.folder_path is None:
            msg = "Folder path must be specified."
            raise TextFileError(message=msg)
        folder = Path(self.folder_path)
        if not folder.exists() or not folder.is_dir():
            msg = f"The folder {self.folder_path} does not exist or is not a directory."
            raise TextFileError(
                message=msg,
            )

        documents: list[Document] = []
        for text_file in folder.glob(pattern="**/*.txt"):
            content: str = self.read_text_file(file_path=text_file)
            chunks: list[str] = self._split_text(text=content, separators=self._separators)
            documents.extend(self.add_documents(chunks=chunks, file_path=text_file))
        return documents
