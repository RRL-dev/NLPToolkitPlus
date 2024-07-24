"""Module for hierarchical text splitting.

This module provides functionality to split text files into smaller chunks,
while preserving hierarchical category information.
"""

from pathlib import Path

from .base import BaseTextSplitter, Document, TextSplitterConfig


class HierarchicalTextSplitter(BaseTextSplitter):
    """Class to split text files into smaller chunks with hierarchical category handling.

    This class extends BaseTextSplitter to add functionality for hierarchical
    category handling. It reads text files from a specified folder, splits them
    into smaller chunks, and includes category metadata based on the folder structure.
    """

    def __init__(self, config: TextSplitterConfig) -> None:
        """Initialize the HierarchicalTextSplitter with the given configuration.

        Args:
        ----
            config (TextSplitterConfig): Configuration for the text splitter,
                including folder path, separators, model name, and hierarchical flag.

        """
        super().__init__(config=config)

    def _get_category(self, path: Path) -> str:
        """Recursively find the category of a text file based on its path."""
        if not self.folder_path:
            msg = "Folder path must be specified."
            raise ValueError(msg)

        category = []
        current: Path = path.parent
        while current != Path(self.folder_path):
            category.append(current.name)
            current = current.parent
        return "/".join(reversed(category))

    def read_text_files_in_folder(self) -> list[Document]:
        """Read all text files in a specified folder and return a list of Documents with hierarchical categories."""
        if self.folder_path is None:
            msg = "Folder path must be specified."
            raise ValueError(msg)

        folder = Path(self.folder_path)
        if not folder.exists() or not folder.is_dir():
            msg: str = f"The folder {self.folder_path} does not exist or is not a directory."
            raise ValueError(msg)

        documents: list[Document] = []
        for text_file in folder.glob(pattern="**/*.txt"):
            content: str = self.read_text_file(file_path=text_file)
            chunks: list[str] = self._split_text(text=content, separators=self._separators)
            category: str = self._get_category(path=text_file)
            parent_doc = Document(page_content=content, metadata={"file_path": text_file, "category": category})
            documents.extend(
                Document(
                    page_content=chunk,
                    metadata={"parent_id": parent_doc.id_, "file_path": text_file, "category": category},
                )
                for chunk in chunks
            )
        return documents
