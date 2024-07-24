"""Module for creating hierarchical document embeddings.

This module provides functionality to load models dynamically,
process documents to generate embeddings, and save the embeddings
and document contents to designated directories based on hierarchical structure.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copy
from typing import TYPE_CHECKING

from joblib import dump
from numpy import float32, ndarray
from torch import Tensor, save
from tqdm import tqdm

from nltkp.cfg import EMBEDDINGS_CFG
from nltkp.models import BaseSentenceModel
from nltkp.modules import HierarchicalTextSplitter, TextSplitterConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from nltkp.modules.reader.utils import Document


class HierarchicalDocumentsEmbeddings(BaseSentenceModel):
    """A class to handle the embedding of documents using specified models.

    This class manages the loading of neural network models, processing of documents
    to generate embeddings, and storage of the results in a hierarchical directory format.
    """

    def __init__(
        self: HierarchicalDocumentsEmbeddings,
        model_name: str,
        folder_path: str,
        pooling_modes: dict,
    ) -> None:
        """Initialize the HierarchicalDocumentsEmbeddings instance.

        Args:
        ----
            model_name (str): The name of the model to be used for generating embeddings.
            folder_path (str): The path to the folder containing documents to be processed.
            pooling_modes (dict): The pooling modes to be used for the model.

        """
        super().__init__(model_name=model_name, pooling_modes=pooling_modes)
        self.folder_path = Path(folder_path)
        config = TextSplitterConfig(folder_path=str(object=self.folder_path), model_name=model_name)

        self.documents: list[Document] = HierarchicalTextSplitter(
            config=config,
        ).read_text_files_in_folder()
        self.processed_files: set[str] = set()  # Set to track processed files

    @staticmethod
    def create_dirs(category: str, file_name: str) -> tuple[Path, Path]:
        """Ensure the directories for embeddings and texts exist.

        Args:
        ----
            category (str): The category of the document.
            file_name (str): The name of the file.

        Returns:
        -------
            tuple[Path, Path]: Paths to the embeddings and chunks directories.

        """
        category = "" if not Path(category).parent.stem else Path(category).stem

        base_dir: Path = Path(EMBEDDINGS_CFG.path)
        embeddings_dir: Path = base_dir / "embeddings" / category / file_name
        chunks_dir: Path = base_dir / "chunks" / category / file_name
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        return embeddings_dir, chunks_dir

    def save_embeddings(self: HierarchicalDocumentsEmbeddings) -> None:
        """Embed all documents and save their embeddings and content based on their hash."""
        progress_bar = tqdm(
            iterable=self.documents,
            desc="Processing Documents",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            colour="green",
        )

        for document in progress_bar:
            if document.page_content == "\n\n":
                continue
            texts: str = document.page_content.replace("\n", "")
            embeddings: NDArray[float32] | Tensor = self.encode(sentences=texts)
            _hash: str = document.hash  # Ensure your Document class has a hash property
            category: str = document.metadata["category"]
            file_name: str = Path(document.metadata["file_path"]).stem

            chunks_dir: Path
            embeddings_dir: Path
            # Create directories for embeddings and chunks
            embeddings_dir, chunks_dir = self.create_dirs(category=category, file_name=file_name)

            # Determine the file path for saving embeddings
            embedding_file_path: Path = embeddings_dir / f"{_hash}"

            if isinstance(embeddings, Tensor):
                # Save as a .pt file if embeddings is a Tensor
                with embedding_file_path.with_suffix(suffix=".pt").open(mode="wb") as f:
                    save(obj=embeddings, f=f)
            elif isinstance(embeddings, ndarray):
                # Save as a .joblib file if embeddings is an ndarray
                with embedding_file_path.with_suffix(suffix=".joblib").open(mode="wb") as f:
                    dump(value=embeddings, filename=f)

            # Save the document's text content
            text_file_path: Path = chunks_dir / f"{_hash}.txt"
            with text_file_path.open(mode="w", encoding="utf-8") as f:
                f.write(document.page_content)

            # Save the original file with a unique name in chunks_dir if not already saved
            file_path: str = str(object=document.metadata["file_path"])
            if file_path not in self.processed_files:
                original_file_path: Path = chunks_dir / f"original_{_hash}{Path(document.metadata['file_path']).suffix}"
                copy(src=document.metadata["file_path"], dst=original_file_path)
                self.processed_files.add(file_path)


# Example usage:
if __name__ == "__main__":
    embeddings = HierarchicalDocumentsEmbeddings(
        model_name=EMBEDDINGS_CFG.model_name,
        folder_path=EMBEDDINGS_CFG.raw_data,
        pooling_modes=EMBEDDINGS_CFG.pooling_modes,
    )
    embeddings.save_embeddings()
