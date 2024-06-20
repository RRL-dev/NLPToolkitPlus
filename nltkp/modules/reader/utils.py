"""Reader utils functionality."""
from __future__ import annotations

from hashlib import sha256
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Class for storing content and metadata of a document, each document has a unique ID."""

    page_content: str = Field(default=..., description="The textual content of the document.")
    metadata: dict[str, Any] = Field(
        default=...,
        description="Metadata associated with the document.",
    )
    id_: str = Field(
        default_factory=lambda: str(object=uuid4()),
        alias="doc_id",
        description="A unique identifier for the document.",
    )

    class Config:
        """Configuration settings for Pydantic models within the Document class.

        Attributes
        ----------
            allow_mutation: Allows fields to be mutable, default is True.
            validate_assignment: Ensures fields are validated upon assignment, default is True.

        """

        allow_mutation = True
        validate_assignment = True

    @property
    def hash(self: Document) -> str:
        """Generate a hash for the document based on its content and metadata.

        Returns
        -------
            A SHA-256 hash of the document's content and metadata.

        """
        doc_identity: str = f"{self.page_content}{self.metadata}"
        return sha256(string=doc_identity.encode(encoding="utf-8")).hexdigest()
