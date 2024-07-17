"""Module for managing history messages in conversations with few-shot examples.

Classes:
- FewShotHistoryMessage: Extended class to handle few-shot examples using Chain of Thought.
"""

from __future__ import annotations

from typing import Any

from nltkp.modules.rag.retriever import SearchInput
from nltkp.utils import LOGGER

from .history import HistoryMessage, Message


class FewShotHistoryMessage(HistoryMessage):
    """Extended HistoryMessage class to handle few-shot examples using Chain of Thought."""

    def __init__(self, examples: dict[str, list[dict[str, str]]]) -> None:
        """Initialize the FewShotHistoryMessage instance."""
        super().__init__()
        self.messages: list[Message] = []
        self.examples: list[dict[str, str]] = self.load_examples(examples=examples)

    def load_examples(self, examples: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
        """Load few-shot examples from a provided dictionary."""
        return examples["few_shot_examples"]

    def create_few_shot_prompt(self) -> str:
        """Create few-shot prompt messages as a single text string."""
        LOGGER.info("Creating few-shot prompt with examples.")
        cot_texts: list[str] = []
        for example in self.examples:
            question: str = example["question"]
            answer: str = example["answer"].strip()
            cot_texts.append(f"Question: {question}\n{answer}")
        return "\n\n".join(cot_texts)

    def add_or_update_system_message_with_context(self, context: str) -> None:
        """Add or update a system message formatted with dynamic context information."""
        cot_prompt = self.create_few_shot_prompt()
        content: str = (
            f"{cot_prompt}\n\n"
            "You are a specialized retail chatbot. You can only use the provided context to answer questions related to retail. "  # noqa: E501
            "If the question is not related to retail or does not match the context provided, clearly state: 'I do not answer questions outside of the retail context.' "  # noqa: E501
            f"Do not quote directly from the source; always paraphrase in your own words. Here is the context: {context}"  # noqa: E501
        )

        for message in reversed(self.messages):
            if message.role == "system":
                message.content = content
                return

        self.add_message(role="system", content=content)

    def process_message(
        self,
        inputs: dict[str, Any] | str,
    ) -> SearchInput | list[dict[str, str]]:
        """Process the incoming data and either update the conversation history or prepare it for a search operation."""
        if isinstance(inputs, str):
            self.add_or_update_system_message_with_context(context=inputs)
            return self.format()

        search_input = SearchInput(top_k=0, sentences="")
        try:
            if isinstance(inputs, dict):
                query: str = inputs.get("query", "")
                top_k: int = inputs.get("top_k", 5)
                if query:
                    self.add_message(role="user", content=query)
                    search_input = SearchInput(top_k=top_k, sentences=query)
                if "context" in inputs:
                    self.add_or_update_system_message_with_context(context=inputs["context"])

        except KeyError as e:
            LOGGER.error("Key error: %s", e)
            msg: str = f"Key error: {e}"
            raise ValueError(msg) from e

        return search_input
