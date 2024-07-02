"""Module for managing history messages in conversations.

Classes:
- HistoryMessage: Manages conversations and supports updating existing messages based on specific roles within the chat.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from nltkp.factory import ChainedRunnable
from nltkp.modules.rag.retriever import SearchInput
from nltkp.utils import LOGGER

from .handler import Message


class HistoryMessage(
    ChainedRunnable[dict[str, Any] | str, SearchInput | list[dict[str, str]]],
):
    """Extended model for managing a conversation with additional functionalities for updating messages."""

    def __init__(self: HistoryMessage) -> None:
        """Initialize the HistoryMessage instance."""
        super().__init__(func=self.process_message)
        self.messages: list[Message] = []

    def add_message(self: HistoryMessage, role: str, content: str) -> None:
        """Add a new message to the conversation."""
        new_message = Message(role=role, content=content)
        self.messages.append(new_message)

    def process_message(
        self: HistoryMessage,
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

    def add_or_update_system_message_with_context(self: HistoryMessage, context: str) -> None:
        """Add or update a system message formatted with dynamic context information."""
        content: str = (
            "You are a friendly chatbot user answer. Use the following context if it's available "
            "but don't quote from source, phrase in your own words. If you do not know, "
            f"answer that you don't have knowledge on that. Here is the context: {context}"
        )

        for message in reversed(self.messages):
            if message.role == "system":
                message.content = content
                return

        self.add_message(role="system", content=content)

    def update_system_message(self: HistoryMessage, content: str) -> None:
        """Update the last system message."""
        self._update_role_message(role="system", content=content)

    def update_user_message(self: HistoryMessage, content: str) -> None:
        """Update the last user message."""
        self._update_role_message(role="user", content=content)

    def _update_role_message(self: HistoryMessage, role: str, content: str) -> None:
        """Update the last message with the given role."""
        for message in reversed(self.messages):
            if message.role == role:
                message.content = content
                return
        msg = f"No message found with role {role}."
        raise ValueError(msg)

    def format(self: HistoryMessage) -> list[dict[str, str]]:
        """Convert the messages to a list of dictionaries, ensuring the system message is first."""
        formatted_messages: list[dict[str, str]] = [
            {"role": message.role, "content": message.content} for message in self.messages
        ]

        system_message: dict[str, str] | None = next(
            (msg for msg in formatted_messages if msg["role"] == "system"),
            None,
        )
        if system_message:
            formatted_messages.remove(system_message)
            formatted_messages.insert(0, system_message)

        return formatted_messages

    def __str__(self: HistoryMessage) -> str:
        """Provide a string representation of the entire conversation, formatted for readability."""
        conversation = [
            f"{('System:' if message.role == 'system' else 'User:')} {message.content}" for message in self.messages
        ]
        return "\n".join(conversation)


if __name__ == "__main__":
    # Example usage, showing how to dynamically update messages within a conversation
    try:
        chat = HistoryMessage()
        chat.add_message(role="system", content="Initial system message.")
        chat.add_message(role="user", content="Initial user query.")

        LOGGER.info(msg=chat)

        # Print formatted messages
        formatted_messages: list[dict[str, str]] = chat.format()
        LOGGER.info("Formatted messages: %s", formatted_messages)

    except (ValidationError, ValueError) as e:
        LOGGER.error("Error during message update: %s", e)
