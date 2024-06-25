"""Module for managing history messages in conversations.

Classes:
- HistoryMessage: Manages conversations and supports updating existing messages based on specific roles within the chat.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from nltkp.utils import LOGGER

from .handler import HandlerMessage

if TYPE_CHECKING:
    from collections.abc import Callable


class HistoryMessage(HandlerMessage):
    """Extended model for managing a conversation with additional functionalities for updating messages."""

    def add_or_update_system_message_with_context(self: HistoryMessage, context: str) -> None:
        """Add or update a system message formatted with dynamic context information."""
        content: str = (
            "You are a friendly chatbot, answer. Use the following context if it's available "
            "but don't quote from source, phrase in your own words. If you do not know, "
            f"answer that you don't have knowledge on that. Here is the context: {context}"
        )

        # Check for existing system message to update
        for message in reversed(self.messages):
            if message.role == "system":
                message.content = content
                return  # Update found and applied, exit function

        # If no existing system message, add a new one
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
        msg: str = f"No message found with role {role}."
        raise ValueError(msg)

    @classmethod
    def get_update_system_message(cls: type[HistoryMessage], instance: HistoryMessage) -> Callable[[str], None]:
        """Get a 'partial' for updating system messages."""
        return partial(instance.update_system_message)

    @classmethod
    def get_update_user_message(cls: type[HistoryMessage], instance: HistoryMessage) -> Callable[[str], None]:
        """Get a 'partial' for updating user messages."""
        return partial(instance.update_user_message)

    def __str__(self: HistoryMessage) -> str:
        """Provide a string representation of the entire conversation, formatted for readability."""
        conversation: list[str] = []
        for message in self.messages:
            prefix: Literal["System: ", "User: "] = "System: " if message.role == "system" else "User: "
            conversation.append(prefix + message.content)
        return "\n".join(conversation)


if __name__ == "__main__":
    # Example usage, showing how to dynamically update messages within a conversation
    try:
        chat = HistoryMessage()
        chat.add_message(role="system", content="Initial system message.")
        chat.add_message(role="user", content="Initial user query.")

        # Using the partial functions to update messages
        system_updater = HistoryMessage.get_update_system_message(instance=chat)
        user_updater = HistoryMessage.get_update_user_message(instance=chat)
        system_updater("Revised system response.")
        user_updater("Revised user query.")

        LOGGER.info(msg=chat)
    except (ValidationError, ValueError) as e:
        LOGGER.error("Error during message update: %s", e)
