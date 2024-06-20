"""The module provides classes and utilities for handling conversation messages in a chat system.

The primary class defined in this module is `Message`, which represents a single message
in a conversation, distinguishing between system and user roles. The module includes
validation to ensure that messages adhere to expected formats and roles.

Classes:
    - Message: Defines a structure for storing message content along with the sender's role.

Usage:
    from message_handling import Message

    try:
        msg = Message(role='user', content='Hello, world!')
        print(msg)
    except ValidationError as e:
        print("Error:", e)

This module is part of a larger chatbot framework that aims to provide robust tools for
handling and processing conversation data.

Dependencies:
    - Pydantic: Used for data validation and settings management via BaseModel.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from nltkp.utils import LOGGER


class Message(BaseModel):
    """Class representing a single message in a conversation, specifying the role of the sender.

    Attributes
    ----------
        role (str): The role of the message sender, either 'system' or 'user'.
        content (str): The textual content of the message.

    """

    role: str = Field(default=...)
    content: str = Field(default=...)

    @field_validator("role")
    def validate_role(cls: Message, v: str) -> str:  # noqa: N805
        """Validate the role field to ensure it only contains 'system' or 'user'.

        Args:
        ----
            v (str): The role value to validate.

        Returns:
        -------
            str: The validated role value if it is valid.

        Raises:
        ------
            ValueError: If the role value is not 'system' or 'user'.

        """
        if v not in ["system", "user"]:
            msg = "Role must be either 'system' or 'user'"
            raise ValueError(msg)
        return v


class ConversationMessage(BaseModel):
    """A model for managing a conversation consisting of a sequence of messages.

    Attributes
    ----------
        messages (list[Message]): A list of messages that form the conversation.

    """

    messages: list[Message] = []

    def add_message(self: ConversationMessage, role: str, content: str) -> None:
        """Add a new message to the conversation.

        Args:
        ----
            role (str): The role of the speaker ('system' or 'user').
            content (str): The content of the message.

        """
        new_message = Message(role=role, content=content)
        self.messages.append(new_message)

    def get_last_user_message(self: ConversationMessage) -> str:
        """Retrieve the last message from the user in the conversation.

        Returns
        -------
            str: The content of the last user message, if any.

        """
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""

    def __str__(self: ConversationMessage) -> str:
        """Provide a string representation of the entire conversation, formatted for readability.

        Returns
        -------
            str: A formatted string of the conversation.

        """
        conversation: list[str] = []
        for message in self.messages:
            prefix: Literal["System: ", "User: "] = (
                "System: " if message.role == "system" else "User: "
            )
            conversation.append(prefix + message.content)
        return "\n".join(conversation)


if __name__ == "__main__":
    # Example usage, assuming all other aspects of your setup are correct.
    try:
        prompt = ConversationMessage()
        prompt.add_message(
            role="system",
            content="You are a friendly chatbot who always responds in the style of a pirate.",
        )
        prompt.add_message(
            role="user",
            content="How many helicopters can a human eat in one sitting?",
        )
        LOGGER.info(msg=prompt)
    except ValidationError as e:
        LOGGER.error("Error:", e)
