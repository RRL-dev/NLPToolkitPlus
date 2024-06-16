"""The module contains the SQLExplanationChain class and its methods."""

from __future__ import annotations

from os import environ
from typing import TYPE_CHECKING

from openai import OpenAI

from nltkp.utils import LOGGER

from .prompt import sql_translation_prompt

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import (
        ChatCompletion,
        Choice,
    )


class SQLExplanationChain:
    """Handle explanations of SQL query results using a language model."""

    def __init__(self: SQLExplanationChain) -> None:
        """Initialize an instance of SQLExplanationChain with an OpenAI client."""
        self.client: OpenAI = OpenAI(api_key=environ.get("OPENAI_API_KEY"))

    def explain_sql_results(self: SQLExplanationChain, nl_query: str, sql_results: str) -> str:
        """Generate an explanation of the SQL query results using the language model.

        Args:
        ----
            nl_query (str): The natural language representation of the SQL query.
            sql_results (str): The results of the SQL query formatted as a string.

        Returns:
        -------
            str: The natural language explanation of the SQL query results.

        """
        system_prompt: str = sql_translation_prompt()
        response: ChatCompletion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"SQL Query: {nl_query}\nQuery Results: {sql_results}",
                },
            ],
            temperature=0.0,
        )

        response_text: str = self.extract_sql_from_response(response=response)
        return response_text

    def extract_sql_from_response(self: SQLExplanationChain, response: ChatCompletion) -> str:
        """Extract the explanation from the chat model's response.

        Args:
        ----
            response (ChatCompletion): The response from the chat model.

        Returns:
        -------
            str: The extracted explanation for the SQL query results.

        """
        choices: list[Choice] = response.choices
        response_text: str | None = choices[0].message.content if choices else None

        if response_text is None:
            return "Response content is None"
        return response_text


if __name__ == "__main__":
    # Example usage
    agent = SQLExplanationChain()
    nl_query = 'SELECT "name", "grade" FROM students WHERE "grade" > 90;'
    sql_results = "[['John Doe', 95], ['Jane Smith', 98]]"
    explanation: str = agent.explain_sql_results(nl_query=nl_query, sql_results=sql_results)
    LOGGER.info(msg=explanation)
