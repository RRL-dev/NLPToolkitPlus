"""The module contains the definition of the SQLQueryChain class and its methods."""

from __future__ import annotations

from collections.abc import Sequence
from os import environ
from typing import TYPE_CHECKING, Any, Literal

from openai import OpenAI
from sqlalchemy.exc import (
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
)

from lightchain.connector import SQLDatabase, get_table_info_from_sql_database
from lightchain.utils import LOGGER

from .prompt import sqlite_query_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai.types.chat.chat_completion import (
        ChatCompletion,
        Choice,
    )
    from sqlalchemy.engine import Result


class SQLQueryChain:
    """Handle the conversion of natural language queries to SQL queries and execute them on a SQL database.

    Attributes
    ----------
        db (SQLDatabase): A connection to a SQL database.
        client (OpenAI): An OpenAI client initialized with an API key from the environment.
        top_k (int): The maximum number of results to retrieve from queries without a specified limit.

    """  # noqa: E501

    def __init__(
        self: SQLQueryChain,
        db: SQLDatabase,
        top_k: int = 5,
    ) -> None:
        """Initialize the SQLQueryChain class with a database connection and top_k value for query limits.

        Args
        ----
            db (SQLDatabase): A connection to a SQL database.
            top_k (int): The maximum number of results to retrieve from queries without a specified limit.

        """  # noqa: E501
        self.db: SQLDatabase = db
        self.client: OpenAI = OpenAI(api_key=environ.get("OPENAI_API_KEY"))
        self.top_k: int = top_k

    def interpret_nl_to_sql(self: SQLQueryChain, nl_query: str) -> str:
        """Convert a natural language query into an SQL query using a large language model.

        Args:
        ----
            nl_query (str): The natural language query to interpret.

        Returns:
        -------
            str: The SQL query derived from the natural language input.

        """
        create_statements: str = get_table_info_from_sql_database(
            engine=self.db.engine,
            schema=None,  # Specify if your tables are in a specific schema
            table_names=None,  # List specific tables if needed, or None to get all
            view_support=False,  # Whether to include views in the table information
            custom_table_info=None,  # Custom information or formatting for table descriptions
            max_string_length=300,  # Maximum length for string fields in the database
            indexes_in_table_info=False,  # Include index information in table descriptions
            sample_rows_in_table_info=3,  # Include sample rows in the table descriptions
        )
        system_prompt: str = sqlite_query_prompt(
            top_k=self.top_k,
            create_statements=create_statements,
        )

        response: ChatCompletion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": nl_query},
            ],
            temperature=0.0,
        )
        return self.extract_sql_from_response(response=response)

    def extract_sql_from_response(self: SQLQueryChain, response: ChatCompletion) -> str:
        """Extract the SQL query from the chat model's response.

        Args:
        ----
            response (ChatCompletion): The response from the chat model.

        Returns:
        -------
            str: The SQL query, if found; otherwise a default error message.

        """
        choices: list[Choice] = response.choices
        response_text: str | None = choices[0].message.content if choices else None

        if response_text is None:
            return "Response content is None"
        try:
            start_index: int = response_text.index("SQLQuery:") + len("SQLQuery:")
            return response_text[start_index:].strip()
        except ValueError:
            return response_text

    def query(
        self: SQLQueryChain,
        nl_query: str,
        fetch: Literal["all", "one", "cursor"] = "all",
    ) -> str | Sequence[dict[str, Any]] | Result[Any]:
        """Execute a query on the database using an SQL statement derived from a natural language description.

        Args:
        ----
            nl_query (str): The natural language description of the query.
            fetch (Literal["all", "one", "cursor"]): How the results should be returned. Defaults to "all".

        Returns:
        -------
            Union[str, Sequence[Dict[str, Any]], Result[Any]]: The result of the query, which might be a single value, a list of dictionaries, or a raw SQL result set.

        """  # noqa: E501
        sql_query: str = self.interpret_nl_to_sql(nl_query=nl_query)
        LOGGER.info(msg=f"The answer for the nl_query: {sql_query}")

        try:
            result: str | Sequence[dict[str, Any]] | Result[Any] = self.db.run(
                command=sql_query, fetch=fetch, include_columns=True
            )
            return result

        except IntegrityError as e:
            LOGGER.error(msg=f"Integrity error during SQL execution: {e}")
            return sql_query

        except OperationalError as e:
            LOGGER.error(msg=f"Operational error with the database: {e}")
            return sql_query

        except SQLAlchemyError as e:
            LOGGER.error(msg=f"General SQLAlchemy error: {e}")
            return sql_query


if __name__ == "__main__":
    db: SQLDatabase = SQLDatabase.from_uri(database_uri="sqlite:///resources/database/data.db")

    chain = SQLQueryChain(db=db)
    nl_query = "how many people are below the age 55?"
    chain.query(nl_query=nl_query)
