"""The module contains the definition of the SqlAgent class and its methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy.exc import (
    OperationalError,
    SQLAlchemyError,
)

from lightchain.connector import SQLDatabase
from lightchain.modules.chain import (
    SQLExplanationChain,
    SQLQueryChain,
)
from lightchain.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine import Result


class SqlAgent(SQLQueryChain, SQLExplanationChain):
    """Handles both natural language interpretation of SQL queries and translation of SQL results into plain language explanations.

    This class inherits from SQLQueryChain for converting natural language to SQL
    and from SQLExplanationChain for translating SQL results into human-friendly explanations.

    Attributes
    ----------
        db (SQLDatabase): A connection to a SQL database.
        client (OpenAI): An OpenAI client initialized with an API key from the environment.
        top_k (int): The maximum number of results to consider when retrieving query results.

    """  # noqa: E501

    def __init__(
        self: SqlAgent,
        db: SQLDatabase,
        top_k: int = 5,
    ) -> None:
        """Initialize the SqlAgent class with a database connection and top_k value for query limits.

        Args:
        ----
            db (SQLDatabase): A connection to a SQL database.
            top_k (int): The maximum number of results to retrieve from queries without a specified limit.

        """  # noqa: E501
        SQLQueryChain.__init__(self=self, db=db, top_k=top_k)
        SQLExplanationChain.__init__(self=self)

    def interpret_and_explain(
        self: SqlAgent,
        nl_query: str,
        fetch: Literal["all", "one", "cursor"] = "all",
    ) -> str:
        """Convert a natural language query into an SQL query, execute it, and provide a plain language explanation of the results.

        Args:
        ----
            nl_query (str): The natural language description of the query.
            fetch (Literal["all", "one", "cursor"]): The fetch mode for SQL execution results.

        Returns:
        -------
            str: The plain language explanation of the SQL query results.

        """  # noqa: E501
        # Interpret NL to SQL and execute
        sql_query: str = self.interpret_nl_to_sql(nl_query=nl_query)

        try:
            result: str | Sequence[dict[str, Any]] | Result[Any] = self.db.run(
                command=sql_query,
                fetch=fetch,
                include_columns=True,
            )

            # Format results for explanation
            sql_results = str(object=result)
            explanation: str = self.explain_sql_results(
                nl_query=nl_query,
                sql_results=sql_results,
            )
        except (
            OperationalError,
            SQLAlchemyError,
        ) as e:
            LOGGER.error(msg=f"SQL error occurred: {e}")
            return sql_query
        else:
            return explanation

    def __str__(self: SqlAgent) -> str:
        """Provide a string representation of the SqlAgent instance, highlighting key configurations.

        Returns
        -------
            str: A descriptive string of the SqlAgent instance.

        """
        return f"SqlAgent connected to {self.db.__class__.__name__} with a top_k of {self.top_k}."


if __name__ == "__main__":
    DB: SQLDatabase = SQLDatabase.from_uri(database_uri="sqlite:///resources/database/data.db")
    agent = SqlAgent(db=DB)
    NL_QUERY = "which type of data we have at our data?"
    EXPLANATION: str = agent.interpret_and_explain(nl_query=NL_QUERY)
    LOGGER.info(msg=EXPLANATION)
