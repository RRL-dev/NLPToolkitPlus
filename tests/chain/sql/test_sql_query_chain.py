r"""The module contains integration tests for the SQLQueryChain class, which interacts with a real,
SQLite database to execute SQL queries derived from natural language inputs.
"""  # noqa: D205

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

from superchain.connector import SQLDatabase
from superchain.modules.chain import SQLQueryChain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy import Result


class TestSQLQueryChainIntegration(unittest.TestCase):
    """Integration tests for the SQLQueryChain class."""

    def setUp(self: TestSQLQueryChainIntegration) -> None:
        """Prepare the test environment with a real SQLite database."""
        # Initialize the database
        self.db: SQLDatabase = SQLDatabase.from_uri(
            database_uri="sqlite:///resources/database/data.db"
        )
        self.chain: SQLQueryChain = SQLQueryChain(db=self.db)

    def test_query_diabetes_count(self: TestSQLQueryChainIntegration) -> None:
        """Test to see how many patients are diagnosed with diabetes."""
        nl_query: str = "how many patients has diabetes?, the meant Outcome equal to 1."
        result: str | Sequence[dict[str, Any]] | Result[Any] = self.chain.query(
            nl_query=nl_query, fetch="all"
        )
        expected_result: str = "[{'Number of Patients with Diabetes': '268'}]"
        self.assertEqual(first=result, second=expected_result)  # noqa: PT009

    def test_query_average_age(self: TestSQLQueryChainIntegration) -> None:
        """Test to calculate the average age of the patients."""
        nl_query: str = "what is the average age?"
        result: str | Sequence[dict[str, Any]] | Result[Any] = self.chain.query(
            nl_query=nl_query,
            fetch="all",
        )
        expected_result: str = "[{'average_age': '33.240885416666664'}]"
        self.assertEqual(first=result, second=expected_result)  # noqa: PT009

    def test_query_age_below(self: TestSQLQueryChainIntegration) -> None:
        """Test to calculate the number of people below the age of 55."""
        nl_query: str = "how many people are below the age 55?"
        result: str | Sequence[dict[str, Any]] | Result[Any] = self.chain.query(
            nl_query=nl_query,
            fetch="all",
        )
        expected_result: str = "[{'Number of People Below Age 55': '714'}]"
        self.assertEqual(first=result, second=expected_result)  # noqa: PT009


if __name__ == "__main__":
    unittest.main()
