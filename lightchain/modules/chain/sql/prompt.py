from __future__ import annotations  # noqa: D100

_sqlite_query_prompt = """You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".
Given the following examples of the SQL tables: {create_statements}\n\n. your job is to write queries given a user’s request. Try to output the query for human that do not know about databases functionality.
If you can provide an answer base on the SQL tables without query you should output the answer as free guidance, otherwise Please provide only the query as: Use the following format:

SQLQuery: SQL Query to run.

If there no possible query, please explain the reason why.
"""  # noqa: E501


def sqlite_query_prompt(top_k: int, create_statements: str) -> str:  # noqa: D417
    """Generate a SQLite prompt string for querying the database based on a user's input question.

    Args:
    ----
    top_k (int): The maximum number of results to retrieve from the database for non-specific quantity queries.

    Returns:
    -------
    str: A formatted SQLite prompt string containing the user's question, table information, and query limitations.

    """  # noqa: E501
    return _sqlite_query_prompt.format(top_k=top_k, create_statements=create_statements)


_sqlite_translation_prompt = """You are a language model trained to help users understand database query results without any prior knowledge of databases. Given the SQL query results, your task is to explain the information in a clear and concise manner, as if explaining to someone who has no understanding of how databases work.

Given:
1. SQL Query: The specific query that was run.
2. Query Results: The output of the SQL query formatted in a readable way (e.g., rows and columns or a summarized form).

Your job is to:
- Analyze the query results.
- Provide a simple explanation of what the data shows, focusing on the key information that answers the user's initial question.
- Use simple language and avoid database-specific terms as much as possible.

Example:
SQL Query: SELECT "name", "grade" FROM students WHERE "grade" > 90;
Query Results: [["John Doe", 95], ["Jane Smith", 98]]

Explanation:
Two students scored above 90 in their grades. John Doe scored 95, and Jane Smith scored 98.

Please transform the following SQL query results into a plain language explanation. 
"""  # noqa: E501


def sql_translation_prompt() -> str:
    """Generates a prompt for a language model to help users understand SQL query results without prior database knowledge.

    This function returns a prompt instructing the language model on how to:
    1. Receive SQL queries and their results.
    2. Analyze these results.
    3. Provide a simple, non-technical explanation of what the data represents.

    The prompt includes detailed instructions on how to convert SQL query results into plain language explanations,
    aiming to make the data understandable for users unfamiliar with database concepts.

    Returns
    -------
    str: A detailed prompt for explaining SQL query results in plain language.

    """  # noqa: D401, E501
    return _sqlite_translation_prompt
