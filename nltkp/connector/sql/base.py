"""The module contains the SQLDatabase class for interacting with SQL databases using SQLAlchemy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import MetaData, Table, create_engine, inspect, text

from nltkp.utils import LOGGER

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy import TextClause
    from sqlalchemy.engine import CursorResult, Engine, Result, Row
    from sqlalchemy.engine.reflection import Inspector
    from sqlalchemy.sql import Executable


class SQLDatabase:
    """Wrapper class for SQLAlchemy interactions with a database."""

    def __init__(self: SQLDatabase, engine: Engine) -> None:
        """Initialize the SQLDatabase instance with a connection engine.

        Args:
        ----
            engine (Engine): SQLAlchemy Engine object for database connections.

        """
        self.engine: Engine = engine
        self.inspector: Inspector = inspect(subject=self.engine)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        self.all_tables = set(self.inspector.get_table_names())

    @classmethod
    def create_engine(cls: type[SQLDatabase], database_uri: str) -> Engine:
        """Create an SQLAlchemy engine from a database URI.

        Args:
        ----
            database_uri (str): Database connection URI.

        Returns:
        -------
            Engine: A new SQLAlchemy Engine instance.

        """
        return create_engine(url=database_uri)

    @classmethod
    def from_uri(cls: type[SQLDatabase], database_uri: str) -> SQLDatabase:
        """Construct a SQLDatabase instance from a database URI.

        Args:
        ----
            database_uri (str): Database connection URI.

        Returns:
        -------
            SQLDatabase: A new instance of SQLDatabase.

        """
        engine: Engine = cls.create_engine(database_uri=database_uri)
        return cls(engine)

    @property
    def dialect(self: SQLDatabase) -> str:
        """Return the dialect used by the database engine.

        Returns
        -------
            str: The name of the database dialect.

        """
        return self.engine.dialect.name

    def get_usable_table_names(self: SQLDatabase) -> list[str]:
        """Retrieve a list of all table names in the database.

        Returns
        -------
            list[str]: Sorted list of table names.

        """
        return sorted(self.all_tables)

    def get_table_create_statement(self: SQLDatabase, table: Table) -> str:
        """Generate the SQL CREATE TABLE statement for a given table.

        Args:
        ----
            table (Table): SQLAlchemy Table object.

        Returns:
        -------
            str: SQL CREATE TABLE statement.

        """
        columns: list[str] = [f'  "{col.name}" {str(object=col.type)}' for col in table.columns]
        primary_key: list[str] = [f'PRIMARY KEY ("{col.name}")' for col in table.primary_key.columns]
        constraints: list[str] = primary_key + [
            str(object=cons) for cons in table.constraints if not str(object=cons).startswith("PRIMARY KEY")
        ]
        columns_and_constraints: str = ",\n".join(columns + constraints)
        create_statement: str = f'CREATE TABLE "{table.name}" (\n{columns_and_constraints}\n);'
        return create_statement

    def get_create_statements(self: SQLDatabase) -> str:
        """Compile SQL CREATE TABLE statements for all tables in the database.

        Returns
        -------
            str: Concatenated SQL CREATE statements for each table.

        """
        return "\n".join(self.get_table_create_statement(table=table) for table in self.metadata.tables.values())

    def run(  # noqa: PLR0913
        self: SQLDatabase,
        command: str | Executable,
        fetch: Literal["all", "one", "cursor"] = "all",
        include_columns: bool = False,  # noqa: FBT001, FBT002
        *,
        parameters: dict[str, Any] | None = None,
        execution_options: dict[str, Any] | None = None,
    ) -> str | Sequence[dict[str, Any]] | Result[Any]:
        """Execute a SQL command or query and fetch results based on the specified mode.

        Args:
        ----
            command (str | Executable): The SQL command or query to execute.
            fetch (Literal["all", "one", "cursor"]): The fetching strategy for query results.
            include_columns (bool): If True, include column names in the result set.
            parameters (dict[str, Any] | None): Parameters for parameterized queries.
            execution_options (dict[str, Any] | None): Options to be passed to the execution context.

        Returns:
        -------
            str | Sequence[dict[str, Any]] | Result[Any]: The result of the executed command.

        """
        result: str | Sequence[dict[str, Any]] | Result[Any] = self._execute(
            command=command,
            fetch=fetch,
            parameters=parameters,
            execution_options=execution_options,
        )

        if fetch == "cursor":
            return result

        res: list[dict[str, Any] | tuple] = [
            {column: str(object=value)[:100] for column, value in r.items()} for r in result
        ]

        if not include_columns:
            res = [tuple(iterable=row.values()) for row in res]  # type: ignore  # noqa: PGH003

        if not res:
            return ""
        return str(object=res)  # Output the result list as a string

    def _execute(
        self: SQLDatabase,
        command: str | Executable | TextClause,
        fetch: Literal["all", "one", "cursor"] = "all",
        *,
        parameters: dict[str, Any] | None = None,
        execution_options: dict[str, Any] | None = None,
    ) -> Sequence[dict[str, Any]] | CursorResult:
        """Private method to execute SQL commands through the SQLAlchemy engine.

        Args:
        ----
            command (str | Executable | TextClause): The SQL command or Executable to run.
            fetch (Literal["all", "one", "cursor"]): Determines the fetching strategy.
            parameters (dict[str, Any] | None): SQL command parameters.
            execution_options (dict[str, Any] | None): SQLAlchemy execution options.

        Returns:
        -------
            Sequence[dict[str, Any]] | CursorResult: Result of the SQL command.

        """
        with self.engine.connect() as connection:
            stmt: TextClause | Executable = text(text=command) if isinstance(command, str) else command
            if execution_options:
                stmt = stmt.execution_options(**execution_options)

            cursor: CursorResult[Any] = connection.execute(
                statement=stmt,
                parameters=parameters or {},
            )

            if cursor.returns_rows:
                if fetch == "all":
                    result: list[dict[str, Any]] = [x._asdict() for x in cursor.fetchall()]
                elif fetch == "one":
                    first_result: Row[Any] | None = cursor.fetchone()
                    result = [] if first_result is None else [first_result._asdict()]
                elif fetch == "cursor":
                    return cursor
                else:
                    msg = "Fetch parameter must be either 'one', 'all', or 'cursor'"
                    raise ValueError(msg)
                return result
        return []


if __name__ == "__main__":
    # Usage example
    db_uri = "sqlite:///resources/database/Chinook.db"  # Replace with your SQLite database URI
    db: SQLDatabase = SQLDatabase.from_uri(database_uri=db_uri)
    usable_tables: list[str] = db.get_usable_table_names()
    LOGGER.info(msg=f"The tables at the db: {usable_tables}")

    # Example to run a SQL command
    result: str | Sequence[dict[str, Any]] | Result[Any] = db.run(
        command="SELECT * FROM Artist",
        fetch="all",
        include_columns=True,
    )
    LOGGER.info(msg=f"Result from query of Artist table: {result}")
