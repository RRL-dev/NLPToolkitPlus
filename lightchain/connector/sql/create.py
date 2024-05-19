"""The module contains the FilesToSQL class for interacting with datasets and saving them to an SQL database."""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError

from superchain.utils import LOGGER
from superchain.utils.io import FilesReader

if TYPE_CHECKING:
    from types import SimpleNamespace

    from pandas.core.frame import DataFrame
    from sqlalchemy.engine.interfaces import ReflectedColumn
    from sqlalchemy.engine.reflection import Inspector


class FilesToSQL(FilesReader):
    """Class for handling datasets and saving them to an SQL database."""

    def __init__(self: FilesToSQL, config: SimpleNamespace) -> None:
        """Initialize FilesToSQL with the dataset configuration and SQL database URL.

        Args:
        ----
            config (SimpleNamespace): Configuration specifying the dataset.

        """
        super().__init__(config=config)
        db_path: str = config.dataset["database"]["path"]
        self.db_url: str = f"sqlite:///{db_path}"
        self.engine: Engine | None = self._create_engine()

    def _create_engine(self: FilesToSQL) -> Engine | None:
        """Create and validate the SQLAlchemy engine.

        Returns
        -------
            Optional[Engine]: The SQLAlchemy engine if successfully created, otherwise None.

        """
        try:
            engine: Engine = create_engine(url=self.db_url)
            LOGGER.info(msg="Database connection validated.")
            return engine  # noqa: TRY300
        except SQLAlchemyError as e:
            LOGGER.error(msg=f"Database connection failed: {e}")
            return None

    def save_to_sql(
        self: FilesToSQL, table_name: str, df: DataFrame, if_exists: str = "replace",
    ) -> None:
        """Save a DataFrame to an SQL table.

        Args:
        ----
            table_name (str): The name of the table to save the DataFrame to.
            df (DataFrame): The DataFrame to save.
            if_exists (str, optional): What to do if the table already exists. Default is 'replace'.

        """
        if self.engine:
            try:
                df.to_sql(name=table_name, con=self.engine, if_exists=if_exists, index=False)  # type: ignore  # noqa: PGH003
                LOGGER.info(msg=f"Data saved to table {table_name}.")
            except SQLAlchemyError as e:
                LOGGER.error(msg=f"Error saving to table {table_name}: {e}")
        else:
            LOGGER.error(msg="No valid database engine available.")

    def cache_to_sql(self: FilesToSQL, if_exists: str = "replace") -> None:
        """Save all cached DataFrames to their respective tables in the SQL database.

        Args:
        ----
            if_exists (str, optional): What to do if the table already exists. Default is 'replace'.

        """
        for table_name, df in self.cache.items():
            self.save_to_sql(table_name=table_name, df=df, if_exists=if_exists)
            self.inspect_table(table_name=table_name)

    def dispose_engine(self: FilesToSQL) -> None:
        """Dispose of the SQLAlchemy engine."""
        if self.engine:
            self.engine.dispose()
            LOGGER.info(msg="Database engine disposed.")

    def inspect_table(self: FilesToSQL, table_name: str) -> None:
        """Inspect the structure of a table in the SQL database.

        Args:
        ----
            table_name (str): The name of the table to inspect.

        """
        if not self.engine:
            LOGGER.error(msg="No valid database engine available for inspection.")
            return

        inspector: Inspector = inspect(subject=self.engine)
        if table_name not in inspector.get_table_names():
            LOGGER.warning(msg=f"Table {table_name} does not exist in the database.")
            return

        columns: list[ReflectedColumn] = inspector.get_columns(table_name=table_name)
        LOGGER.info(msg=f"Table {table_name} columns:")
        for column in columns:
            LOGGER.info(msg=f"  {column['name']} ({column['type']})")
