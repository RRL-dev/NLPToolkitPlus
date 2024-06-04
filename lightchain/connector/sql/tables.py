"""The module provides a function to retrieve information about tables in an SQL database using SQLAlchemy."""  # noqa: E501

from typing import Any, Literal

from sqlalchemy import CursorResult, Engine, MetaData, Table, create_engine, inspect, select
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.sqltypes import NullType


def get_table_info_from_sql_database(  # noqa: C901, PLR0913
    engine: Engine,
    schema: str | None = None,
    table_names: list[str] | None = None,
    view_support: bool = False,  # noqa: FBT001, FBT002
    custom_table_info: dict[str, str] | None = None,
    max_string_length: int = 300,
    indexes_in_table_info: bool = False,  # noqa: FBT001, FBT002
    sample_rows_in_table_info: int = 3,
) -> str:
    """Get information about specified tables in an SQL database.

    Args:
    ----
        engine (Engine): The SQLAlchemy engine.
        schema (Optional[str], optional): Schema to reflect.
        table_names (Optional[List[str]], optional): List of specific table names to retrieve information for.
        view_support (bool, optional): Whether to include view names in the info.
        custom_table_info (Optional[dict[str, str]], optional): Custom info for specific tables.
        max_string_length (int, optional): Maximum length for string columns.
        indexes_in_table_info (bool, optional): Whether to include indexes in the info.
        sample_rows_in_table_info (int, optional): Number of sample rows to include in the info.

    Returns:
    -------
        str: The formatted table information.

    """  # noqa: E501
    inspector: Any = inspect(engine)
    all_tables: set[str] = set(
        inspector.get_table_names(schema=schema)
        + (inspector.get_view_names(schema=schema) if view_support else []),
    )

    usable_table_names: set[str] = set(table_names) if table_names else all_tables

    metadata = MetaData()
    metadata.reflect(views=view_support, bind=engine, only=list(usable_table_names), schema=schema)

    meta_tables: list[Table] = [
        tbl
        for tbl in metadata.sorted_tables
        if tbl.name in usable_table_names
        and not (engine.dialect.name == "sqlite" and tbl.name.startswith("sqlite_"))
    ]

    def _get_table_indexes(table: Table) -> str:
        indexes: Any = inspector.get_indexes(table.name)
        indexes_formatted: str = "\n".join(
            [f"INDEX {idx['name']}: {idx['column_names']}" for idx in indexes],
        )
        return f"Table Indexes:\n{indexes_formatted}"

    def _get_sample_rows(table: Table) -> str:
        command = select(table).limit(sample_rows_in_table_info)
        columns_str: str = "\t".join([col.name for col in table.columns])
        sample_rows_str = ""

        try:
            with engine.connect() as connection:
                sample_rows_result: CursorResult[Any] = connection.execute(statement=command)
                sample_rows: list[list[str]] = [
                    [str(i)[:max_string_length] for i in row] for row in sample_rows_result
                ]
                sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])
        except Exception as e:  # noqa: BLE001
            sample_rows_str = f"Error fetching sample rows: {e}"

        return f"{sample_rows_in_table_info} rows from {table.name} table:\n{columns_str}\n{sample_rows_str}"  # noqa: E501

    tables: list[str] = []
    for table in meta_tables:
        if custom_table_info and table.name in custom_table_info:
            tables.append(custom_table_info[table.name])
            continue

        for v in table.columns.values():
            if isinstance(v.type, NullType):
                table._columns.remove(column=v)  # noqa: SLF001

        create_table = str(object=CreateTable(element=table).compile(bind=engine))
        table_info: str = f"{create_table.rstrip()}"
        has_extra_info: int | Literal[True] = indexes_in_table_info or sample_rows_in_table_info

        if has_extra_info:
            table_info += "\n\n/*"
        if indexes_in_table_info:
            table_info += f"\n{_get_table_indexes(table=table)}\n"
        if sample_rows_in_table_info:
            table_info += f"\n{_get_sample_rows(table=table)}\n"
        if has_extra_info:
            table_info += "*/"
        tables.append(table_info)
    tables.sort()
    return "\n\n".join(tables)


# Example usage
if __name__ == "__main__":
    engine: Engine = create_engine(url="sqlite:///resources/database/Chinook.db")
    get_table_info_from_sql_database(engine=engine)
