import os
import sqlite3
from typing import Any

import pandas as pd
import sqlalchemy

from core.data_processing.process_input import (
    convert_to_dataframes,
    process_csv_input,
    sum_structure_volumes,
)

# Note: UploadedFile functionality replaced with Flask file handling

DEFAULT_DB_PATH = "data/reference_dataset.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database, creating it if necessary."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    return sqlite3.connect(db_path)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Checks if a table exists in the database."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = :table_name",
        {"table_name": table_name},
    )
    return cur.fetchone() is not None


def _create_tables_from_schema(conn: sqlite3.Connection, sample_dataframe: pd.DataFrame) -> None:
    """Creates database tables based on a sample dataframe schema if they don't exist."""
    processed_dataframe = process_csv_input(sample_dataframe)
    structures_summary = sum_structure_volumes(processed_dataframe)

    patient_metadata = [
        f"{col} TEXT" if "Age" not in col else f"{col} INTEGER"
        for col in processed_dataframe.columns[:6]
    ]
    structure_columns = patient_metadata + [
        f"{col} REAL" for col in processed_dataframe.columns[6:]
    ]
    summary_columns = patient_metadata + [f"{col} REAL" for col in structures_summary.columns[6:]]

    cur = conn.cursor()
    if not _table_exists(conn, "PatientStructures"):
        cur.execute(
            f"CREATE TABLE PatientStructures ({', '.join(structure_columns)}, "
            "UNIQUE(PatientID, StudyDate, StudyDescription) ON CONFLICT IGNORE)"
        )

    if not _table_exists(conn, "PatientSummary"):
        cur.execute(
            f"CREATE TABLE PatientSummary ({', '.join(summary_columns)}, "
            "UNIQUE(PatientID, StudyDate, StudyDescription) ON CONFLICT IGNORE)"
        )
    conn.commit()


def load_db_data(
    table_name: str,
    db_path: str = DEFAULT_DB_PATH,
    sample_dataframe: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    Loads a database table into a pandas DataFrame.
    If the table doesn't exist and a sample_dataframe is provided,
    it creates the tables before loading.
    """
    conn = get_db_connection(db_path)
    try:
        if not _table_exists(conn, table_name):
            if sample_dataframe is not None:
                _create_tables_from_schema(conn, sample_dataframe)
            else:
                return None  # Table doesn't exist and no schema to create it

        engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
        with engine.connect() as sql_conn:
            return pd.read_sql_table(table_name, sql_conn)
    finally:
        conn.close()


def update_db(
    input_files: list[Any],  # Legacy function - UploadedFile no longer defined
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Updates the database with data from uploaded files."""
    if not input_files:
        return

    dataframes = convert_to_dataframes(input_files)

    # Ensure tables exist before inserting data
    load_db_data("PatientSummary", db_path, sample_dataframe=dataframes[0])

    conn = get_db_connection(db_path)
    try:
        structures_data = []
        summary_data = []

        for input_csv in dataframes:
            processed_dataframe = process_csv_input(input_csv)
            structures_summary = sum_structure_volumes(processed_dataframe)
            structures_data.append(processed_dataframe.iloc[0].to_dict())
            summary_data.append(structures_summary.iloc[0].to_dict())

        cur = conn.cursor()

        # Assuming processed_dataframe and structures_summary are defined from the loop
        cur.executemany(
            (
                "INSERT INTO PatientStructures VALUES "
                f"({', '.join((f':{col}' for col in processed_dataframe.columns))})"
            ),
            structures_data,
        )

        cur.executemany(
            (
                "INSERT INTO PatientSummary VALUES "
                f"({', '.join((f':{col}' for col in structures_summary.columns))})"
            ),
            summary_data,
        )

        conn.commit()
    finally:
        conn.close()
