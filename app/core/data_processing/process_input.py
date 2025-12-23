from datetime import datetime

import pandas as pd
from core.resources.brain_structures import (
    CerebralCerebellumCortex,
    CerebralCortex,
    CerebrospinalFluidTotal,
    NeuralStructuresTotal,
    SubcorticalGreyMatter,
    TotalStructuresVolume,
    VentricularSupratentorialSystem,
    WhiteMatterCerebral,
    WhiteMatterTotal,
)


def _parse_input_file(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the input DataFrame into header and body sections.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame from a CSV or Excel file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the header and body DataFrames.
    """
    head = (
        df.head(5)
        .drop(columns=["Unnamed: 2"])
        .dropna()
        .set_index("Pacjent")
        .T.reset_index(drop=True)
        .iloc[:, [1, 0, 2, 3]]
    )
    head.rename(
        columns={
            "Identyfikator pacjenta": "PatientID",
            "Data urodzenia": "BirthDate",
            "Data badania": "StudyDate",
            "Opis badania": "StudyDescription",
        },
        inplace=True,
    )

    body = df[7:].copy()
    body.columns = df.iloc[6].tolist()
    body = body.set_index("Struktura").T.iloc[1:].reset_index(drop=True)

    body.columns = [
        col.replace(" – ", "_").replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for col in body.columns
    ]
    return head, body


def _calculate_age(birth_date: datetime, study_date: datetime) -> tuple[int, int]:
    """
    Calculate age in years and months from birth and study dates.

    Parameters
    ----------
    birth_date : datetime
        Birth date as a datetime object.
    study_date : datetime
        Study date as a datetime object.

    Returns
    -------
    tuple[int, int]
        A tuple containing age in years and age in months.
    """
    age_years = (
        study_date.year
        - birth_date.year
        - ((study_date.month, study_date.day) < (birth_date.month, birth_date.day))
    )

    age_months = study_date.month - birth_date.month
    if study_date.day < birth_date.day:
        age_months -= 1
    age_months %= 12

    return age_years, age_months


def process_csv_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw CSV/Excel input into a structured DataFrame.

    Extracts patient metadata from the header section and brain structure
    volumes from the body section, calculates patient age, and combines
    them into a single processed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from CSV or Excel file input.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with patient metadata and structure volumes.

    Raises
    ------
    ValueError
        If birth or study date cannot be parsed.
    """
    head, body = _parse_input_file(df)

    # Handle different date formats from CSV and XLSX files
    try:
        # Try standard datetime parsing first
        birth_date = pd.to_datetime(head["BirthDate"].iloc[0])
    except (ValueError, TypeError):
        # If that fails, try to handle Excel date formats
        try:
            birth_date = pd.to_datetime(head["BirthDate"].iloc[0], errors="coerce")
            if pd.isna(birth_date):
                # Try parsing as Excel serial date
                birth_date = pd.to_datetime(
                    head["BirthDate"].iloc[0], unit="D", origin="1899-12-30"
                )
        except Exception as e:
            raise ValueError(
                f"Could not parse birth date: {head['BirthDate'].iloc[0]}, "
                f"error: {str(e)}"
            ) from e

    try:
        # Try standard datetime parsing first
        study_date = pd.to_datetime(head["StudyDate"].iloc[0])
    except (ValueError, TypeError):
        # If that fails, try to handle Excel date formats
        try:
            study_date = pd.to_datetime(head["StudyDate"].iloc[0], errors="coerce")
            if pd.isna(study_date):
                # Try parsing as Excel serial date
                study_date = pd.to_datetime(
                    head["StudyDate"].iloc[0], unit="D", origin="1899-12-30"
                )
        except Exception as e:
            raise ValueError(
                f"Could not parse study date: {head['StudyDate'].iloc[0]}, "
                f"error: {str(e)}"
            ) from e

    head["AgeYears"], head["AgeMonths"] = _calculate_age(birth_date, study_date)

    head = head[
        [
            "PatientID",
            "AgeYears",
            "AgeMonths",
            "BirthDate",
            "StudyDate",
            "StudyDescription",
        ]
    ]

    processed_dataframe = pd.concat([head, body], axis=1)
    return processed_dataframe


def sum_structure_volumes(structures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum brain structure volumes by category.

    Aggregates individual brain structure volumes into summary categories
    such as cerebral cortex, white matter, cerebrospinal fluid, etc.

    Parameters
    ----------
    structures_df : pd.DataFrame
        DataFrame containing patient metadata and individual structure volumes.

    Returns
    -------
    pd.DataFrame
        DataFrame with patient metadata and aggregated structure volumes.
    """
    structure_classes = [
        CerebralCortex,
        CerebralCerebellumCortex,
        SubcorticalGreyMatter,
        WhiteMatterCerebral,
        WhiteMatterTotal,
        NeuralStructuresTotal,
        VentricularSupratentorialSystem,
        CerebrospinalFluidTotal,
        TotalStructuresVolume,
    ]
    summary_table = structures_df.iloc[:, :6].copy()

    for structure_class in structure_classes:
        volume_cols = list(structure_class().model_dump().values())
        # Handle numeric conversion more robustly for both CSV and XLSX data
        try:
            # First try direct conversion
            numeric_data = structures_df[volume_cols].astype(float)
        except (ValueError, TypeError):
            # If that fails, handle each column individually
            numeric_data = structures_df[volume_cols].copy()
            for col in volume_cols:
                if col in numeric_data.columns:
                    # Convert to numeric, coercing errors to NaN, then fill NaN with 0
                    numeric_data[col] = pd.to_numeric(
                        numeric_data[col], errors="coerce"
                    ).fillna(0)

        summed_volumes = numeric_data.sum(axis=1)
        summary_table[structure_class.__name__] = summed_volumes.round(2)

    return summary_table
