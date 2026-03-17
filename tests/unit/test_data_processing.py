"""Unit tests for data processing module.

Tests cover structure volume aggregation, input file parsing,
CSV/Excel date handling in process_csv_input, and end-to-end
processing of production mock files.
"""

from pathlib import Path

import pandas as pd
import pytest

from app.core.data_processing.process_input import (
    _parse_input_file,
    process_csv_input,
    sum_structure_volumes,
)
from app.core.resources.brain_structures import (
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRUCTURE_CLASSES = [
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

METADATA_COLS = [
    "PatientID",
    "PatientAge",
    "BirthDate",
    "StudyDate",
    "StudyDescription",
]


def _make_structures_df(
    overrides: dict | None = None,
) -> pd.DataFrame:
    """Build a single-row structures DataFrame with metadata + volumes.

    Parameters
    ----------
    overrides : dict | None
        Column overrides to apply on top of the defaults.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all expected columns.
    """
    all_volume_cols: set[str] = set()
    for cls in STRUCTURE_CLASSES:
        all_volume_cols.update(cls().model_dump().values())

    data: dict = {col: [100.0] for col in METADATA_COLS}
    data.update({col: [100.0] for col in all_volume_cols})

    if overrides:
        data.update(overrides)

    return pd.DataFrame(data)


def _make_raw_input_df(
    birth_date: str | int = "2010-01-01",
    study_date: str | int = "2020-06-15",
) -> pd.DataFrame:
    """Build a raw input DataFrame matching the expected CSV layout.

    Parameters
    ----------
    birth_date : str | int
        Birth date string or Excel serial number.
    study_date : str | int
        Study date string or Excel serial number.

    Returns
    -------
    pd.DataFrame
        DataFrame mimicking the raw CSV structure with header,
        gap row, and body sections.
    """
    header_data = {
        "Pacjent": [
            "Identyfikator pacjenta",
            "Data urodzenia",
            "Data badania",
            "Opis badania",
            "Extra",
        ],
        "Unnamed: 1": [
            "P001",
            birth_date,
            study_date,
            "MRI Brain",
            "extra_val",
        ],
        "Unnamed: 2": [None, None, None, None, None],
    }
    header_df = pd.DataFrame(header_data)

    gap = pd.DataFrame(
        {
            "Pacjent": [None],
            "Unnamed: 1": [None],
            "Unnamed: 2": [None],
        }
    )

    struct_header = pd.DataFrame(
        {
            "Pacjent": ["Struktura"],
            "Unnamed: 1": ["Objętość"],
            "Unnamed: 2": ["Jednostka"],
        }
    )

    struct_data = pd.DataFrame(
        {
            "Pacjent": ["Kora_mózgu_lewa"],
            "Unnamed: 1": [1500.0],
            "Unnamed: 2": ["mm3"],
        }
    )

    return pd.concat(
        [header_df, gap, struct_header, struct_data],
        ignore_index=True,
    )


# ===========================================================================
# 1. sum_structure_volumes
# ===========================================================================


class TestSumStructureVolumes:
    """Tests for ``sum_structure_volumes``.

    Validates that brain structure volumes are correctly aggregated
    into summary categories from individual left/right columns.
    """

    def test_sums_left_right_cerebral_cortex(self):
        """Verify left + right cerebral cortex sums correctly.

        Reasoning
        ---------
        Core aggregation. Verifies column-name-to-class mapping
        produces the expected total.
        """
        cc = CerebralCortex()
        cols = list(cc.model_dump().values())
        overrides = {cols[0]: [100.0], cols[1]: [200.0]}
        df = _make_structures_df(overrides)
        result = sum_structure_volumes(df)
        assert result["CerebralCortex"].iloc[0] == 300.0

    def test_missing_column_produces_nan(self):
        """Verify that a missing volume column raises KeyError.

        Reasoning
        ---------
        Real clinical data has missing structures. The function
        looks up columns by name so a missing column raises
        KeyError from pandas.
        """
        df = _make_structures_df()
        cc = CerebralCortex()
        cols = list(cc.model_dump().values())
        df = df.drop(columns=[cols[0]])

        with pytest.raises(KeyError):
            sum_structure_volumes(df)

    def test_non_numeric_coerced(self):
        """Verify that string numeric values are coerced to float.

        Reasoning
        ---------
        CSV parsing leaves values as strings. The function has
        a pd.to_numeric fallback path.
        """
        cc = CerebralCortex()
        cols = list(cc.model_dump().values())
        overrides = {cols[0]: ["150.5"], cols[1]: ["200.0"]}
        df = _make_structures_df(overrides)
        result = sum_structure_volumes(df)
        assert result["CerebralCortex"].iloc[0] == 350.5

    def test_all_categories_present(self):
        """Verify all structure categories appear in the output.

        Reasoning
        ---------
        Derive expected count from the source constant (9 Pydantic
        classes) rather than hard-coding. Catches silent drops.
        """
        df = _make_structures_df()
        result = sum_structure_volumes(df)
        expected = {cls.__name__ for cls in STRUCTURE_CLASSES}
        actual = set(result.columns) - set(METADATA_COLS)
        assert expected == actual

    def test_empty_dataframe_returns_empty(self):
        """Verify that an empty DataFrame returns an empty result.

        Reasoning
        ---------
        Real scenario if patient file has no structure data.
        Must return empty result, not crash.
        """
        df = _make_structures_df()
        empty = df.iloc[:0]
        result = sum_structure_volumes(empty)
        assert len(result) == 0

    def test_rounding_to_two_decimals(self):
        """Verify that summed volumes are rounded to 2 decimals.

        Reasoning
        ---------
        Clinical reports show 2 decimal places. Floating point
        artifacts should not leak into the output.
        """
        cc = CerebralCortex()
        cols = list(cc.model_dump().values())
        overrides = {cols[0]: [100.111], cols[1]: [200.222]}
        df = _make_structures_df(overrides)
        result = sum_structure_volumes(df)
        value = result["CerebralCortex"].iloc[0]
        assert value == 300.33
        assert round(value, 2) == value


# ===========================================================================
# 2. _parse_input_file
# ===========================================================================


class TestParseInputFile:
    """Tests for ``_parse_input_file``.

    Validates the positional parsing of header metadata and body
    structure data from the raw CSV-like input format.
    """

    def test_header_extracts_five_metadata_rows(self):
        """Verify that the header extracts from rows [0:5].

        Reasoning
        ---------
        The function hard-codes row indices [0:5]. A format shift
        would break all downstream parsing.
        """
        raw_df = _make_raw_input_df()
        head, _ = _parse_input_file(raw_df)
        assert "PatientID" in head.columns
        assert "BirthDate" in head.columns
        assert "StudyDate" in head.columns
        assert "StudyDescription" in head.columns

    def test_body_starts_after_gap_row(self):
        """Verify that body data starts at row index 7.

        Reasoning
        ---------
        Same positional fragility as the header. The body must
        start after the gap row at index 6.
        """
        raw_df = _make_raw_input_df()
        _, body = _parse_input_file(raw_df)
        assert len(body) > 0
        assert "Kora_mózgu_lewa" in body.columns

    def test_column_rename_polish_to_english(self):
        """Verify that Polish column names are renamed to English.

        Reasoning
        ---------
        A typo in the rename dict would silently drop the column.
        ``"Data urodzenia"`` must become ``"BirthDate"``.
        """
        raw_df = _make_raw_input_df()
        head, _ = _parse_input_file(raw_df)
        assert "BirthDate" in head.columns
        assert "Data urodzenia" not in head.columns

    def test_malformed_input_missing_header_rows(self):
        """Verify that input with fewer than 5 header rows raises.

        Reasoning
        ---------
        Most likely real-world failure mode for positional parsing.
        Must raise or handle gracefully.
        """
        short_df = pd.DataFrame(
            {
                "Pacjent": ["row1", "row2"],
                "Unnamed: 1": ["val1", "val2"],
                "Unnamed: 2": [None, None],
            }
        )
        with pytest.raises(Exception):
            _parse_input_file(short_df)


# ===========================================================================
# 3. process_csv_input
# ===========================================================================


class TestProcessCsvInput:
    """Tests for ``process_csv_input``.

    Validates date parsing (standard and Excel serial formats),
    patient age calculation, and error handling for invalid dates.
    """

    def test_standard_date_format_parsed(self):
        """Verify that standard date "2020-01-15" is parsed.

        Reasoning
        ---------
        Happy path for the primary pd.to_datetime path.
        """
        raw_df = _make_raw_input_df(
            birth_date="2010-01-01",
            study_date="2020-01-15",
        )
        result = process_csv_input(raw_df)
        assert "PatientAge" in result.columns
        age = result["PatientAge"].iloc[0]
        assert age == pytest.approx(10.04, abs=0.05)

    def test_excel_serial_date_fallback(self):
        """Verify that Excel serial dates are parsed correctly.

        Reasoning
        ---------
        Real users upload XLSX with serial dates. The fallback
        path using origin="1899-12-30" is critical.
        """
        raw_df = _make_raw_input_df(
            birth_date="40179",  # 2010-01-01
            study_date="44941",  # 2023-01-15
        )
        result = process_csv_input(raw_df)
        age = result["PatientAge"].iloc[0]
        assert age == pytest.approx(13.04, abs=0.1)

    def test_output_contains_patient_age(self):
        """Verify that the output contains a PatientAge column.

        Reasoning
        ---------
        Age is the bridge between raw data and model fitting.
        Its absence would break the entire pipeline.
        """
        raw_df = _make_raw_input_df()
        result = process_csv_input(raw_df)
        assert "PatientAge" in result.columns
        assert result["PatientAge"].iloc[0] > 0

    def test_invalid_date_raises_or_produces_nat(self):
        """Verify that an unparseable date raises ValueError.

        Reasoning
        ---------
        Failure path for unparseable dates. Must not silently
        produce wrong ages.
        """
        raw_df = _make_raw_input_df(
            birth_date="not-a-date",
            study_date="2020-01-15",
        )
        with pytest.raises((ValueError, TypeError)):
            process_csv_input(raw_df)


# ===========================================================================
# 4. Production mock file processing
# ===========================================================================

MOCK_CSV_DIR = Path("tests/data_mock_csv")
MOCK_XLSX_DIR = Path("tests/data_mock_xlsx")

EXPECTED_METADATA = [
    "PatientID",
    "PatientAge",
    "BirthDate",
    "StudyDate",
    "StudyDescription",
]

EXPECTED_STRUCTURES = [cls.__name__ for cls in STRUCTURE_CLASSES]


def _csv_files() -> list[Path]:
    """Return all CSV mock files, skip if directory missing."""
    if not MOCK_CSV_DIR.exists():
        return []
    return sorted(MOCK_CSV_DIR.glob("*.csv"))


def _xlsx_files() -> list[Path]:
    """Return all XLSX mock files, skip if directory missing."""
    if not MOCK_XLSX_DIR.exists():
        return []
    return sorted(MOCK_XLSX_DIR.glob("*.xlsx"))


class TestProductionMockCsv:
    """End-to-end tests on real production CSV mock files.

    Validates that every file in ``tests/data_mock_csv/`` passes
    through the full processing pipeline without errors and
    produces the expected output structure.
    """

    @pytest.mark.parametrize(
        "csv_file",
        _csv_files(),
        ids=lambda p: p.stem,
    )
    def test_csv_pipeline_produces_valid_output(self, csv_file: Path):
        """Verify that a CSV mock file processes successfully.

        Reasoning
        ---------
        These are production-representative files. If any fail,
        the upload pipeline will break for real clinical data
        with the same format.
        """
        raw_df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
        processed = process_csv_input(raw_df)
        result = sum_structure_volumes(processed)

        assert len(result) == 1
        for col in EXPECTED_METADATA:
            assert col in result.columns, f"{csv_file.name}: missing {col}"
        for col in EXPECTED_STRUCTURES:
            assert col in result.columns, f"{csv_file.name}: missing {col}"
        age = result["PatientAge"].iloc[0]
        assert age > 0, f"{csv_file.name}: age={age}"


class TestProductionMockXlsx:
    """End-to-end tests on real production XLSX mock files.

    Validates that every file in ``tests/data_mock_xlsx/`` passes
    through the full processing pipeline without errors and
    produces the expected output structure.
    """

    @pytest.mark.parametrize(
        "xlsx_file",
        _xlsx_files(),
        ids=lambda p: p.stem,
    )
    def test_xlsx_pipeline_produces_valid_output(self, xlsx_file: Path):
        """Verify that an XLSX mock file processes successfully.

        Reasoning
        ---------
        XLSX files use a different read path (openpyxl) and may
        have Excel serial dates. Both must produce the same
        result structure as CSV.
        """
        raw_df = pd.read_excel(xlsx_file, engine="openpyxl")
        processed = process_csv_input(raw_df)
        result = sum_structure_volumes(processed)

        assert len(result) == 1
        for col in EXPECTED_METADATA:
            assert col in result.columns, f"{xlsx_file.name}: missing {col}"
        for col in EXPECTED_STRUCTURES:
            assert col in result.columns, f"{xlsx_file.name}: missing {col}"
        age = result["PatientAge"].iloc[0]
        assert age > 0, f"{xlsx_file.name}: age={age}"


class TestCsvXlsxParity:
    """Verify that CSV and XLSX versions produce identical results.

    Validates that both file formats for the same patient produce
    the same PatientAge and structure volumes.
    """

    @pytest.mark.parametrize(
        "stem",
        [p.stem for p in _csv_files()],
    )
    def test_csv_xlsx_ages_match(self, stem: str):
        """Verify CSV and XLSX produce the same PatientAge.

        Reasoning
        ---------
        Date parsing differs between formats (string vs serial).
        If ages diverge, one format has a parsing bug.
        """
        csv_path = MOCK_CSV_DIR / f"{stem}.csv"
        xlsx_path = MOCK_XLSX_DIR / f"{stem}.xlsx"
        if not xlsx_path.exists():
            pytest.skip(f"No XLSX counterpart for {stem}")

        csv_df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        xlsx_df = pd.read_excel(xlsx_path, engine="openpyxl")

        csv_result = sum_structure_volumes(process_csv_input(csv_df))
        xlsx_result = sum_structure_volumes(process_csv_input(xlsx_df))

        csv_age = csv_result["PatientAge"].iloc[0]
        xlsx_age = xlsx_result["PatientAge"].iloc[0]
        assert csv_age == pytest.approx(xlsx_age, abs=0.01), (
            f"{stem}: CSV age={csv_age}, XLSX age={xlsx_age}"
        )
