"""Unit tests for file utility classes."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.fastapi.utils.file_utils import PatientDataProcessor, ValidatedFile


class TestSecureFilename:
    """Tests for ValidatedFile filename sanitization."""

    @pytest.fixture
    def _make_validated_file(self):
        """Return a factory that creates a ValidatedFile with a given filename."""

        def factory(filename):
            upload = Mock()
            upload.filename = filename
            upload.content_type = "text/csv"
            return ValidatedFile(upload, b"content")

        return factory

    def test_normal_filename(self, _make_validated_file):
        """Verify normal filenames are preserved."""
        vf = _make_validated_file("patient_data.csv")
        assert vf.name == "patient_data.csv"

    def test_path_traversal(self, _make_validated_file):
        """Verify path separator is stripped, preventing directory traversal."""
        vf = _make_validated_file("../../etc/passwd")
        assert "/" not in vf.name
        assert vf.name == "....etcpasswd"

    def test_spaces_stripped(self, _make_validated_file):
        """Verify spaces are removed from filenames."""
        vf = _make_validated_file("my file name.csv")
        assert " " not in vf.name

    def test_special_characters_stripped(self, _make_validated_file):
        """Verify special characters are removed."""
        vf = _make_validated_file('file<>:"|?*.csv')
        assert vf.name == "file.csv"

    def test_empty_filename_fallback(self):
        """Verify missing filename defaults to 'unknown'."""
        upload = Mock()
        upload.filename = None
        upload.content_type = "text/csv"
        vf = ValidatedFile(upload, b"content")
        assert vf.name == "unknown"


class TestValidatedFileExtension:
    """Tests for ValidatedFile extension extraction."""

    def test_csv_extension(self):
        """Verify .csv extension is extracted."""
        upload = Mock()
        upload.filename = "data.CSV"
        upload.content_type = "text/csv"
        vf = ValidatedFile(upload, b"content")
        assert vf.extension == ".csv"

    def test_xlsx_extension(self):
        """Verify .xlsx extension is extracted."""
        upload = Mock()
        upload.filename = "data.XLSX"
        upload.content_type = "application/vnd.ms-excel"
        vf = ValidatedFile(upload, b"content")
        assert vf.extension == ".xlsx"


class TestValidatedFileBuffer:
    """Tests for ValidatedFile content access."""

    def test_to_buffer_roundtrip(self):
        """Verify content survives the bytes-to-BytesIO roundtrip."""
        upload = Mock()
        upload.filename = "test.csv"
        upload.content_type = "text/csv"
        content = b"PatientID,AgeYears\np1,25"
        vf = ValidatedFile(upload, content)

        assert vf.read() == content
        assert vf.to_buffer().read() == content


# === PatientDataProcessor tests ===


def _make_file(name, extension, content=b"dummy"):
    """Create a ValidatedFile mock with the given name and extension."""
    vf = Mock(spec=ValidatedFile)
    vf.name = name
    vf.extension = extension
    vf.content = content
    vf.to_buffer.return_value = Mock()
    return vf


class TestReadFile:
    """Tests for PatientDataProcessor._read_file dispatch."""

    def test_csv_uses_read_csv(self):
        """Verify .csv files are read with pd.read_csv."""
        processor = PatientDataProcessor()
        vf = _make_file("data.csv", ".csv")

        with patch("app.fastapi.utils.file_utils.pd.read_csv") as mock_read:
            mock_read.return_value = pd.DataFrame({"a": [1]})
            result = processor._read_file(vf)

        mock_read.assert_called_once()
        assert len(result) == 1

    def test_xlsx_uses_read_excel_openpyxl(self):
        """Verify .xlsx files are read with openpyxl engine."""
        processor = PatientDataProcessor()
        vf = _make_file("data.xlsx", ".xlsx")

        with patch("app.fastapi.utils.file_utils.pd.read_excel") as mock_read:
            mock_read.return_value = pd.DataFrame({"a": [1]})
            result = processor._read_file(vf)

        mock_read.assert_called_once_with(vf.to_buffer(), engine="openpyxl")
        assert len(result) == 1

    def test_xls_uses_read_excel_xlrd(self):
        """Verify .xls files are read with xlrd engine."""
        processor = PatientDataProcessor()
        vf = _make_file("data.xls", ".xls")

        with patch("app.fastapi.utils.file_utils.pd.read_excel") as mock_read:
            mock_read.return_value = pd.DataFrame({"a": [1]})
            result = processor._read_file(vf)

        mock_read.assert_called_once_with(vf.to_buffer(), engine="xlrd")
        assert len(result) == 1

    def test_unsupported_extension_raises(self):
        """Verify unsupported extension raises ValueError."""
        processor = PatientDataProcessor()
        vf = _make_file("data.json", ".json")

        with pytest.raises(ValueError, match="Unsupported file type"):
            processor._read_file(vf)


class TestReadAndProcessFile:
    """Tests for PatientDataProcessor._read_and_process_file."""

    def test_empty_file_returns_empty_df(self):
        """Verify empty file returns an empty DataFrame."""
        processor = PatientDataProcessor()
        vf = _make_file("empty.csv", ".csv")

        with patch.object(processor, "_read_file", return_value=pd.DataFrame()):
            result = processor._read_and_process_file(vf)

        assert result.empty

    def test_processing_pipeline_applied(self):
        """Verify process_csv_input and sum_structure_volumes are called."""
        processor = PatientDataProcessor()
        vf = _make_file("data.csv", ".csv")
        raw_df = pd.DataFrame({"col": [1]})
        processed_df = pd.DataFrame(
            {
                "PatientID": ["p1"],
                "AgeYears": [25],
                "hippo": [0.5],
            }
        )

        with (
            patch.object(processor, "_read_file", return_value=raw_df),
            patch(
                "app.fastapi.utils.file_utils.process_csv_input",
                return_value=processed_df,
            ) as mock_process,
            patch(
                "app.fastapi.utils.file_utils.sum_structure_volumes",
                return_value=processed_df.copy(),
            ) as mock_sum,
        ):
            result = processor._read_and_process_file(vf)

        mock_process.assert_called_once()
        mock_sum.assert_called_once()
        assert "Filename" in result.columns
        assert result["Filename"].iloc[0] == "data.csv"

    def test_processing_error_wraps_in_valueerror(self):
        """Verify exceptions from core processing are wrapped in ValueError."""
        processor = PatientDataProcessor()
        vf = _make_file("bad.csv", ".csv")

        with (
            patch.object(
                processor,
                "_read_file",
                side_effect=RuntimeError("parse failed"),
            ),
            pytest.raises(ValueError, match="Error processing bad.csv"),
        ):
            processor._read_and_process_file(vf)


class TestProcessFiles:
    """Tests for PatientDataProcessor.process_files."""

    def test_filters_empty_dataframes(self):
        """Verify empty DataFrames from individual files are excluded."""
        processor = PatientDataProcessor()
        files = [
            _make_file("good.csv", ".csv"),
            _make_file("empty.csv", ".csv"),
        ]

        results = [
            pd.DataFrame({"PatientID": ["p1"], "hippo": [0.5]}),
            pd.DataFrame(),
        ]

        with patch.object(
            processor,
            "_read_and_process_file",
            side_effect=results,
        ):
            processed = processor.process_files(files)

        assert len(processed) == 1
        assert processed[0]["PatientID"].iloc[0] == "p1"

    def test_empty_file_list(self):
        """Verify empty input returns empty list."""
        processor = PatientDataProcessor()
        assert processor.process_files([]) == []


class TestReorderColumns:
    """Tests for PatientDataProcessor._reorder_columns."""

    def test_metadata_first_then_structures_sorted(self):
        """Verify metadata columns come first, structures are sorted."""
        processor = PatientDataProcessor()
        df = pd.DataFrame(
            {
                "zeta_structure": [1],
                "PatientID": ["p1"],
                "alpha_structure": [2],
                "AgeYears": [25],
            }
        )

        result = processor._reorder_columns(df)
        columns = list(result.columns)

        assert columns == [
            "PatientID",
            "AgeYears",
            "alpha_structure",
            "zeta_structure",
        ]

    def test_missing_metadata_columns_skipped(self):
        """Verify missing metadata columns don't cause errors."""
        processor = PatientDataProcessor()
        df = pd.DataFrame(
            {
                "hippo": [0.5],
                "PatientID": ["p1"],
            }
        )

        result = processor._reorder_columns(df)
        assert list(result.columns) == ["PatientID", "hippo"]


class TestGetStructureColumns:
    """Tests for PatientDataProcessor.get_structure_columns."""

    def test_excludes_metadata(self):
        """Verify metadata columns are excluded from structure list."""
        df = pd.DataFrame(
            {
                "PatientID": ["p1"],
                "AgeYears": [25],
                "StudyDate": ["2024-01-01"],
                "hippo": [0.5],
                "amygdala": [0.3],
            }
        )

        structures = PatientDataProcessor.get_structure_columns(df)
        assert sorted(structures) == ["amygdala", "hippo"]

    def test_no_structures(self):
        """Verify returns empty list when only metadata columns exist."""
        df = pd.DataFrame(
            {
                "PatientID": ["p1"],
                "AgeYears": [25],
            }
        )

        structures = PatientDataProcessor.get_structure_columns(df)
        assert structures == []
