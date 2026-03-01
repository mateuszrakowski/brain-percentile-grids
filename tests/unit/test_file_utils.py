"""Unit tests for file utility classes."""

from unittest.mock import Mock

import pytest

from app.fastapi.utils.file_utils import ValidatedFile


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
        vf = _make_validated_file("file<>:\"|?*.csv")
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
