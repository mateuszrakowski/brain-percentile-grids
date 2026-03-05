"""Unit tests for shared FastAPI dependencies.

Tests get_user_dataset and get_validated_files directly,
bypassing the HTTP layer.
"""

from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, UploadFile

from app.fastapi.db.models import User
from app.fastapi.dependencies import get_user_dataset, get_validated_files


class TestUserDataset:
    """Tests for get_user_dataset dependency."""

    async def test_get_user_dataset(self, test_session, test_dataset_db):
        """Verify dependency returns the dataset when user owns it."""
        owner = test_session.get(User, test_dataset_db.user_id)
        user_dataset = await get_user_dataset(
            dataset_id=test_dataset_db.id,
            current_user=owner,
            session=test_session,
        )

        assert user_dataset.id == test_dataset_db.id

    async def test_not_existing_user_dataset(self, test_session, test_dataset_db):
        """Verify dependency raises 404 for non-existent dataset ID."""
        owner = test_session.get(User, test_dataset_db.user_id)
        with pytest.raises(HTTPException) as exc_info:
            await get_user_dataset(
                dataset_id=99,
                current_user=owner,
                session=test_session,
            )

        assert exc_info.value.status_code == 404

    async def test_user_cannot_access_other_users_dataset(
        self, test_session, test_dataset_db
    ):
        """Verify dependency raises 404 when dataset belongs to a different user."""
        other_user = User(username="other-user", hashed_password="hashed")
        test_session.add(other_user)
        test_session.flush()

        with pytest.raises(HTTPException) as exc_info:
            await get_user_dataset(
                dataset_id=test_dataset_db.id,
                current_user=other_user,
                session=test_session,
            )

        assert exc_info.value.status_code == 404


class TestValidatedFiles:
    """Tests for get_validated_files dependency."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with small limits for testing."""
        return Mock(
            max_files_count=2,
            allowed_extensions=["csv", "xlsx"],
            max_upload_size=1024,
        )

    def _make_upload(self, filename: str | None, content: bytes) -> UploadFile:
        """Create an UploadFile with the given filename and content."""
        return UploadFile(filename=filename, file=BytesIO(content))

    async def test_valid_file(self, mock_settings):
        """Verify a valid CSV file passes validation."""
        files = [self._make_upload("data.csv", b"col1,col2\n1,2")]
        result = await get_validated_files(files=files, settings=mock_settings)

        assert len(result) == 1
        assert result[0].name == "data.csv"

    async def test_too_many_files(self, mock_settings):
        """Verify exceeding max_files_count raises 400."""
        files = [
            self._make_upload("a.csv", b"data"),
            self._make_upload("b.csv", b"data"),
            self._make_upload("c.csv", b"data"),
        ]

        with pytest.raises(HTTPException) as exc_info:
            await get_validated_files(files=files, settings=mock_settings)

        assert exc_info.value.status_code == 400
        assert "Too many files" in exc_info.value.detail

    async def test_missing_filename(self, mock_settings):
        """Verify file without filename raises 400."""
        files = [self._make_upload(None, b"data")]

        with pytest.raises(HTTPException) as exc_info:
            await get_validated_files(files=files, settings=mock_settings)

        assert exc_info.value.status_code == 400
        assert "missing filename" in exc_info.value.detail

    async def test_unsupported_extension(self, mock_settings):
        """Verify file with disallowed extension raises 400."""
        files = [self._make_upload("data.txt", b"data")]

        with pytest.raises(HTTPException) as exc_info:
            await get_validated_files(files=files, settings=mock_settings)

        assert exc_info.value.status_code == 400
        assert "not allowed" in exc_info.value.detail

    async def test_file_too_large(self, mock_settings):
        """Verify file exceeding max_upload_size raises 400."""
        files = [self._make_upload("data.csv", b"x" * 2048)]

        with pytest.raises(HTTPException) as exc_info:
            await get_validated_files(files=files, settings=mock_settings)

        assert exc_info.value.status_code == 400
        assert "too large" in exc_info.value.detail

    async def test_empty_file(self, mock_settings):
        """Verify empty file raises 400."""
        files = [self._make_upload("data.csv", b"")]

        with pytest.raises(HTTPException) as exc_info:
            await get_validated_files(files=files, settings=mock_settings)

        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail
