"""Integration tests for dataset data endpoints.

Tests data upload, retrieval, clearing, and structure listing
through the HTTP client (routers/data.py).
"""

from unittest.mock import patch

import pandas as pd

from app.fastapi.services.reference_data import (
    ReferenceDataService,
    ReferenceSummary,
    SampleRecord,
)
from app.fastapi.utils.file_utils import PatientDataProcessor

MOCK_PATIENT_DFS = [
    pd.DataFrame(
        {
            "PatientID": ["p1", "p2"],
            "AgeYears": [25, 35],
            "StudyDate": ["2024-01-01", "2024-01-02"],
            "StudyDescription": ["scan1", "scan2"],
            "hippo": [0.5, 0.6],
        }
    )
]


class TestUploadData:
    """Tests for POST /api/datasets/{id}/upload.

    process_files is patched because real CSV parsing expects
    domain-specific column formats. The real save_reference_data runs
    against the test DB, including duplicate detection.
    """

    def test_upload_success(self, client, test_user_token, test_dataset):
        """Verify successful upload inserts records and returns correct message."""
        with patch.object(
            PatientDataProcessor,
            "process_files",
            return_value=MOCK_PATIENT_DFS,
        ):
            response = client.post(
                f"/api/datasets/{test_dataset['id']}/upload",
                files=[
                    (
                        "files",
                        ("patient1.csv", b"content", "text/csv"),
                    ),
                    (
                        "files",
                        ("patient2.csv", b"content", "text/csv"),
                    ),
                ],
                headers={"Authorization": f"Bearer {test_user_token}"},
            )

        assert response.status_code == 200
        assert (
            response.json()["message"]
            == "Successfully added 2 records to 'test-dataset'"
        )

    def test_upload_duplicates(
        self, client, test_user_token, test_dataset
    ):
        """Verify second upload of same data detects duplicates and adds 0 records."""
        with patch.object(
            PatientDataProcessor,
            "process_files",
            return_value=MOCK_PATIENT_DFS,
        ):
            _ = client.post(
                f"/api/datasets/{test_dataset['id']}/upload",
                files=[
                    (
                        "files",
                        ("patient1.csv", b"content", "text/csv"),
                    ),
                    (
                        "files",
                        ("patient2.csv", b"content", "text/csv"),
                    ),
                ],
                headers={"Authorization": f"Bearer {test_user_token}"},
            )

            response_second = client.post(
                f"/api/datasets/{test_dataset['id']}/upload",
                files=[
                    (
                        "files",
                        ("patient1.csv", b"content", "text/csv"),
                    ),
                    (
                        "files",
                        ("patient2.csv", b"content", "text/csv"),
                    ),
                ],
                headers={"Authorization": f"Bearer {test_user_token}"},
            )

        assert response_second.status_code == 200
        assert (
            response_second.json()["message"]
            == "Successfully added 0 records to 'test-dataset'"
        )

    def test_upload_empty_files(self, client, test_user_token, test_dataset):
        """Verify empty file list is rejected with 422."""
        response = client.post(
            f"/api/datasets/{test_dataset['id']}/upload",
            files=[],
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 422


class TestGetDatasetData:
    """Tests for GET /api/datasets/{id}/data."""

    def test_empty_dataset(self, client, test_user_token, test_dataset):
        """Verify dataset with no uploaded data returns empty summary."""
        response = client.get(
            f"/api/datasets/{test_dataset['id']}/data",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == test_dataset["id"]
        assert data["dataset_name"] == test_dataset["name"]
        assert data["total_records"] == 0
        assert data["structures"] == []
        assert data["sample"] == []

    def test_populated_dataset(
        self, client, test_user_token, test_dataset
    ):
        """Verify patched summary is correctly mapped to GetDataResponse."""
        mock_summary = ReferenceSummary(
            total_records=3,
            structures=["hippo", "amygdala"],
            sample=[
                SampleRecord(
                    patient_id="p1",
                    study_date="2024-01-01",
                    created_at="2024-06-15T10:30:00",
                ),
                SampleRecord(
                    patient_id="p2",
                    study_date="2024-02-01",
                    created_at="2024-06-15T11:00:00",
                ),
            ],
        )

        with patch.object(
            ReferenceDataService,
            "get_reference_summary",
            return_value=mock_summary,
        ):
            response = client.get(
                f"/api/datasets/{test_dataset['id']}/data",
                headers={"Authorization": f"Bearer {test_user_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == test_dataset["id"]
        assert data["dataset_name"] == test_dataset["name"]
        assert data["total_records"] == 3
        assert data["structures"] == ["hippo", "amygdala"]
        assert len(data["sample"]) == 2
        assert data["sample"][0]["patient_id"] == "p1"
        assert data["sample"][1]["patient_id"] == "p2"


class TestClearDatasetData:
    """Tests for DELETE /api/datasets/{id}/data."""

    def test_clear_empty_dataset(self, client, test_user_token, test_dataset):
        """Verify clearing an empty dataset returns 0 deleted records."""
        response = client.delete(
            f"/api/datasets/{test_dataset['id']}/data",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == test_dataset["id"]
        assert data["records_deleted"] == 0
        assert data["message"] == "Deleted 0 records from 'test-dataset'"


class TestGetDatasetStructures:
    """Tests for GET /api/datasets/{id}/structures."""

    def test_returns_structures(
        self, client, test_user_token, test_dataset
    ):
        """Verify patched structures are returned with correct count."""
        with patch.object(
            ReferenceDataService,
            "get_available_structures",
            return_value=["hippo", "amygdala", "thalamus"],
        ):
            response = client.get(
                f"/api/datasets/{test_dataset['id']}/structures",
                headers={"Authorization": f"Bearer {test_user_token}"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == test_dataset["id"]
        assert data["structures"] == ["hippo", "amygdala", "thalamus"]
        assert data["count"] == 3
