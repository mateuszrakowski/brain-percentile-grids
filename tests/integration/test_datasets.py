"""Integration tests for dataset management endpoints.

Tests dataset CRUD operations through the HTTP client,
including conflict detection for duplicate names.
"""

import pandas as pd


class TestCreateDataset:
    """Tests for POST /api/datasets."""

    def test_existing_create_dataset(self, client, test_user_token, test_dataset):
        response = client.post(
            "/api/datasets",
            json={"name": "test-dataset"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.json()["detail"] == "Dataset 'test-dataset' already exists"
        assert response.status_code == 409


class TestUploadData:
    """Tests for POST /api/datasets/{id}/upload.

    process_files is monkeypatched because real CSV parsing expects
    domain-specific column formats. The real save_reference_data runs
    against the test DB, including duplicate detection.
    """

    def test_upload_data_to_dataset(
        self, client, test_user_token, test_dataset, monkeypatch
    ):
        monkeypatch.setattr(
            "app.fastapi.routers.data.PatientDataProcessor.process_files",
            lambda self, files: [
                pd.DataFrame(
                    {
                        "PatientID": ["p1", "p2"],
                        "AgeYears": [25, 35],
                        "StudyDate": ["2024-01-01", "2024-01-02"],
                        "StudyDescription": ["scan1", "scan2"],
                        "hippo": [0.5, 0.6],
                    }
                )
            ],
        )

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

    def test_upload_data_to_dataset_duplicates(
        self, client, test_user_token, test_dataset, monkeypatch
    ):
        monkeypatch.setattr(
            "app.fastapi.routers.data.PatientDataProcessor.process_files",
            lambda self, files: [
                pd.DataFrame(
                    {
                        "PatientID": ["p1", "p2"],
                        "AgeYears": [25, 35],
                        "StudyDate": ["2024-01-01", "2024-01-02"],
                        "StudyDescription": ["scan1", "scan2"],
                        "hippo": [0.5, 0.6],
                    }
                )
            ],
        )

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

    def test_upload_data_to_dataset_empty(
        self, client, test_user_token, test_dataset
    ):
        response = client.post(
            f"/api/datasets/{test_dataset['id']}/upload",
            files=[],
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 422
