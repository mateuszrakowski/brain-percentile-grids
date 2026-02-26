"""Integration tests for dataset management endpoints.

Tests dataset CRUD operations through the HTTP client,
including conflict detection for duplicate names.
"""

import pandas as pd


class TestCreateDataset:
    """Tests for POST /api/datasets."""

    def test_duplicate_name(self, client, test_user_token, test_dataset):
        response = client.post(
            "/api/datasets",
            json={"name": "test-dataset"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.json()["detail"] == "Dataset 'test-dataset' already exists"
        assert response.status_code == 409

    def test_empty_name(self, client, test_user_token):
        response = client.post(
            "/api/datasets",
            json={"name": ""},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 422


class TestListDatasets:
    """Tests for GET /api/datasets."""

    def test_empty_list(self, client, test_user_token):
        response = client.get(
            "/api/datasets",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["datasets"] == []

    def test_list_datasets(self, client, test_user_token, test_dataset):
        response = client.get(
            "/api/datasets",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["datasets"]) == 1
        dataset = data["datasets"][0]
        assert dataset["id"] == test_dataset["id"]
        assert dataset["name"] == test_dataset["name"]
        assert dataset["sample_count"] == 0
        assert dataset["has_models"] is False
        assert dataset["structures"] == []


class TestGetDataset:
    """Tests for GET /api/datasets/{id}."""

    def test_get_dataset(self, client, test_user_token, test_dataset):
        response = client.get(
            f"/api/datasets/{test_dataset['id']}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_dataset["id"]
        assert data["name"] == test_dataset["name"]
        assert data["sample_count"] == 0
        assert data["structures"] == []
        assert data["models"] == []


class TestUpdateDataset:
    """Tests for PATCH /api/datasets/{id}."""

    def test_update_name_and_description(self, client, test_user_token, test_dataset):
        response = client.patch(
            f"/api/datasets/{test_dataset['id']}",
            json={"name": "updated-name", "description": "new desc"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "updated-name"
        assert data["description"] == "new desc"
        assert data["id"] == test_dataset["id"]

    def test_rename_to_existing_name(self, client, test_user_token, test_dataset):
        second = client.post(
            "/api/datasets",
            json={"name": "second-dataset"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert second.status_code == 200

        response = client.patch(
            f"/api/datasets/{second.json()['id']}",
            json={"name": "test-dataset"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]


class TestUploadData:
    """Tests for POST /api/datasets/{id}/upload.

    process_files is monkeypatched because real CSV parsing expects
    domain-specific column formats. The real save_reference_data runs
    against the test DB, including duplicate detection.
    """

    def test_upload_success(self, client, test_user_token, test_dataset, monkeypatch):
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

    def test_upload_duplicates(
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

    def test_upload_empty_files(self, client, test_user_token, test_dataset):
        response = client.post(
            f"/api/datasets/{test_dataset['id']}/upload",
            files=[],
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 422


class TestGetDatasetData:
    """Tests for GET /api/datasets/{id}/data."""

    def test_empty_dataset(self, client, test_user_token, test_dataset):
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
        self, client, test_user_token, test_dataset, monkeypatch
    ):
        mock_summary = {
            "total_records": 3,
            "structures": ["hippo", "amygdala"],
            "sample": [
                {
                    "patient_id": "p1",
                    "study_date": "2024-01-01",
                    "created_at": "2024-06-15T10:30:00",
                },
                {
                    "patient_id": "p2",
                    "study_date": "2024-02-01",
                    "created_at": "2024-06-15T11:00:00",
                },
            ],
        }

        monkeypatch.setattr(
            "app.fastapi.routers.data.ReferenceDataService.get_reference_summary",
            lambda self, dataset_id: mock_summary,
        )

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
        self, client, test_user_token, test_dataset, monkeypatch
    ):
        monkeypatch.setattr(
            "app.fastapi.routers.data.ReferenceDataService.get_available_structures",
            lambda self, dataset_id: ["hippo", "amygdala", "thalamus"],
        )

        response = client.get(
            f"/api/datasets/{test_dataset['id']}/structures",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == test_dataset["id"]
        assert data["structures"] == ["hippo", "amygdala", "thalamus"]
        assert data["count"] == 3


class TestDeleteDataset:
    """Tests for DELETE /api/datasets/{id}."""

    def test_delete_dataset(self, client, test_user_token, test_dataset):
        response = client.delete(
            f"/api/datasets/{test_dataset['id']}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Dataset 'test-dataset' deleted"
        assert data["patients_deleted"] == 0
        assert data["models_deleted"] == 0
        assert data["values_deleted"] == 0
