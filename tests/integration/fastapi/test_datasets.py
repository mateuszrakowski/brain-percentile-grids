"""Integration tests for dataset management endpoints.

Tests dataset CRUD operations through the HTTP client,
including conflict detection for duplicate names.
"""


class TestCreateDataset:
    """Tests for POST /api/datasets."""

    def test_duplicate_name(self, client, test_user_token, test_dataset):
        """Verify creating a dataset with an existing name returns 409."""
        response = client.post(
            "/api/datasets",
            json={"name": "test-dataset"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.json()["detail"] == "Dataset 'test-dataset' already exists"
        assert response.status_code == 409

    def test_empty_name(self, client, test_user_token):
        """Verify empty name is rejected by Pydantic min_length=1 validation."""
        response = client.post(
            "/api/datasets",
            json={"name": ""},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 422


class TestListDatasets:
    """Tests for GET /api/datasets."""

    def test_empty_list(self, client, test_user_token):
        """Verify listing with no datasets returns empty list and total 0."""
        response = client.get(
            "/api/datasets",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["datasets"] == []

    def test_list_datasets(self, client, test_user_token, test_dataset):
        """Verify created dataset appears in the list with correct fields."""
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
        """Verify dataset detail returns correct fields for an empty dataset."""
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
        """Verify both name and description can be updated in a single PATCH."""
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
        """Verify renaming to another user's existing dataset name returns 409."""
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


class TestDeleteDataset:
    """Tests for DELETE /api/datasets/{id}."""

    def test_delete_dataset(self, client, test_user_token, test_dataset):
        """Verify deleting an empty dataset returns 0 counts for all deletions."""
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
