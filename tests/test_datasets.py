class TestCreateDataset:
    def test_existing_create_dataset(self, client, test_user_token, test_dataset):
        response = client.post(
            "/api/datasets",
            json={"name": "test-dataset"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.json()["detail"] == "Dataset 'test-dataset' already exists"
        assert response.status_code == 409
