import pytest
from fastapi import HTTPException

from app.fastapi.dependencies import get_user_dataset


class TestUserDataset:
    async def test_get_user_dataset(self, test_session, test_dataset, test_user_db):
        user_dataset = await get_user_dataset(
            dataset_id=test_dataset["id"],
            current_user=test_user_db,
            session=test_session,
        )

        assert user_dataset.id == test_dataset["id"]

    async def test_not_existing_user_dataset(
        self, test_session, test_user_db
    ):
        with pytest.raises(HTTPException) as exc_info:
            await get_user_dataset(
                dataset_id=99,
                current_user=test_user_db,
                session=test_session,
            )

        assert exc_info.value.status_code == 404
