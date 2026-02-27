"""Integration tests for shared FastAPI dependencies.

Tests get_user_dataset by calling the async function directly
(not through TestClient) with real DB objects. This is async
because we need to await the function — there's no TestClient
to handle async wrapping for us.
"""

import pytest
from fastapi import HTTPException

from app.fastapi.dependencies import get_user_dataset


class TestUserDataset:
    """Tests for get_user_dataset dependency."""
    async def test_get_user_dataset(self, test_session, test_dataset, test_user_db):
        """Verify dependency returns the dataset when user owns it."""
        user_dataset = await get_user_dataset(
            dataset_id=test_dataset["id"],
            current_user=test_user_db,
            session=test_session,
        )

        assert user_dataset.id == test_dataset["id"]

    async def test_not_existing_user_dataset(self, test_session, test_user_db):
        """Verify dependency raises 404 for non-existent dataset ID."""
        with pytest.raises(HTTPException) as exc_info:
            await get_user_dataset(
                dataset_id=99,
                current_user=test_user_db,
                session=test_session,
            )

        assert exc_info.value.status_code == 404
