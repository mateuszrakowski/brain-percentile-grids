"""Integration tests for authentication enforcement on protected endpoints.

Verifies that all protected routers reject unauthenticated requests with 401.
"""

import pytest


class TestAuthEnforcement:
    """Verify protected endpoints reject requests without a valid token."""

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/api/datasets"),
            ("GET", "/api/datasets"),
            ("GET", "/api/datasets/1"),
            ("PATCH", "/api/datasets/1"),
            ("DELETE", "/api/datasets/1"),
            ("POST", "/api/datasets/1/upload"),
            ("GET", "/api/datasets/1/data"),
            ("DELETE", "/api/datasets/1/data"),
            ("GET", "/api/datasets/1/structures"),
            ("POST", "/api/datasets/1/fit"),
            ("POST", "/api/datasets/1/fit/stream"),
            ("POST", "/api/datasets/1/calculate"),
            ("GET", "/api/datasets/1/data/table"),
        ],
    )
    def test_unauthenticated_request(self, client, method, path):
        """Verify {method} {path} returns 401 without a token."""
        response = client.request(method, path)
        assert response.status_code == 401
