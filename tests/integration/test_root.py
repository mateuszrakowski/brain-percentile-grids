"""Integration tests for root-level endpoints defined in main.py."""


class TestRootEndpoint:
    """Tests for GET /."""

    def test_returns_html(self, client):
        """Verify root returns HTML with app name."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Percentile Grids API" in response.text


class TestRootHealthCheck:
    """Tests for GET /health (root-level, not /api/monitoring/health)."""

    def test_health_check(self, client):
        """Verify root health endpoint returns status and version."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data
