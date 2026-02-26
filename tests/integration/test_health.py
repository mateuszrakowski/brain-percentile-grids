"""Integration tests for health and monitoring endpoints."""


class TestHealthCheck:
    """Tests for GET /api/monitoring/health."""

    def test_health_check(self, client):
        response = client.get("/api/monitoring/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.0.1"
        assert data["environment"] == "test"


class TestDetailedStatus:
    """Tests for GET /api/monitoring/status."""

    def test_detailed_status(self, client):
        response = client.get("/api/monitoring/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "timestamp" in data
        assert "uptime" in data
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "disk_usage" in system


class TestReadinessCheck:
    """Tests for GET /api/monitoring/ready."""

    def test_ready(self, client, monkeypatch):
        monkeypatch.setattr(
            "app.fastapi.routers.health.check_r_environment",
            lambda: True,
        )

        response = client.get("/api/monitoring/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["checks"]["database"] is True
        assert data["checks"]["r_environment"] is True

    def test_not_ready(self, client, monkeypatch):
        monkeypatch.setattr(
            "app.fastapi.routers.health.check_r_environment",
            lambda: False,
        )

        response = client.get("/api/monitoring/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is False
        assert data["checks"]["r_environment"] is False
