"""Integration tests for root-level endpoints defined in main.py."""


class TestRootEndpoint:
    """Tests for GET /."""

    def test_returns_html(self, client):
        """Verify root returns HTML with app name."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Brain Percentile Grids" in response.text

    def test_root_serves_doctype_html(self, client):
        """Verify root response contains a valid HTML doctype."""
        response = client.get("/")

        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text


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


class TestStaticFiles:
    """Tests for static file serving at /static."""

    def test_css_accessible(self, client):
        """Verify CSS stylesheet is served."""
        response = client.get("/static/css/styles.css")

        assert response.status_code == 200

    def test_js_accessible(self, client):
        """Verify JavaScript app file is served."""
        response = client.get("/static/js/app.js")

        assert response.status_code == 200

    def test_nonexistent_returns_404(self, client):
        """Verify nonexistent static file returns 404."""
        response = client.get("/static/nope.js")

        assert response.status_code == 404
