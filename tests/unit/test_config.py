"""Unit tests for Settings configuration validators."""

from app.fastapi.config import Settings


class TestParseCorsOrigins:
    """Tests for the parse_cors_origins field validator."""

    def test_comma_separated_string(self):
        """Verify comma-separated string is split into a list."""
        settings = Settings(cors_origins="http://a.com, http://b.com, http://c.com")
        assert settings.cors_origins == [
            "http://a.com",
            "http://b.com",
            "http://c.com",
        ]

    def test_list_passthrough(self):
        """Verify list input is returned unchanged."""
        origins = ["http://a.com", "http://b.com"]
        settings = Settings(cors_origins=origins)
        assert settings.cors_origins == origins

    def test_single_origin_string(self):
        """Verify single origin string becomes a one-element list."""
        settings = Settings(cors_origins="http://localhost:3000")
        assert settings.cors_origins == ["http://localhost:3000"]
