"""
FastAPI configuration using Pydantic Settings.

https://fastapi.tiangolo.com/advanced/settings/#run-the-server
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with automatic environment variable loading.

    Environment variables are automatically loaded with the same name
    (case-insensitive). For example, SECRET_KEY env var maps to secret_key.
    """

    # Application settings
    app_name: str = "Percentile Grids API"
    app_version: str = "0.0.1"
    debug: bool = True
    environment: str = "development"  # development, staging, production

    # Database settings
    db_url: str = "sqlite:///grids_database.db"

    # Security
    secret_key: str = "your-secret-key-change-in-production"

    # CORS settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5000"]
    cors_allow_credentials: bool = True

    # File upload settings
    max_upload_size: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: List[str] = ["csv", "xlsx", "xls"]

    # Cache settings
    persistent_cache_dir: str = "./cache"
    persistent_cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds

    # R environment
    r_home: Optional[str] = None
    r_libs: Optional[str] = None

    # Redis settings (for production)
    redis_url: Optional[str] = None
    redis_ttl: int = 86400  # 24 hours

    # Worker settings
    background_task_timeout: int = 600  # 10 minutes
    max_workers: int = 4  # For thread pool executor

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins from environment variable."""
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v


@lru_cache()
def get_settings():
    """
    Create cached settings instance.
    Use this function to get settings throughout the app.
    """
    return Settings()
