"""
FastAPI configuration using Pydantic Settings.

https://fastapi.tiangolo.com/advanced/settings/#run-the-server
"""

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
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
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        validation_alias=AliasChoices("SECRET_KEY", "SECRET_KEY_DEV"),
    )
    access_token_expire_minutes: int = 30

    # CORS settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5000"]
    cors_allow_credentials: bool = True

    # File upload settings
    max_upload_size: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: list[str] = ["csv", "xlsx", "xls"]
    max_files_count: int = 300
    upload_folder: str = "./uploads"

    # Model storage
    models_dir: str = "./models"

    # R environment
    r_home: str | None = None
    r_libs: str | None = None

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """
        Parse comma-separated CORS origins from environment variable.

        Parameters
        ----------
        v : str | list[str]
            Either a comma-separated string or a list of origins.

        Returns
        -------
        list[str]
            List of CORS origin URLs.
        """
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Create cached settings instance.

    Use this function to get settings throughout the app.
    The result is cached using lru_cache for performance.

    Returns
    -------
    Settings
        The application settings instance.
    """
    return Settings()
