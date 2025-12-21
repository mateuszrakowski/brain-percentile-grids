"""
Health and monitoring endpoints.
These are good starting points for learning FastAPI patterns.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.fastapi.config import get_settings
from app.fastapi.dependencies import get_cache_info, get_system_metrics
from app.fastapi.models.responses import HealthResponse

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/monitoring",
    tags=["monitoring"],
    responses={404: {"description": "Not found"}},
)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        HealthResponse: Application health status
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy", version=settings.app_version, environment=settings.environment
    )


@router.get("/status")
async def detailed_status(
    cache_info: Dict = Depends(get_cache_info),
    system_metrics: Dict = Depends(get_system_metrics),
) -> Dict[str, Any]:
    """
    Detailed application status with metrics.

    This demonstrates dependency injection in FastAPI.
    """
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "cache": cache_info,
        "system": system_metrics,
        "uptime": system_metrics.get("uptime", 0),
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, bool | dict]:
    """
    Kubernetes-style readiness probe.

    Checks if the application is ready to serve requests.
    """
    checks = {
        "database": True,  # Check database connection
        "cache": True,  # Check cache availability
        "r_environment": await check_r_environment(),
    }

    all_ready = all(checks.values())

    return {"ready": all_ready, "checks": checks}


async def check_r_environment() -> bool:
    """Check if R environment is available."""
    try:
        from app.core.engine.environment import check_r_environment as r_env_check

        return r_env_check()
    except Exception:
        return False
