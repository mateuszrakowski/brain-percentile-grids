"""
FastAPI application for Percentile Grids.
This is the main entry point that sets up the app, middleware, and routers.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import rpy2.rinterface as rinterface
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.fastapi.models.responses import ErrorResponse, HealthResponse

from .config import get_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    This context manager runs on startup and shutdown, initializing
    the R environment, upload folder, and database.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Yields
    ------
    None
        Control is yielded to the application during its lifetime.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize R environment - required for core functionality
    from app.core.engine.environment import REnvironment, REnvironmentError

    try:
        REnvironment()
        logger.info("R environment initialized successfully")
    except REnvironmentError as e:
        logger.error(f"R environment initialization failed: {e}")
        raise  # Fail fast - R is required for this application

    import os

    os.makedirs(settings.upload_folder, exist_ok=True)
    logger.info(f"Upload folder ready: {settings.upload_folder}")

    # Initialize database
    from app.fastapi.db import models  # noqa: F401 - import to register models
    from app.fastapi.db.database import init_db

    init_db()
    logger.info("Database initialized")

    try:
        yield
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Handle shutdown interrupt - cleanup R before exiting
        logger.info("Received shutdown signal, cleaning up...")
        try:
            rinterface.endr(0)
            logger.info("R environment cleaned up")
        except Exception as cleanup_error:
            logger.debug(f"R cleanup: {cleanup_error}")

    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.app_name,
    description="GAMLSS-based percentile calculations for neuroimaging data",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else "/api/openapi.json",
    lifespan=lifespan,
)

# Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Gzip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time to response headers.

    Parameters
    ----------
    request : Request
        The incoming request.
    call_next : Callable
        The next middleware or route handler.

    Returns
    -------
    Response
        The response with X-Process-Time header added.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Add unique request ID for tracking.

    Parameters
    ----------
    request : Request
        The incoming request.
    call_next : Callable
        The next middleware or route handler.

    Returns
    -------
    Response
        The response with X-Request-ID header added.
    """
    import uuid

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle uncaught exceptions gracefully.

    Parameters
    ----------
    request : Request
        The request that caused the exception.
    exc : Exception
        The exception that was raised.

    Returns
    -------
    JSONResponse
        JSON response with error details (more verbose in debug mode).
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    request_id = getattr(request.state, "request_id", "unknown")

    if settings.debug:
        error_response = ErrorResponse(
            error="Internal server error",
            detail=f"{type(exc).__name__}: {str(exc)}",
            request_id=request_id,
        )
    else:
        error_response = ErrorResponse(
            error="Internal server error",
            request_id=request_id,
        )

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint - API information.

    Returns
    -------
    HTMLResponse
        HTML page with API information and documentation links.
    """
    html_content = f"""
    <html>
        <head>
            <title>{settings.app_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .info {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>{settings.app_name} v{settings.app_version}</h1>
            <div class="info">
                <p>FastAPI-based GAMLSS percentile calculation service</p>
                <p>Environment: <strong>{settings.environment}</strong></p>
                <ul>
                    <li>
                        <a href="/docs">Interactive API Documentation (Swagger UI)</a>
                    </li>
                    <li>
                        <a href="/redoc">Alternative API Documentation (ReDoc)</a>
                    </li>
                    <li><a href="/health">Health Check</a></li>
                </ul>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", tags=["monitoring"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns
    -------
    HealthResponse
        Health status with version and environment information.
    """
    return HealthResponse(
        status="healthy", version=settings.app_version, environment=settings.environment
    )


# Include routers
from .routers import auth, calculations, data, health  # noqa: E402

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(data.router)
app.include_router(calculations.router)

if __name__ == "__main__":
    """Run the application with Uvicorn when executed directly."""
    import uvicorn

    uvicorn.run(
        "fastapi_app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
        access_log=settings.debug,
    )
