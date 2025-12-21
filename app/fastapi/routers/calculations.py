"""
Calculation endpoints for GAMLSS modeling.

TODO: Implement calculation endpoints with:
1. User authentication (JWT) - to identify user's data
2. SSE for progress updates - to replace WebSocket
3. Database for user data - to replace session storage
4. Background task queue for long-running calculations
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/calculate", tags=["calculations"])

# TODO: Implement these endpoints once authentication, database, and SSE are set up:
#
# @router.post("/reference")
# - Calculate reference percentiles using GAMLSS for authenticated user
# - Use SSE endpoint for progress updates
# - Store results in database linked to user_id
#
# @router.get("/reference/progress/{task_id}")
# - SSE endpoint for streaming progress updates
# - Returns EventSourceResponse with progress events
#
# @router.post("/patient")
# - Calculate patient percentiles against user's reference models
# - Use SSE for progress updates
# - Return results from database
