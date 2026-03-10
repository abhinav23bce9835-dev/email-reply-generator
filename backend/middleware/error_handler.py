"""Global error handling middleware for FastAPI."""
import logging
import traceback
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global exception handler middleware providing structured error responses."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """
        Process request and handle any unhandled exceptions.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response from the route handler, or a structured error response.
        """
        try:
            response = await call_next(request)
            return response
        except ValueError as exc:
            logger.warning("Validation error on %s: %s", request.url.path, exc)
            return JSONResponse(
                status_code=400,
                content={
                    "status_code": 400,
                    "message": "Validation error",
                    "details": str(exc),
                },
            )
        except Exception as exc:
            logger.error(
                "Unhandled exception on %s: %s\n%s",
                request.url.path,
                exc,
                traceback.format_exc(),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "status_code": 500,
                    "message": "Internal server error",
                    "details": "An unexpected error occurred. Please try again.",
                },
            )
