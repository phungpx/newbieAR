from datetime import datetime
from fastapi import Request, status
from fastapi.responses import JSONResponse
from loguru import logger


class ErrorCode:
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_API_KEY = "INVALID_API_KEY"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


def create_error_response(
    code: str,
    message: str,
    status_code: int,
    details: dict | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """Create standardized error response"""
    error_data = {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id or "unknown",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    }
    return JSONResponse(status_code=status_code, content=error_data)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors"""
    logger.exception(f"Unhandled exception: {exc}")

    return create_error_response(
        code=ErrorCode.INTERNAL_ERROR,
        message="An internal server error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"error": str(exc)} if logger.level == "DEBUG" else {},
        request_id=getattr(request.state, "request_id", None),
    )
