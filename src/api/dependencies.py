from fastapi import Header, HTTPException, status, Depends
from src.api.services import auth_service
from src.api.models import APIKey
from src.api.middleware.error_handler import ErrorCode, create_error_response


async def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> APIKey:
    """Validate API key from header"""
    api_key = auth_service.validate_key(x_api_key)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    return api_key


async def require_permission(permission: str):
    """Factory for permission-checking dependencies"""
    async def check_permission(api_key: APIKey = Depends(get_api_key)) -> APIKey:
        if not auth_service.has_permission(api_key, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}",
            )
        return api_key

    return check_permission


# Specific permission dependencies
async def require_ingest_permission(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    if not auth_service.has_permission(api_key, "ingest"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Required: ingest",
        )
    return api_key


async def require_retrieval_permission(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    if not auth_service.has_permission(api_key, "retrieval"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Required: retrieval",
        )
    return api_key


async def require_agents_permission(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    if not auth_service.has_permission(api_key, "agents"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Required: agents",
        )
    return api_key
