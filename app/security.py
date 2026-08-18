from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials

from app.config import get_settings

security = HTTPBearer()


async def require_api_key(credentials: HTTPAuthCredentials = Depends(security)) -> None:
    """Validate API key from X-API-Key header."""
    settings = get_settings()
    if credentials.credentials != settings.app_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
