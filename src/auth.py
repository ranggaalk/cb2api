"""
Client and admin authentication for CodeBuddy2API.
"""
import secrets
from dataclasses import dataclass, field
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_admin_password, get_client_auth_mode, get_server_password

security = HTTPBearer(auto_error=False)


@dataclass(repr=False)
class ClientAuthContext:
    """Request-scoped inference authentication result."""

    mode: str
    passthrough_key: Optional[str] = field(default=None, repr=False)


def _extract_bearer(
    credentials: Optional[HTTPAuthorizationCredentials],
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authorization header must use a non-empty Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials.strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authorization header must use a non-empty Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def _matches(token: str, expected: Optional[str]) -> bool:
    return bool(expected) and secrets.compare_digest(token, expected)


def authenticate_inference(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> ClientAuthContext:
    """Resolve relay/passthrough/hybrid inference authentication."""
    token = _extract_bearer(credentials)
    try:
        mode = get_client_auth_mode()
    except ValueError as exc:
        raise HTTPException(
            status_code=500, detail="Client authentication mode is misconfigured"
        ) from exc

    relay_password = get_server_password()
    if mode == "relay":
        if not relay_password:
            raise HTTPException(
                status_code=500,
                detail="CODEBUDDY_PASSWORD is not configured on the server.",
            )
        if not _matches(token, relay_password):
            raise HTTPException(status_code=403, detail="Invalid relay password")
        return ClientAuthContext(mode="relay")

    if mode == "hybrid" and _matches(token, relay_password):
        return ClientAuthContext(mode="relay")

    if mode == "hybrid" and not relay_password:
        raise HTTPException(
            status_code=500,
            detail="CODEBUDDY_PASSWORD is required when hybrid mode is enabled",
        )

    return ClientAuthContext(mode="passthrough", passthrough_key=token)


def authenticate_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """Authenticate dashboard and management endpoints only."""
    token = _extract_bearer(credentials)
    password = get_admin_password()
    if not password:
        raise HTTPException(
            status_code=500,
            detail="CODEBUDDY_ADMIN_PASSWORD or CODEBUDDY_PASSWORD is required",
        )
    if not _matches(token, password):
        raise HTTPException(status_code=403, detail="Invalid admin password")
    return token


# Backward-compatible import name for existing admin routes.
authenticate = authenticate_admin
