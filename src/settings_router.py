"""
Settings Router - For loading and saving environment configurations.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any

from .auth import authenticate_admin
from config import get_active_config, update_settings
from .usage_stats_manager import usage_stats_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# English labels displayed by the settings interface.
SETTING_LABELS = {
    "CODEBUDDY_HOST": "Service host address",
    "CODEBUDDY_PORT": "Service port",
    "CODEBUDDY_PASSWORD": "API service access password",
    "CODEBUDDY_API_ENDPOINT": "Official CodeBuddy API endpoint",
    "CODEBUDDY_CREDS_DIR": "Credential file directory",
    "CODEBUDDY_LOG_LEVEL": "Log level",
    "CODEBUDDY_MODELS": "Available model list (comma-separated)",
    "CODEBUDDY_ROTATION_COUNT": "Credential rotation frequency (requests per credential; 0 disables rotation)",
    "CODEBUDDY_AUTH_MODE": "Upstream authentication mode (auto/api_key_file/credentials)",
    "CODEBUDDY_API_KEYS_FILE": "Upstream API key TXT file",
    "CODEBUDDY_API_KEY_ROTATION": "API key rotation strategy",
    "CODEBUDDY_API_KEY_RELOAD_INTERVAL": "API key file reload interval (seconds)",
    "CODEBUDDY_API_KEY_COOLDOWN_SECONDS": "API key cooldown period (seconds)",
    "CODEBUDDY_CLIENT_AUTH_MODE": "Client authentication mode (relay/passthrough/hybrid)",
    "CODEBUDDY_ADMIN_PASSWORD": "Admin dashboard password (empty falls back to service password)",
    "CODEBUDDY_UPSTREAM_API_KEY_HEADER": "Upstream key header (x-api-key/bearer/both)"
}


class Settings(BaseModel):
    settings: Dict[str, Any]


@router.get("/settings", summary="Get all current active settings and labels")
async def get_settings(_token: str = Depends(authenticate_admin)):
    """Return the active configuration and display labels."""
    try:
        return {
            "settings": get_active_config(),
            "labels": SETTING_LABELS
        }
    except Exception as e:
        logger.error(f"Error retrieving active config: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve settings.")


@router.post("/settings", summary="Save and hot-reload settings")
async def save_settings(new_settings: Settings, _token: str = Depends(authenticate_admin)):
    """Save settings and hot-reload them into memory."""
    try:
        update_settings(new_settings.settings)
        return {"message": "Settings saved and hot-reloaded successfully."}
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        raise HTTPException(status_code=500, detail="Could not save the settings file.")


@router.get("/stats", summary="Get usage statistics")
async def get_usage_stats(_token: str = Depends(authenticate_admin)):
    """Return usage statistics for models and credentials."""
    try:
        return usage_stats_manager.get_stats()
    except Exception as e:
        logger.error(f"Error retrieving usage stats: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve usage statistics.")
