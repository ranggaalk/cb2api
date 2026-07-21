"""
Configuration management for CodeBuddy2API

Implements a multi-layered configuration system with hot-reloading.
Priority order:
1. In-memory config (for hot-settings from the UI)
2. config.json file (for persisted user overrides)
3. Environment variables (for deployment, e.g., Docker)
4. Hard-coded defaults
"""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- Private State ---
_config_cache: Dict[str, Any] = {}
_CONFIG_JSON_PATH = 'config/config.json'  # Use a path inside a directory

_MASKED_SECRET_SENTINEL = "********"
_SECRET_CONFIG_KEYS = {"CODEBUDDY_PASSWORD", "CODEBUDDY_ADMIN_PASSWORD"}

_DEFAULT_CONFIG = {
    "CODEBUDDY_HOST": "127.0.0.1",
    "CODEBUDDY_PORT": 8001,
    "CODEBUDDY_PASSWORD": None,
    "CODEBUDDY_API_ENDPOINT": "https://www.codebuddy.ai",
    "CODEBUDDY_CREDS_DIR": ".codebuddy_creds",
    "CODEBUDDY_LOG_LEVEL": "INFO",
    "CODEBUDDY_MODELS": "claude-4.0,claude-3.7,gpt-5,gpt-5-mini,gpt-5-nano,o4-mini,gemini-2.5-flash,gemini-2.5-pro,auto-chat",
    "CODEBUDDY_ROTATION_COUNT": 1,
    "CODEBUDDY_AUTH_MODE": "auto",
    "CODEBUDDY_API_KEYS_FILE": "./config/codebuddy_api_keys.txt",
    "CODEBUDDY_API_KEY_ROTATION": "round_robin",
    "CODEBUDDY_API_KEY_RELOAD_INTERVAL": 5,
    "CODEBUDDY_API_KEY_COOLDOWN_SECONDS": 300,
    "CODEBUDDY_CLIENT_AUTH_MODE": "relay",
    "CODEBUDDY_ADMIN_PASSWORD": None,
    "CODEBUDDY_UPSTREAM_API_KEY_HEADER": "bearer",
    "CODEBUDDY_REQUEST_PROFILE": "web",
    "CODEBUDDY_SANITIZE_AGENT_PROMPT": True,
    "CODEBUDDY_MAX_SYSTEM_PROMPT_LENGTH": 2000,
    "CODEBUDDY_MODEL_ALIASES": "",
    "CODEBUDDY_DEFAULT_MODEL": "auto-chat",
    "CODEBUDDY_UNKNOWN_MODEL_POLICY": "passthrough",
    # --- Adapter V2 ---
    # Which request pipeline handles /v1/chat/completions: "v2" (the isolated
    # CodeBuddyAdapterV2) or "legacy" (the original monolithic handler).
    "CODEBUDDY_ADAPTER_VERSION": "v2",
    # Granular upstream timeouts (seconds). Each stage is distinguished so a
    # failure can be reported precisely instead of a generic "connect timeout".
    "CODEBUDDY_CONNECT_TIMEOUT_SECONDS": 30,
    "CODEBUDDY_POOL_TIMEOUT_SECONDS": 30,
    "CODEBUDDY_WRITE_TIMEOUT_SECONDS": 60,
    "CODEBUDDY_HEADERS_TIMEOUT_SECONDS": 300,
    "CODEBUDDY_FIRST_CHUNK_TIMEOUT_SECONDS": 300,
    "CODEBUDDY_STREAM_IDLE_TIMEOUT_SECONDS": 600,
    # Upstream concurrency control.
    "CODEBUDDY_MAX_CONCURRENT_UPSTREAM_REQUESTS": 20,
    "CODEBUDDY_UPSTREAM_QUEUE_TIMEOUT_SECONDS": 60,
    # Large-agentic-request warning thresholds (never used to truncate).
    "CODEBUDDY_WARN_TOTAL_CONTENT_LENGTH": 50000,
    "CODEBUDDY_WARN_MESSAGE_COUNT": 40,
    "CODEBUDDY_WARN_TOOL_COUNT": 30,
}

# --- Core Functions ---

def load_config():
    """
    Loads configuration from all sources into the in-memory cache.
    This should be called once at application startup.
    """
    global _config_cache
    
    config = _DEFAULT_CONFIG.copy()
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded environment variables from .env file.")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading.")

    for key in config:
        env_value = os.getenv(key)
        if env_value is not None:
            config[key] = env_value
            
    if os.path.exists(_CONFIG_JSON_PATH):
        try:
            with open(_CONFIG_JSON_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    persisted_config = json.loads(content)
                    config.update(persisted_config)
                    logger.info(f"Loaded and merged persisted settings from {_CONFIG_JSON_PATH}.")
        except Exception as e:
            logger.error(f"Error loading {_CONFIG_JSON_PATH}: {e}")

    _config_cache = config
    logger.info("Configuration loaded successfully.")


def _get_config_value(key: str) -> Any:
    return _config_cache.get(key, _DEFAULT_CONFIG.get(key))

def _update_config_value(key: str, value: Any):
    global _config_cache
    _config_cache[key] = value
    # Downgrade to debug to avoid verbose logging in production
    logger.debug(f"Hot-reloaded setting '{key}' to new value.")


def save_config_to_json():
    """
    Saves the entire current in-memory configuration to config.json.
    This is simpler and more robust, ensuring a complete snapshot is always saved.
    This will create the file if it doesn't exist.
    """
    try:
        # Ensure the directory exists before writing the file
        config_dir = os.path.dirname(_CONFIG_JSON_PATH)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            logger.info(f"Created config directory at {config_dir}")

        with open(_CONFIG_JSON_PATH, 'w', encoding='utf-8') as f:
            # Only save keys that are part of the original default config
            # to avoid saving runtime-only variables.
            config_to_save = {key: _config_cache.get(key) for key in _DEFAULT_CONFIG}
            json.dump(config_to_save, f, indent=4)
        logger.info(f"Settings successfully persisted to {_CONFIG_JSON_PATH}.")
    except Exception as e:
        logger.error(f"Failed to save config to {_CONFIG_JSON_PATH}: {e}")
        raise

# --- Public Getter Functions ---

def get_active_config() -> Dict[str, Any]:
    config = {key: _config_cache.get(key) for key in _DEFAULT_CONFIG}
    for key in _SECRET_CONFIG_KEYS:
        if config.get(key):
            config[key] = _MASKED_SECRET_SENTINEL
    return config

def get_server_host() -> str:
    return str(_get_config_value("CODEBUDDY_HOST"))

def get_server_port() -> int:
    return int(_get_config_value("CODEBUDDY_PORT"))

def get_server_password() -> Optional[str]:
    return _get_config_value("CODEBUDDY_PASSWORD")


def get_admin_password() -> Optional[str]:
    return _get_config_value("CODEBUDDY_ADMIN_PASSWORD") or get_server_password()


def get_client_auth_mode() -> str:
    mode = str(_get_config_value("CODEBUDDY_CLIENT_AUTH_MODE")).strip().lower()
    if mode not in {"relay", "passthrough", "hybrid"}:
        raise ValueError(
            "CODEBUDDY_CLIENT_AUTH_MODE must be relay, passthrough, or hybrid"
        )
    return mode


def get_upstream_api_key_header() -> str:
    mode = str(_get_config_value("CODEBUDDY_UPSTREAM_API_KEY_HEADER")).strip().lower()
    if mode not in {"x-api-key", "bearer", "both"}:
        raise ValueError(
            "CODEBUDDY_UPSTREAM_API_KEY_HEADER must be x-api-key, bearer, or both"
        )
    return mode

def get_codebuddy_request_profile() -> str:
    profile = str(_get_config_value("CODEBUDDY_REQUEST_PROFILE")).strip().lower()
    if profile not in {"web", "cli"}:
        raise ValueError("CODEBUDDY_REQUEST_PROFILE must be web or cli")
    return profile


def get_codebuddy_default_model() -> str:
    value = str(_get_config_value("CODEBUDDY_DEFAULT_MODEL")).strip()
    return value or "auto-chat"


def get_codebuddy_unknown_model_policy() -> str:
    """How to handle a requested model that is neither a known upstream model
    ID nor a configured alias.

      * passthrough (default): forward the requested model verbatim and let
        CodeBuddy accept or reject it. Never silently rewrite it.
      * reject: return HTTP 400 (code=unknown_model) without calling upstream.
      * default: fall back to CODEBUDDY_DEFAULT_MODEL.
    """
    policy = str(_get_config_value("CODEBUDDY_UNKNOWN_MODEL_POLICY")).strip().lower()
    if policy not in {"passthrough", "reject", "default"}:
        raise ValueError(
            "CODEBUDDY_UNKNOWN_MODEL_POLICY must be passthrough, reject, or default"
        )
    return policy


def get_codebuddy_model_aliases() -> Dict[str, str]:
    """Parse CODEBUDDY_MODEL_ALIASES into a lower-cased alias -> upstream map.

    Format: comma-separated ``alias=upstream`` pairs, e.g.
    ``Claude Opus 4.7=claude-4.0,gpt-5-ui=gpt-5``. Aliases are matched
    case-insensitively; upstream IDs are preserved verbatim.
    """
    raw = _get_config_value("CODEBUDDY_MODEL_ALIASES")
    aliases: Dict[str, str] = {}
    if not raw:
        return aliases
    for pair in str(raw).split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        alias, upstream = pair.split("=", 1)
        alias = alias.strip().lower()
        upstream = upstream.strip()
        if alias and upstream:
            aliases[alias] = upstream
    return aliases


def get_codebuddy_adapter_version() -> str:
    """Which chat-completions pipeline handles the request.

      * ``v2`` (default): the isolated :class:`CodeBuddyAdapterV2`.
      * ``legacy``: the original monolithic handler in ``codebuddy_router``.

    An unrecognized value falls back to ``v2`` rather than raising, so a typo
    never takes the service down; the effective value is logged by the router.
    """
    version = str(_get_config_value("CODEBUDDY_ADAPTER_VERSION")).strip().lower()
    return version if version in {"v2", "legacy"} else "v2"


def _get_positive_int(key: str, default: int) -> int:
    """Return a strictly-positive int config value, else ``default``."""
    try:
        value = int(_get_config_value(key))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def get_codebuddy_connect_timeout_seconds() -> float:
    return float(_get_positive_int("CODEBUDDY_CONNECT_TIMEOUT_SECONDS", 30))


def get_codebuddy_pool_timeout_seconds() -> float:
    return float(_get_positive_int("CODEBUDDY_POOL_TIMEOUT_SECONDS", 30))


def get_codebuddy_write_timeout_seconds() -> float:
    return float(_get_positive_int("CODEBUDDY_WRITE_TIMEOUT_SECONDS", 60))


def get_codebuddy_headers_timeout_seconds() -> float:
    return float(_get_positive_int("CODEBUDDY_HEADERS_TIMEOUT_SECONDS", 300))


def get_codebuddy_first_chunk_timeout_seconds() -> float:
    return float(_get_positive_int("CODEBUDDY_FIRST_CHUNK_TIMEOUT_SECONDS", 300))


def get_codebuddy_stream_idle_timeout_seconds() -> float:
    return float(_get_positive_int("CODEBUDDY_STREAM_IDLE_TIMEOUT_SECONDS", 600))


def get_codebuddy_max_concurrent_upstream_requests() -> int:
    return _get_positive_int("CODEBUDDY_MAX_CONCURRENT_UPSTREAM_REQUESTS", 20)


def get_codebuddy_upstream_queue_timeout_seconds() -> float:
    return float(_get_positive_int("CODEBUDDY_UPSTREAM_QUEUE_TIMEOUT_SECONDS", 60))


def get_codebuddy_warn_total_content_length() -> int:
    return _get_positive_int("CODEBUDDY_WARN_TOTAL_CONTENT_LENGTH", 50000)


def get_codebuddy_warn_message_count() -> int:
    return _get_positive_int("CODEBUDDY_WARN_MESSAGE_COUNT", 40)


def get_codebuddy_warn_tool_count() -> int:
    return _get_positive_int("CODEBUDDY_WARN_TOOL_COUNT", 30)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "t", "y", "yes"}


def get_sanitize_agent_prompt() -> bool:
    return _coerce_bool(_get_config_value("CODEBUDDY_SANITIZE_AGENT_PROMPT"), True)


def get_max_system_prompt_length() -> int:
    try:
        value = int(_get_config_value("CODEBUDDY_MAX_SYSTEM_PROMPT_LENGTH"))
    except (TypeError, ValueError):
        return 2000
    return value if value > 0 else 2000


def get_codebuddy_api_endpoint() -> str:
    return str(_get_config_value("CODEBUDDY_API_ENDPOINT"))

def get_codebuddy_creds_dir() -> str:
    return str(_get_config_value("CODEBUDDY_CREDS_DIR"))

def get_log_level() -> str:
    return str(_get_config_value("CODEBUDDY_LOG_LEVEL")).upper()

def get_available_models() -> list:
    models_str = str(_get_config_value("CODEBUDDY_MODELS"))
    return [model.strip() for model in models_str.split(",")]

def get_rotation_count() -> int:
    return int(_get_config_value("CODEBUDDY_ROTATION_COUNT"))


def get_codebuddy_auth_mode() -> str:
    mode = str(_get_config_value("CODEBUDDY_AUTH_MODE")).strip().lower()
    if mode not in {"auto", "api_key_file", "credentials"}:
        raise ValueError("CODEBUDDY_AUTH_MODE must be auto, api_key_file, or credentials")
    return mode


def get_codebuddy_api_keys_file() -> str:
    return str(_get_config_value("CODEBUDDY_API_KEYS_FILE")).strip()


def get_codebuddy_api_key_rotation() -> str:
    rotation = str(_get_config_value("CODEBUDDY_API_KEY_ROTATION")).strip().lower()
    if rotation != "round_robin":
        raise ValueError("CODEBUDDY_API_KEY_ROTATION must be round_robin")
    return rotation


def get_codebuddy_api_key_reload_interval() -> int:
    interval = int(_get_config_value("CODEBUDDY_API_KEY_RELOAD_INTERVAL"))
    if interval < 0:
        raise ValueError("CODEBUDDY_API_KEY_RELOAD_INTERVAL must be zero or greater")
    return interval


def get_codebuddy_api_key_cooldown_seconds() -> int:
    cooldown = int(_get_config_value("CODEBUDDY_API_KEY_COOLDOWN_SECONDS"))
    if cooldown < 0:
        raise ValueError("CODEBUDDY_API_KEY_COOLDOWN_SECONDS must be zero or greater")
    return cooldown

# --- Public Setter for Hot-Reload ---

def update_settings(new_settings: Dict[str, Any]):
    """Updates the live config and persists it to config.json."""
    for key, value in new_settings.items():
        if key in _config_cache:
            if key in _SECRET_CONFIG_KEYS and value == _MASKED_SECRET_SENTINEL:
                continue
            default_value = _DEFAULT_CONFIG.get(key, value)
            original_type = type(default_value)
            try:
                if default_value is None:
                    typed_value = value or None
                elif original_type is bool:
                    typed_value = str(value).lower() in ('true', '1', 't', 'y', 'yes')
                else:
                    typed_value = original_type(value)
                _update_config_value(key, typed_value)
            except (ValueError, TypeError):
                logger.warning(f"Could not cast new value for '{key}' to {original_type}. Using as string.")
                _update_config_value(key, value)
    
    save_config_to_json()

# --- Initial Load ---
load_config()
