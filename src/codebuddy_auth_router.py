"""
CodeBuddy Authentication Router
Authentication implementation based on the real CodeBuddy API
"""
import secrets
import httpx
import base64
import json
import uuid
import time
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Body, Depends

from .auth import authenticate_admin
import logging

logger = logging.getLogger(__name__)

# --- Constants ---
CODEBUDDY_BASE_URL = 'https://www.codebuddy.ai'
CODEBUDDY_AUTH_TOKEN_ENDPOINT = f'{CODEBUDDY_BASE_URL}/v2/plugin/auth/token'
CODEBUDDY_AUTH_STATE_ENDPOINT = f'{CODEBUDDY_BASE_URL}/v2/plugin/auth/state'
_last_auth_state: Optional[str] = None

# --- Router Setup ---
router = APIRouter()
# --- Helper Functions ---
def generate_auth_state() -> str:
    """Generate the state parameter for CodeBuddy authentication."""
    timestamp = int(time.time())
    random_part = secrets.token_hex(16)
    return f"{random_part}_{timestamp}"

def get_auth_start_headers() -> Dict[str, str]:
    """Generate headers required to start authentication through /state."""
    request_id = str(uuid.uuid4()).replace('-', '')
    return {
        'Host': 'www.codebuddy.ai',
        'Accept': 'application/json, text/plain, */*',
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Connection': 'close',
        'X-Requested-With': 'XMLHttpRequest',
        'X-Domain': 'www.codebuddy.ai',
        'X-No-Authorization': 'true',
        'X-No-User-Id': 'true',
        'X-No-Enterprise-Id': 'true',
        'X-No-Department-Info': 'true',
        'User-Agent': 'CLI/1.0.8 CodeBuddy/1.0.8',
        'X-Product': 'SaaS',
        'X-Request-ID': request_id,
    }

def get_auth_poll_headers() -> Dict[str, str]:
    """Generate headers required to poll authentication through /token."""
    request_id = str(uuid.uuid4()).replace('-', '')
    span_id = secrets.token_hex(8)
    return {
        'Host': 'www.codebuddy.ai',
        'Accept': 'application/json, text/plain, */*',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Connection': 'close',
        'X-Requested-With': 'XMLHttpRequest',
        'X-Request-ID': request_id,
        'b3': f'{request_id}-{span_id}-1-',
        'X-B3-TraceId': request_id,
        'X-B3-ParentSpanId': '',
        'X-B3-SpanId': span_id,
        'X-B3-Sampled': '1',
        'X-No-Authorization': 'true',
        'X-No-User-Id': 'true',
        'X-No-Enterprise-Id': 'true',
        'X-No-Department-Info': 'true',
        'X-Domain': 'www.codebuddy.ai',
        'User-Agent': 'CLI/1.0.8 CodeBuddy/1.0.8',
        'X-Product': 'SaaS',
    }

async def start_codebuddy_auth() -> Dict[str, Any]:
    """Start the CodeBuddy authentication flow."""
    try:
        logger.info("Starting the CodeBuddy authentication flow...")
        
        headers = get_auth_start_headers()
        
        # Call /v2/plugin/auth/state to retrieve the authentication state and URL.
        async with httpx.AsyncClient(verify=False) as client:
            # Add a random nonce so every request is unique and bypasses upstream/intermediary caches.
            nonce = secrets.token_hex(8)
            state_url = f"{CODEBUDDY_AUTH_STATE_ENDPOINT}?platform=CLI&nonce={nonce}"
            payload = {"nonce": nonce}
            
            response = await client.post(state_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0 and result.get('data'):
                    data = result['data']
                    auth_state = data.get('state')
                    auth_url = data.get('authUrl')
                    
                    if auth_state and auth_url:
                        global _last_auth_state
                        if _last_auth_state and auth_state == _last_auth_state:
                            logger.warning("Upstream returned the previous state; trying to retrieve a new state...")
                            try:
                                nonce2 = secrets.token_hex(8)
                                state_url2 = f"{CODEBUDDY_AUTH_STATE_ENDPOINT}?platform=CLI&nonce={nonce2}"
                                payload2 = {"nonce": nonce2}
                                async with httpx.AsyncClient(verify=False) as client2:
                                    response2 = await client2.post(state_url2, json=payload2, headers=headers, timeout=30)
                                if response2.status_code == 200:
                                    result2 = response2.json()
                                    if result2.get('code') == 0 and result2.get('data'):
                                        data2 = result2['data']
                                        ns = data2.get('state')
                                        nu = data2.get('authUrl')
                                        if ns and nu and ns != auth_state:
                                            auth_state = ns
                                            auth_url = nu
                            except Exception:
                                pass
                        token_endpoint = f"{CODEBUDDY_AUTH_TOKEN_ENDPOINT}?state={auth_state}"
                        _last_auth_state = auth_state
                        
                        return {
                            "success": True,
                            "method": "codebuddy_real_auth",
                            "auth_state": auth_state,
                            "verification_uri_complete": auth_url,
                            "verification_uri": CODEBUDDY_BASE_URL,
                            "token_endpoint": token_endpoint,
                            "expires_in": 1800,
                            "interval": 5,
                            "status": "awaiting_login",
                            "instructions": "Click the link to complete CodeBuddy sign-in.",
                            "message": "Sign in to CodeBuddy using the provided link.",
                            "platform": "CLI"
                        }
                        
        return {
            "success": False,
            "error": "auth_start_failed",
            "message": "Unable to start the authentication flow."
        }
        
    except Exception as e:
        logger.error(f"Failed to start CodeBuddy authentication: {e}")
        return {
            "success": False,
            "error": "auth_start_failed", 
            "message": f"Authentication startup failed: {str(e)}"
        }

async def poll_codebuddy_auth_status(auth_state: str) -> Dict[str, Any]:
    """Poll CodeBuddy authentication status."""
    try:
        headers = get_auth_poll_headers()
        url = f"{CODEBUDDY_AUTH_TOKEN_ENDPOINT}?state={auth_state}"
        
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('code') == 11217:
                    # Sign-in is still pending.
                    return {
                        "status": "pending",
                        "message": result.get('msg', 'login ing...'),
                        "code": result.get('code')
                    }
                elif result.get('code') == 0 and result.get('data') and result.get('data', {}).get('accessToken'):
                    # Authentication succeeded and returned a token.
                    data = result.get('data', {})
                    return {
                        "status": "success",
                        "message": "Authentication succeeded.",
                        "token_data": {
                            "access_token": data.get('accessToken'),
                            "bearer_token": data.get('accessToken'),
                            "token_type": data.get('tokenType', 'Bearer'),
                            "expires_in": data.get('expiresIn'),
                            "refresh_token": data.get('refreshToken'),
                            "session_state": data.get('sessionState'),
                            "scope": data.get('scope'),
                            "domain": data.get('domain'),
                            "full_response": result
                        }
                    }
                else:
                    # Handle other status codes.
                    return {
                        "status": "unknown",
                        "message": result.get('msg', 'Unknown status'),
                        "code": result.get('code'),
                        "response": result
                    }
            else:
                return {
                    "status": "error",
                    "message": f"API request failed with status: {response.status_code}",
                    "response_text": response.text
                }
                
    except Exception as e:
        logger.error(f"Failed to poll authentication status: {e}")
        return {
            "status": "error",
            "message": f"Polling failed: {str(e)}"
        }

async def save_codebuddy_token(token_data: Dict[str, Any]) -> bool:
    """Save the CodeBuddy token to a file."""
    try:
        from .codebuddy_token_manager import codebuddy_token_manager
        
        # Add the creation timestamp.
        token_data["created_at"] = int(time.time())
        
        # Parse user information from the JWT.
        bearer_token = token_data.get("access_token") or token_data.get("bearer_token")
        user_id = "unknown"
        user_info = {}
        
        try:
            if bearer_token and '.' in bearer_token:
                # Split the JWT token.
                parts = bearer_token.split('.')
                if len(parts) >= 2:
                    payload_part = parts[1]
                    
                    # Repair Base64 padding.
                    missing_padding = len(payload_part) % 4
                    if missing_padding:
                        payload_part += '=' * (4 - missing_padding)
                    
                    # Decode the JWT payload.
                    try:
                        payload = base64.urlsafe_b64decode(payload_part)
                        jwt_data = json.loads(payload.decode('utf-8'))
                        
                        # Extract user information, preferring email as the identifier.
                        user_id = (jwt_data.get('email') or 
                                 jwt_data.get('preferred_username') or 
                                 jwt_data.get('sub') or 
                                 "unknown")
                        
                        # Preserve complete user information.
                        user_info = {
                            'sub': jwt_data.get('sub'),
                            'email': jwt_data.get('email'),
                            'preferred_username': jwt_data.get('preferred_username'),
                            'name': jwt_data.get('name'),
                            'given_name': jwt_data.get('given_name'),
                            'family_name': jwt_data.get('family_name'),
                            'exp': jwt_data.get('exp'),
                            'iat': jwt_data.get('iat'),
                            'scope': jwt_data.get('scope'),
                            'session_state': jwt_data.get('sid')
                        }
                        
                        # Remove None values.
                        user_info = {k: v for k, v in user_info.items() if v is not None}
                        
                        logger.info(f"Successfully parsed JWT for user: {user_id}")
                        logger.debug(f"JWT user information: {user_info}")
                        
                    except (json.JSONDecodeError, UnicodeDecodeError) as decode_error:
                        logger.warning(f"Failed to decode JWT payload: {decode_error}")
                        user_id = token_data.get('domain', 'unknown')
                else:
                    logger.warning("Invalid JWT format: required sections are missing")
                    user_id = token_data.get('domain', 'unknown')
            else:
                logger.warning("Bearer token is empty or malformed")
                user_id = token_data.get('domain', 'unknown')
                
        except Exception as e:
            logger.error(f"Unexpected JWT parsing error: {e}")
            user_id = token_data.get('domain', 'unknown')
        
        # Build the complete credential data.
        credential_data = {
            "bearer_token": bearer_token,
            "user_id": user_id,
            "created_at": int(time.time()),
            "expires_in": token_data.get('expires_in'),
            "refresh_token": token_data.get('refresh_token'),
            "token_type": token_data.get('token_type', 'Bearer'),
            "scope": token_data.get('scope'),
            "domain": token_data.get('domain'),
            "session_state": token_data.get('session_state'),
            "user_info": user_info,
            "full_response": token_data  # Preserve the complete original response.
        }
        
        # Remove None values to keep the file clean.
        credential_data = {k: v for k, v in credential_data.items() if v is not None}
        
        # Generate a readable filename.
        timestamp = int(time.time())
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "._-")[:20]
        filename = f"codebuddy_{safe_user_id}_{timestamp}.json"
        
        # Save through the token manager.
        success = codebuddy_token_manager.add_credential_with_data(
            credential_data=credential_data,
            filename=filename
        )
        
        if success:
            logger.info(f"Saved CodeBuddy token for user {user_id} in file {filename}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to save CodeBuddy token: {e}")
        return False

# --- API Endpoints ---
@router.get("/auth/start", summary="Start CodeBuddy Authentication")
async def start_device_auth(_token: str = Depends(authenticate_admin)):
    """Start the CodeBuddy authentication flow."""
    try:
        logger.info("Starting the CodeBuddy authentication flow...")
        
        # Try the real CodeBuddy authentication API.
        real_auth_result = await start_codebuddy_auth()
        
        if real_auth_result.get('success'):
            logger.info("The real CodeBuddy authentication API started successfully.")
            return real_auth_result
        else:
            logger.warning(f"The real authentication API failed: {real_auth_result}")
            return real_auth_result
        
    except Exception as e:
        logger.error(f"Unexpected authentication startup error: {e}")
        return {
            "success": False,
            "error": "Unexpected error",
            "message": f"Authentication startup failed: {str(e)}"
        }

@router.post("/auth/poll", summary="Poll for OAuth token")
async def poll_for_token(
    device_code: str = Body(None, embed=True),
    code_verifier: str = Body(None, embed=True),
    auth_state: str = Body(None, embed=True),
    _token: str = Depends(authenticate_admin),
):
    """Poll the CodeBuddy token endpoint."""
    from .codebuddy_token_manager import codebuddy_token_manager
    
    # auth_state indicates the real CodeBuddy authentication flow.
    if auth_state:
        logger.info(f"Polling real CodeBuddy authentication state: {auth_state}")
        poll_result = await poll_codebuddy_auth_status(auth_state)
        
        if poll_result.get('status') == 'success':
            # Authentication succeeded; save the token.
            token_data = poll_result.get('token_data', {})
            if token_data:
                # Extract token information.
                bearer_token = token_data.get('access_token') or token_data.get('bearer_token')
                if bearer_token:
                    # Save the token.
                    token_saved = await save_codebuddy_token(token_data)
                    return JSONResponse(content={
                        "access_token": bearer_token,
                        "token_type": token_data.get('token_type', 'Bearer'),
                        "expires_in": token_data.get('expires_in'),
                        "refresh_token": token_data.get('refresh_token'),
                        "scope": token_data.get('scope'),
                        "saved": token_saved,
                        "message": "Authentication succeeded! 🎉",
                        "user_info": token_data,
                        "domain": token_data.get('domain')
                    }, status_code=200)
                else:
                    return JSONResponse(content={
                        "error": "invalid_token_response",
                        "error_description": "The API response did not contain a token."
                    }, status_code=400)
        elif poll_result.get('status') == 'pending':
            # Authorization is still pending.
            return JSONResponse(content={
                "error": "authorization_pending",
                "error_description": poll_result.get('message', 'Waiting for user sign-in...'),
                "code": poll_result.get('code')
            }, status_code=400)
        else:
            # Error state.
            return JSONResponse(content={
                "error": "auth_error",
                "error_description": poll_result.get('message', 'The authentication flow failed.'),
                "details": poll_result
            }, status_code=400)
    else:
        return JSONResponse(content={
            "error": "missing_parameters",
            "error_description": "Required parameter is missing: auth_state"
        }, status_code=400)

@router.get("/auth/callback", summary="OAuth2 callback endpoint")
async def oauth_callback(code: str = None, state: str = None, error: str = None):
    """OAuth2 callback endpoint."""
    if error:
        return JSONResponse(
            content={"error": error, "error_description": "Authorization was denied or failed."},
            status_code=400
        )
    
    return JSONResponse(
        content={
            "message": "Authorization succeeded. Return to the application.",
            "code": code,
            "state": state
        }
    )
