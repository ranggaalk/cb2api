"""
CodeBuddy Token Manager - Manages CodeBuddy authentication tokens
"""
import os
import glob
import json
import time
import logging
from typing import Dict, Optional, List, Any
from .usage_stats_manager import usage_stats_manager

logger = logging.getLogger(__name__)


class CodeBuddyTokenManager:
    """CodeBuddy token manager."""
    
    def __init__(self, creds_dir=None):
        if creds_dir is None:
            from config import get_codebuddy_creds_dir, get_rotation_count
            creds_dir = get_codebuddy_creds_dir()
        
        self.creds_dir = os.path.join(os.path.dirname(__file__), '..', creds_dir)
        self.state_file = os.path.join(self.creds_dir, 'manager_state.json')
        self.credentials = []
        self.current_index = 0  # Start from the first credential
        self.usage_count = 0    # Counter for the current credential usage
        self.manual_selected_index = None  # Manually selected credential index.
        self.auto_rotation_enabled = True  # Automatic rotation is enabled by default.
        self.load_all_tokens()
        self.load_state()  # Load the saved state.
    
    def load_all_tokens(self):
        """Load all token files."""
        self.credentials = []
        self.current_index = -1
        
        logger.info(f"Loading CodeBuddy credentials from: {self.creds_dir}")
        
        if not os.path.exists(self.creds_dir):
            os.makedirs(self.creds_dir)
            logger.warning(f"Credentials directory created at {self.creds_dir}. No credentials found.")
            return
        
        token_files = glob.glob(os.path.join(self.creds_dir, '*.json'))
        for file_path in token_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'bearer_token' in data:
                        self.credentials.append({
                            'file_path': file_path,
                            'data': data
                        })
                        logger.info(f"Successfully loaded credential: {os.path.basename(file_path)}")
                    else:
                        logger.warning(f"Skipping invalid credential file (missing bearer_token): {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Failed to load credential file {os.path.basename(file_path)}: {e}")
        
        logger.info(f"Loaded a total of {len(self.credentials)} CodeBuddy credentials.")
    
    def load_state(self):
        """Load manager state."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                # Restore state after validating the saved index.
                saved_manual_index = state.get('manual_selected_index')
                if saved_manual_index is not None and 0 <= saved_manual_index < len(self.credentials):
                    # Verify that the credential file still exists.
                    if saved_manual_index < len(self.credentials):
                        saved_filename = state.get('manual_selected_filename')
                        current_filename = os.path.basename(self.credentials[saved_manual_index]['file_path'])
                        if saved_filename == current_filename:
                            self.manual_selected_index = saved_manual_index
                            self.current_index = saved_manual_index
                            logger.info(f"Restored manual selection: {current_filename} (index: {saved_manual_index})")
                        else:
                            logger.warning(f"Saved credential filename mismatch, ignoring saved selection")
                
                # Restore automatic rotation state.
                self.auto_rotation_enabled = state.get('auto_rotation_enabled', True)
                
                # Restore the current index when no manual selection exists.
                if self.manual_selected_index is None:
                    saved_current_index = state.get('current_index', 0)
                    if 0 <= saved_current_index < len(self.credentials):
                        self.current_index = saved_current_index
                    
                logger.info(f"State loaded: auto_rotation={self.auto_rotation_enabled}, current_index={self.current_index}")
        except Exception as e:
            logger.warning(f"Failed to load manager state: {e}")
    
    def save_state(self):
        """Save manager state."""
        try:
            # Ensure the directory exists.
            if not os.path.exists(self.creds_dir):
                os.makedirs(self.creds_dir)
            
            state = {
                'auto_rotation_enabled': self.auto_rotation_enabled,
                'current_index': self.current_index,
                'manual_selected_index': self.manual_selected_index,
                'manual_selected_filename': None,
                'saved_at': int(time.time())
            }
            
            # Save the manually selected filename for validation.
            if self.manual_selected_index is not None and 0 <= self.manual_selected_index < len(self.credentials):
                state['manual_selected_filename'] = os.path.basename(
                    self.credentials[self.manual_selected_index]['file_path']
                )
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Manager state saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save manager state: {e}")
    
    def is_token_expired(self, credential_data: Dict) -> bool:
        """Check whether a token is expired."""
        try:
            created_at = credential_data.get('created_at')
            expires_in = credential_data.get('expires_in')
            
            if not created_at or not expires_in:
                # Assume the token is valid when expiry data is absent for backward compatibility.
                return False
            
            current_time = int(time.time())
            expiry_time = created_at + expires_in
            
            # Treat the token as expired five minutes early to allow refresh time.
            buffer_time = 300  # Five minutes.
            is_expired = current_time >= (expiry_time - buffer_time)
            
            if is_expired:
                user_id = credential_data.get('user_id', 'unknown')
                logger.warning(f"Token for user {user_id} is expired or will expire soon")
            
            return is_expired
        except Exception as e:
            logger.error(f"Error checking token expiry: {e}")
            return False
    
    def get_next_credential(self) -> Optional[Dict]:
        """Return the next available credential according to rotation and expiry state."""
        from config import get_rotation_count

        if not self.credentials:
            return None
        
        # Filter out expired credentials.
        valid_credentials = []
        for i, cred in enumerate(self.credentials):
            if not self.is_token_expired(cred['data']):
                valid_credentials.append((i, cred))
            else:
                filename = os.path.basename(cred['file_path'])
                logger.warning(f"Skipping expired credential: {filename}")
        
        if not valid_credentials:
            logger.error("No valid (non-expired) credentials available")
            return None
        
        # Reset to the first valid credential when the current index is invalid or expired.
        current_valid_indices = [i for i, _ in valid_credentials]
        if self.current_index not in current_valid_indices:
            self.current_index = current_valid_indices[0]
            self.usage_count = 0
            logger.info(f"Reset to first valid credential index: {self.current_index}")

        rotation_count = get_rotation_count()
        
        # Prefer a manually selected credential when it is not expired.
        if self.manual_selected_index is not None and 0 <= self.manual_selected_index < len(self.credentials):
            manual_cred = self.credentials[self.manual_selected_index]
            if not self.is_token_expired(manual_cred['data']):
                credential_filename = os.path.basename(manual_cred['file_path'])
                usage_stats_manager.record_credential_usage(credential_filename)
                logger.info(f"Using manually selected credential: {credential_filename}")
                return manual_cred['data']
            else:
                logger.warning("Manually selected credential is expired, falling back to automatic rotation")
                self.manual_selected_index = None
        
        # Find the current index among valid credentials.
        try:
            current_valid_position = current_valid_indices.index(self.current_index)
        except ValueError:
            current_valid_position = 0
            self.current_index = current_valid_indices[0]
            self.usage_count = 0
        
        # Rotate only when automatic rotation is enabled and the rotation count is positive.
        should_rotate = self.auto_rotation_enabled and rotation_count > 0
        
        if not should_rotate:
            # Keep using the current credential when rotation is disabled.
            credential = self.credentials[self.current_index]
            credential_filename = os.path.basename(credential['file_path'])
            usage_stats_manager.record_credential_usage(credential_filename)
            if rotation_count == 0:
                logger.info(f"Using fixed credential (rotation count is 0): {credential_filename}")
            else:
                logger.info(f"Using fixed credential (auto rotation disabled): {credential_filename}")
            return credential['data']

        # Automatic rotation logic for a positive rotation count.
        if self.usage_count >= rotation_count:
            # Rotate to the next valid credential.
            next_valid_position = (current_valid_position + 1) % len(valid_credentials)
            self.current_index = current_valid_indices[next_valid_position]
            self.usage_count = 0  # Reset the counter.
            logger.info("Credential rotation triggered.")

        credential = self.credentials[self.current_index]
        self.usage_count += 1
        
        # Record usage stats
        credential_filename = os.path.basename(credential['file_path'])
        usage_stats_manager.record_credential_usage(credential_filename)
        
        logger.info(
            f"Using credential: {credential_filename} "
            f"(Usage: {self.usage_count}/{rotation_count})"
        )
        return credential['data']
    
    def get_all_credentials(self) -> List[Dict]:
        """Return all credentials."""
        return [cred['data'] for cred in self.credentials]
    
    def get_credentials_info(self) -> List[Dict]:
        """Return credential details, including expiry state."""
        credentials_info = []
        for i, cred in enumerate(self.credentials):
            data = cred['data']
            filename = os.path.basename(cred['file_path'])
            
            # Calculate expiry information.
            is_expired = self.is_token_expired(data)
            expires_at = None
            time_remaining = None
            
            if data.get('created_at') and data.get('expires_in'):
                expires_at = data['created_at'] + data['expires_in']
                time_remaining = expires_at - int(time.time())
            
            # Extract user information.
            user_info = data.get('user_info', {})
            
            info = {
                'index': i,
                'filename': filename,
                'user_id': data.get('user_id', 'unknown'),
                'email': user_info.get('email') or data.get('user_id'),
                'name': user_info.get('name'),
                'created_at': data.get('created_at'),
                'expires_in': data.get('expires_in'),
                'expires_at': expires_at,
                'time_remaining': time_remaining,
                'is_expired': is_expired,
                'token_type': data.get('token_type', 'Bearer'),
                'scope': data.get('scope'),
                'domain': data.get('domain'),
                'has_refresh_token': bool(data.get('refresh_token')),
                'session_state': data.get('session_state'),
                'file_path': cred['file_path']
            }
            
            credentials_info.append(info)
        
        return credentials_info
    
    def add_credential(self, bearer_token: str, user_id: str = None, filename: str = None) -> bool:
        """Add a credential using the simplified backward-compatible format."""
        if not filename:
            filename = f"codebuddy_token_{len(self.credentials) + 1}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        credential_data = {
            "bearer_token": bearer_token,
            "user_id": user_id,
            "created_at": int(time.time())
        }
        
        return self.add_credential_with_data(credential_data, filename)
    
    def add_credential_with_data(self, credential_data: Dict[str, Any], filename: str = None) -> bool:
        """Add a credential using the complete data format."""
        if not filename:
            user_id = credential_data.get('user_id', 'unknown')
            timestamp = credential_data.get('created_at', int(time.time()))
            safe_user_id = "".join(c for c in str(user_id) if c.isalnum() or c in "._-")[:20]
            filename = f"codebuddy_{safe_user_id}_{timestamp}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        file_path = os.path.join(self.creds_dir, filename)
        
        # Ensure required fields exist.
        if 'created_at' not in credential_data:
            credential_data['created_at'] = int(time.time())
        
        try:
            # Ensure the directory exists.
            if not os.path.exists(self.creds_dir):
                os.makedirs(self.creds_dir)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(credential_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Added new credential: {filename}")
            self.load_all_tokens()  # Reload credentials.
            return True
        except Exception as e:
            logger.error(f"Failed to save credential: {e}")
            return False

    def delete_credential_by_index(self, index: int) -> bool:
        """Delete the credential file at an index and reload the list."""
        try:
            if not (0 <= index < len(self.credentials)):
                logger.error(f"Invalid credential index for deletion: {index}")
                return False

            file_path = self.credentials[index]['file_path']
            filename = os.path.basename(file_path)

            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted credential file: {filename}")
            else:
                logger.warning(f"Credential file already missing: {filename}")

            # Reload credentials and reset related state.
            self.load_all_tokens()
            # Clear manual selection when the deleted index was selected.
            if self.manual_selected_index is not None and self.manual_selected_index == index:
                self.manual_selected_index = None
                logger.info("Cleared manual selection because deleted credential was selected")
            return True
        except Exception as e:
            logger.error(f"Failed to delete credential at index {index}: {e}")
            return False

    def set_manual_credential(self, index: int) -> bool:
        """Manually select the credential at an index."""
        if 0 <= index < len(self.credentials):
            self.manual_selected_index = index
            self.current_index = index  # Update the current index.
            credential_filename = os.path.basename(self.credentials[index]['file_path'])
            logger.info(f"Manually selected credential: {credential_filename} (index: {index})")
            self.save_state()  # Save state.
            return True
        else:
            logger.error(f"Invalid credential index: {index}")
            return False
    
    def clear_manual_selection(self):
        """Clear manual selection and resume automatic rotation."""
        self.manual_selected_index = None
        logger.info("Cleared manual credential selection, resumed automatic rotation")
        self.save_state()  # Save state.
    
    def enable_auto_rotation(self):
        """Enable automatic rotation."""
        self.auto_rotation_enabled = True
        logger.info("Auto rotation enabled")
    
    def disable_auto_rotation(self):
        """Disable automatic rotation."""
        self.auto_rotation_enabled = False
        logger.info("Auto rotation disabled")
    
    def toggle_auto_rotation(self):
        """Toggle automatic rotation."""
        self.auto_rotation_enabled = not self.auto_rotation_enabled
        status = "enabled" if self.auto_rotation_enabled else "disabled"
        logger.info(f"Auto rotation toggled: {status}")
        self.save_state()  # Save state.
        return self.auto_rotation_enabled
    
    def get_current_credential_info(self) -> Dict:
        """Return information about the current credential."""
        from config import get_rotation_count
        
        if not self.credentials:
            return {"status": "no_credentials"}
        
        rotation_count = get_rotation_count()
        
        if self.manual_selected_index is not None and 0 <= self.manual_selected_index < len(self.credentials):
            credential = self.credentials[self.manual_selected_index]
            return {
                "status": "manual_selected",
                "index": self.manual_selected_index,
                "filename": os.path.basename(credential['file_path']),
                "user_id": credential['data'].get('user_id', 'unknown')
            }
        elif not self.auto_rotation_enabled:
            # Ensure current_index is valid.
            if not (0 <= self.current_index < len(self.credentials)):
                self.current_index = 0
            credential = self.credentials[self.current_index]
            return {
                "status": "auto_rotation_disabled",
                "index": self.current_index,
                "filename": os.path.basename(credential['file_path']),
                "user_id": credential['data'].get('user_id', 'unknown'),
                "rotation_count": rotation_count,
                "auto_rotation_enabled": False
            }
        elif rotation_count == 0:
            # Ensure current_index is valid.
            if not (0 <= self.current_index < len(self.credentials)):
                self.current_index = 0
            credential = self.credentials[self.current_index]
            return {
                "status": "rotation_count_zero",
                "index": self.current_index,
                "filename": os.path.basename(credential['file_path']),
                "user_id": credential['data'].get('user_id', 'unknown'),
                "rotation_count": rotation_count,
                "auto_rotation_enabled": True
            }
        else:
            # Ensure current_index is valid.
            if not (0 <= self.current_index < len(self.credentials)):
                self.current_index = 0
            credential = self.credentials[self.current_index]
            return {
                "status": "auto_rotation",
                "index": self.current_index,
                "filename": os.path.basename(credential['file_path']),
                "user_id": credential['data'].get('user_id', 'unknown'),
                "usage_count": self.usage_count,
                "rotation_count": rotation_count,
                "auto_rotation_enabled": True
            }


# Global token manager instance.
codebuddy_token_manager = CodeBuddyTokenManager()
