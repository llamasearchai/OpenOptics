import logging
from typing import Dict, Any, Optional
import uuid
import hashlib
import os
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AuthHandler:
    """Handles authentication and authorization for OpenOptics"""
    
    def __init__(self):
        """Initialize the auth handler"""
        self.users = {}  # In a real implementation, this would be a database
        self.sessions = {}
        self.token_expiry = 3600  # 1 hour
        logger.info("AuthHandler initialized")
        
        # Add a default user for testing
        self._add_default_user()
    
    def _add_default_user(self):
        """Add a default admin user"""
        self.users["admin"] = {
            "user_id": "admin",
            "password_hash": self._hash_password("admin123"),
            "role": "admin",
            "created_at": datetime.now().isoformat()
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using a secure method"""
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt, 
            100000
        )
        return salt.hex() + ':' + key.hex()
    
    def _verify_password(self, stored_hash: str, provided_password: str) -> bool:
        """Verify a password against a stored hash"""
        salt_hex, key_hex = stored_hash.split(':')
        salt = bytes.fromhex(salt_hex)
        stored_key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac(
            'sha256', 
            provided_password.encode('utf-8'), 
            salt, 
            100000
        )
        return stored_key == new_key
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password"""
        logger.info(f"Authentication attempt for user: {username}")
        
        if username not in self.users:
            logger.warning(f"User not found: {username}")
            return None
        
        user = self.users[username]
        
        if not self._verify_password(user["password_hash"], password):
            logger.warning(f"Invalid password for user: {username}")
            return None
        
        # Create a session
        session_id = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(seconds=self.token_expiry)
        
        self.sessions[session_id] = {
            "user_id": username,
            "expiry": expiry.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Authentication successful for user: {username}")
        
        return {
            "session_id": session_id,
            "user": {
                "user_id": username,
                "role": user["role"]
            },
            "expiry": expiry.isoformat()
        }
    
    def get_current_user(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current user from a session ID"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        expiry = datetime.fromisoformat(session["expiry"])
        
        if datetime.now() > expiry:
            # Session expired
            del self.sessions[session_id]
            return None
        
        username = session["user_id"]
        if username not in self.users:
            return None
        
        return {
            "user_id": username,
            "role": self.users[username]["role"]
        }
    
    def logout(self, session_id: str) -> bool:
        """Log out a user by invalidating their session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Logged out session: {session_id}")
            return True
        return False
    
    def register_user(self, username: str, password: str, role: str = "user") -> Dict[str, Any]:
        """Register a new user"""
        if username in self.users:
            raise ValueError(f"User already exists: {username}")
        
        self.users[username] = {
            "user_id": username,
            "password_hash": self._hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Registered new user: {username} with role: {role}")
        
        return {
            "user_id": username,
            "role": role,
            "created_at": self.users[username]["created_at"]
        }
    
    def update_user_role(self, username: str, new_role: str) -> bool:
        """Update a user's role"""
        if username not in self.users:
            return False
        
        self.users[username]["role"] = new_role
        logger.info(f"Updated role for user {username} to {new_role}")
        return True
    
    def delete_user(self, username: str) -> bool:
        """Delete a user"""
        if username not in self.users:
            return False
        
        del self.users[username]
        
        # Also delete any active sessions for this user
        session_ids_to_delete = []
        for session_id, session in self.sessions.items():
            if session["user_id"] == username:
                session_ids_to_delete.append(session_id)
        
        for session_id in session_ids_to_delete:
            del self.sessions[session_id]
        
        logger.info(f"Deleted user: {username}")
        return True
    
    def check_permission(self, user: Dict[str, Any], required_role: str) -> bool:
        """Check if a user has a required role"""
        role_hierarchy = {
            "admin": 3,
            "manager": 2,
            "user": 1,
            "guest": 0
        }
        
        user_role = user.get("role", "guest")
        
        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)

# Create a singleton instance
auth_handler = AuthHandler()

def get_auth_handler() -> AuthHandler:
    """Get the AuthHandler instance"""
    return auth_handler

# Other authentication related functions could go here, e.g.:
# - create_access_token
# - password hashing utilities
# - OAuth2 password bearer scheme setup 