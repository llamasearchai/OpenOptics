from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, Optional, List
import base64
import logging

from ..security.quantum_resistant import QuantumResistantSecurity
from ..auth.auth_handler import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/security",
    tags=["security"],
    responses={404: {"description": "Not found"}},
)

# Dependency to get quantum-resistant security manager
async def get_qr_security():
    # In a real implementation, this would be a singleton instance
    # For now, we'll create a new instance for demo purposes
    from ..core.app_context import get_app_context
    context = await get_app_context()
    return context.security_manager.quantum_resistant

@router.get("/quantum-resistant/status")
async def get_qr_security_status(
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Get status of quantum-resistant security features"""
    if current_user["role"] not in ["admin", "security_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    result = await qr_security.get_security_status()
    return result

@router.post("/quantum-resistant/generate-key-pair")
async def generate_key_pair(
    algorithm: Optional[str] = "default",
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Generate a quantum-resistant key pair"""
    if current_user["role"] not in ["admin", "security_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    result = await qr_security.generate_key_pair(algorithm)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@router.post("/quantum-resistant/initiate-key-exchange")
async def initiate_key_exchange(
    peer_id: str,
    algorithm: Optional[str] = "default",
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Initiate a quantum-resistant key exchange with a peer"""
    if current_user["role"] not in ["admin", "security_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    result = await qr_security.initiate_key_exchange(peer_id, algorithm)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@router.post("/quantum-resistant/complete-key-exchange")
async def complete_key_exchange(
    session_id: str,
    peer_public_key: str,
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Complete a quantum-resistant key exchange with a peer"""
    if current_user["role"] not in ["admin", "security_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    result = await qr_security.complete_key_exchange(session_id, peer_public_key)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@router.post("/quantum-resistant/encrypt")
async def encrypt_data(
    data: bytes = Body(...),
    key_id: Optional[str] = None,
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Encrypt data using quantum-resistant encryption"""
    if current_user["role"] not in ["admin", "security_engineer", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Base64 encode the input data for the API request
    data_base64 = base64.b64encode(data).decode('ascii')
    
    result = await qr_security.encrypt_data(data_base64, key_id)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@router.post("/quantum-resistant/decrypt")
async def decrypt_data(
    ciphertext: str,
    nonce: str,
    key_id: str,
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Decrypt data using quantum-resistant encryption"""
    if current_user["role"] not in ["admin", "security_engineer", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    result = await qr_security.decrypt_data(ciphertext, nonce, key_id)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["message"])
    
    # Return plaintext as base64 encoded to ensure it can be properly transmitted in JSON
    return {
        "status": "success",
        "plaintext_base64": base64.b64encode(result["plaintext"]).decode('ascii')
    }

@router.post("/quantum-resistant/sign")
async def sign_data(
    data: bytes = Body(...),
    key_id: Optional[str] = None,
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Sign data using quantum-resistant digital signature"""
    if current_user["role"] not in ["admin", "security_engineer", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Base64 encode the input data for the API request
    data_base64 = base64.b64encode(data).decode('ascii')
    
    result = await qr_security.sign_data(data_base64, key_id)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@router.post("/quantum-resistant/verify")
async def verify_signature(
    data: bytes = Body(...),
    signature: str = Body(...),
    key_id: str = Body(...),
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Verify a quantum-resistant digital signature"""
    if current_user["role"] not in ["admin", "security_engineer", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Base64 encode the input data for the API request
    data_base64 = base64.b64encode(data).decode('ascii')
    
    result = await qr_security.verify_signature(data_base64, signature, key_id)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result

@router.post("/quantum-resistant/secure-channel/{device_id}")
async def secure_control_channel_route(
    device_id: str,
    # Allow optional body for peer-initiated flow, where peer sends its public key or KEM ciphertext
    payload: Optional[Dict[str, str]] = Body(None), # e.g., {"peer_public_key_b64": "..."} or {"kem_ciphertext_b64": "..."}
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Establish or continue a quantum-resistant secure channel with a device."""
    if current_user["role"] not in ["admin", "security_engineer", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    peer_public_key_or_ct_b64 = None
    if payload:
        peer_public_key_or_ct_b64 = payload.get("peer_public_key_b64") or payload.get("kem_ciphertext_b64")

    try:
        result = await qr_security.establish_secure_channel(device_id, peer_public_key_or_ct_b64)
        if result.get("status") != "success":
            # Determine appropriate status code based on error type if possible
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to establish secure channel"))
        return result
    except Exception as e:
        logger.exception(f"Error in secure_control_channel for device {device_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error establishing secure channel: {str(e)}")