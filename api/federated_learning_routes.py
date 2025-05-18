from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, Optional, List
import logging
import base64
import uuid
import asyncio
import pickle

from ..auth.auth_handler import get_current_user
from ..federated_learning import FederatedNetworkOptimizer, AggregationServer, FederatedClient
from ..security.quantum_resistant import QuantumResistantSecurity

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/federated-learning",
    tags=["federated_learning"],
    responses={404: {"description": "Not found"}}
)

# Dependency to get federated learning manager
async def get_federated_learning():
    from ..core.app_context import get_app_context
    context = await get_app_context()
    return context.federated_learning

# Dependency to get quantum-resistant security
async def get_qr_security():
    from ..core.app_context import get_app_context
    context = await get_app_context()
    return context.security_manager.quantum_resistant

@router.get("/status")
async def get_federated_learning_status(
    fl_manager = Depends(get_federated_learning),
    current_user: Dict = Depends(get_current_user)
):
    """Get status of federated learning system"""
    if current_user["role"] not in ["admin", "data_scientist", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Basic status information
    status = {
        "is_coordinator": fl_manager.is_coordinator,
        "coordinator_running": False,
        "client_count": len(fl_manager.clients) if hasattr(fl_manager, "clients") else 0,
        "is_client": fl_manager.client_node is not None,
        "client_id": fl_manager.client_node.client_id if fl_manager.client_node else None,
        "aggregation_server": fl_manager.aggregation_server is not None
    }
    
    # Add more status details if this is a coordinator
    if fl_manager.is_coordinator and fl_manager.aggregation_server:
        status["coordinator_running"] = True
        status["current_round"] = fl_manager.aggregation_server.current_round
        
    return status

@router.post("/initialize")
async def initialize_federation(
    config: Dict[str, Any] = Body(...),
    fl_manager = Depends(get_federated_learning),
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Initialize federated learning (as coordinator or client)"""
    if current_user["role"] not in ["admin", "data_scientist"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    role = config.get("role", "client")
    client_id = config.get("client_id", f"client_{str(uuid.uuid4())[:8]}")
    
    # Initialize as coordinator
    if role == "coordinator":
        if not fl_manager.is_coordinator:
            raise HTTPException(status_code=400, detail="This node is not configured as a coordinator")
            
        # Initialize the federation
        await fl_manager.initialize_federation()
        
        # For security, establish a secure channel for each client that connects
        for client in fl_manager.clients:
            # Initiate secure channel
            channel_result = await qr_security.establish_secure_channel(client.client_id)
            
            if channel_result["status"] != "success":
                logger.warning(f"Failed to establish secure channel with client {client.client_id}")
        
        return {
            "status": "initialized",
            "role": "coordinator",
            "client_count": len(fl_manager.clients),
            "secure_channels": [c for c in (await qr_security.list_secure_channels())["channels"]]
        }
    
    # Initialize as client
    elif role == "client":
        coordinator_address = config.get("coordinator_address")
        if not coordinator_address:
            raise HTTPException(status_code=400, detail="Coordinator address is required for client initialization")
        
        # Typically would connect to a remote aggregation server
        # For demo purposes, we'll assume coordinator is local or pre-configured
        
        # Initialize as client node
        await fl_manager.initialize_federation(client_id=client_id)
        
        # Establish secure channel with coordinator
        channel_result = await qr_security.establish_secure_channel("coordinator")
        
        if channel_result["status"] != "success":
            logger.warning("Failed to establish secure channel with coordinator")
        
        return {
            "status": "initialized",
            "role": "client",
            "client_id": client_id,
            "secure_channel": channel_result if channel_result["status"] == "success" else None
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

@router.post("/run-round")
async def run_federation_round(
    config: Optional[Dict[str, Any]] = Body(None),
    fl_manager = Depends(get_federated_learning),
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Run a round of federated learning"""
    if current_user["role"] not in ["admin", "data_scientist"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Check if federation is initialized
    if fl_manager.is_coordinator and fl_manager.aggregation_server:
        # Run federation round as coordinator
        try:
            new_global_model = await fl_manager.run_federation_round()
            
            # If configured, encrypt the model parameters before storing/transmitting
            if config and config.get("encrypt_model", False):
                # List secure channels
                channels = await qr_security.list_secure_channels()
                
                # Encrypt model parameters for each client
                encrypted_models = {}
                for channel_info in channels["channels"]:
                    channel_id = channel_info["channel_id"]
                    device_id = channel_info["device_id"]
                    
                    # Serialize model using pickle, then encode to base64
                    try:
                        model_bytes = pickle.dumps(new_global_model)
                    except Exception as ser_exc:
                        logger.error(f"Failed to serialize model for client {device_id}: {ser_exc}")
                        # Decide how to handle: skip this client, or raise error
                        encrypted_models[device_id] = {"error": "Serialization failed"}
                        continue # Skip to next client
                    
                    # Encrypt for this client
                    encrypt_result = await qr_security.secure_channel_send(
                        channel_id,
                        model_bytes # Already bytes from pickle.dumps
                    )
                    
                    if encrypt_result["status"] == "success":
                        encrypted_models[device_id] = {
                            "channel_id": channel_id,
                            "ciphertext_b64": encrypt_result["ciphertext_b64"],
                            "nonce_b64": encrypt_result["nonce_b64"]
                        }
                
                return {
                    "status": "success",
                    "new_model_available": True,
                    "encrypted_models": encrypted_models,
                    "model_round": fl_manager.aggregation_server.current_round
                }
            
            return {
                "status": "success",
                "new_model_available": True,
                "model": new_global_model,
                "model_round": fl_manager.aggregation_server.current_round
            }
            
        except Exception as e:
            logger.error(f"Error running federation round: {e}")
            raise HTTPException(status_code=500, detail=f"Error running federation round: {str(e)}")
    
    elif fl_manager.client_node:
        # Participate as client
        try:
            await fl_manager.participate_as_client()
            return {
                "status": "success",
                "message": "Participated in federation round"
            }
        except Exception as e:
            logger.error(f"Error participating in federation round: {e}")
            raise HTTPException(status_code=500, detail=f"Error participating in federation round: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="Federated learning not properly initialized")

@router.post("/secure-submit")
async def secure_model_submission(
    data: Dict[str, Any] = Body(...),
    fl_manager = Depends(get_federated_learning),
    qr_security: QuantumResistantSecurity = Depends(get_qr_security),
    current_user: Dict = Depends(get_current_user)
):
    """Submit model updates over a secure channel"""
    if current_user["role"] not in ["admin", "data_scientist", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    channel_id = data.get("channel_id")
    encrypted_update = data.get("encrypted_update")
    nonce = data.get("nonce")
    
    if not (channel_id and encrypted_update and nonce):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    # Decrypt the model update
    try:
        decrypt_result = await qr_security.secure_channel_receive(
            channel_id,
            encrypted_update,
            nonce
        )
        
        if decrypt_result["status"] != "success":
            raise HTTPException(status_code=400, detail=f"Decryption failed: {decrypt_result.get('message')}")
        
        # Decrypted data is in bytes
        decrypted_bytes = decrypt_result["plaintext"]
        
        # Deserialize model update using pickle
        try:
            model_update_data = pickle.loads(decrypted_bytes)
        except Exception as deser_exc:
            logger.error(f"Failed to deserialize model update: {deser_exc}")
            raise HTTPException(status_code=400, detail=f"Invalid model update format after decryption: {deser_exc}")
        
        # In a real system, you would process and apply the model update.
        # For this example, we'll just acknowledge receipt and type.
        # The actual submission to AggregationServer needs to happen here.
        # This part requires redesigning how FederatedClient submits updates when in remote mode.
        # For now, just log the received data.
        logger.info(f"Securely received and deserialized model update of type: {type(model_update_data)}")
        
        return {
            "status": "success",
            "message": "Securely received and deserialized model update",
            "update_type": str(type(model_update_data)), # Example: show type of deserialized data
            "original_size_bytes": len(decrypted_bytes) # Size of pickled data
        }
        
    except Exception as e:
        logger.error(f"Error processing secure model submission: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing submission: {str(e)}")

@router.get("/global-model")
async def get_global_model_parameters_api(
    fl_manager = Depends(get_federated_learning),
    # Potentially add qr_security if model needs to be fetched securely even if not for a specific client round
    current_user: Dict = Depends(get_current_user) 
):
    """Endpoint for clients to fetch the current global model parameters."""
    if current_user["role"] not in ["admin", "data_scientist", "network_engineer", "client_node"]:
        # Added "client_node" as a potential role for automated clients
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if not fl_manager.aggregation_server:
        raise HTTPException(status_code=404, detail="Aggregation server not available or not initialized.")

    global_model_params = await fl_manager.aggregation_server.distribute_model_parameters()
    
    try:
        pickled_model = pickle.dumps(global_model_params)
        # It's common to base64 encode bytes when sending in JSON
        model_b64 = base64.b64encode(pickled_model).decode('utf-8')
        return {"model_b64": model_b64, "format": "pickle_base64"}
    except Exception as e:
        logger.error(f"Failed to serialize global model for API: {e}")
        raise HTTPException(status_code=500, detail="Failed to serialize global model")

@router.post("/client-update")
async def receive_client_update_api(
    client_id: str, # Can be from path, query, or body
    num_samples: int, # Can be from path, query, or body
    update_data: Dict[str, str] = Body(...), # Expecting {"update_b64": "...", "format": "pickle_base64"}
    fl_manager = Depends(get_federated_learning),
    current_user: Dict = Depends(get_current_user)
):
    """Endpoint for clients to submit their model updates."""
    if current_user["role"] not in ["admin", "data_scientist", "client_node"]: # Added client_node
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if not fl_manager.aggregation_server:
        raise HTTPException(status_code=404, detail="Aggregation server not available or not initialized.")

    update_b64 = update_data.get("update_b64")
    data_format = update_data.get("format")

    if not update_b64 or data_format != "pickle_base64":
        raise HTTPException(status_code=400, detail="Invalid update_data format. Expected pickle_base64.")

    try:
        pickled_update = base64.b64decode(update_b64)
        local_update_payload = pickle.loads(pickled_update)
    except Exception as e:
        logger.error(f"Failed to deserialize client update for client {client_id}: {e}")
        raise HTTPException(status_code=400, detail="Failed to deserialize client update")

    # Pass to the aggregation server
    # Note: client_id and num_samples could also be part of the update_data payload
    await fl_manager.aggregation_server.receive_update(client_id, local_update_payload, num_samples)
    
    return {"status": "success", "message": f"Update received from client {client_id}"}

# Add more endpoints as needed for your federated learning system