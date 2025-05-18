# Add these routes to network_routes.py

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, List # List might be needed for response models

# Placeholder for an authentication dependency
# In a real app, this would handle actual authentication and user loading.
async def get_current_user() -> Dict[str, Any]:
    # For placeholder purposes, assume a default admin/engineer user.
    # Replace with actual authentication logic.
    return {"username": "dummy_user", "role": "admin", "email": "user@example.com"}


# Initialize the ProgrammableOpticsController once
# This assumes ProgrammableOpticsController can be used as a singleton here.
# If it has request-specific state or heavy dependencies, FastAPI's Depends might be better.
from ..network.programmable_optics import ProgrammableOpticsController
programmable_optics_controller = ProgrammableOpticsController()

router = APIRouter(
    prefix="/network", # Add a prefix for these network-related routes
    tags=["Network - Programmable Optics"], # Tag for OpenAPI docs
)

@router.get("/programmable/devices")
async def get_programmable_devices_route(
    current_user: Dict = Depends(get_current_user)
):
    """Get a list of all programmable optical devices"""
    if current_user["role"] not in ["admin", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use the shared controller instance
    devices = await programmable_optics_controller.get_programmable_devices()
    return {"status": "success", "devices": devices}

@router.get("/programmable/device/{device_id}")
async def get_device_parameters_route(
    device_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get parameters for a specific programmable device"""
    if current_user["role"] not in ["admin", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use the shared controller instance
    result = await programmable_optics_controller.get_device_parameters(device_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    
    return result

@router.post("/programmable/device/{device_id}/configure")
async def configure_device_route(
    device_id: str,
    parameters: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """Configure parameters for a programmable device"""
    if current_user["role"] not in ["admin", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use the shared controller instance
    result = await programmable_optics_controller.configure_device(device_id, parameters)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    
    return result

@router.post("/programmable/device/{device_id}/simulate")
async def simulate_device_performance_route(
    device_id: str,
    parameters: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """Simulate performance of a device with specified parameters"""
    if current_user["role"] not in ["admin", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use the shared controller instance
    result = await programmable_optics_controller.simulate_performance(device_id, parameters)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    
    return result

@router.post("/programmable/controller/{device_id}/start")
async def start_adaptive_controller_route(
    device_id: str,
    controller_config: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """Start an adaptive controller for a device"""
    if current_user["role"] not in ["admin", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    controller_type = controller_config.get("controller_type")
    if not controller_type:
        raise HTTPException(status_code=400, detail="controller_type is required")
    
    parameters = controller_config.get("parameters", {})
    
    # Use the shared controller instance
    result = await programmable_optics_controller.start_adaptive_controller(device_id, controller_type, parameters)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    
    return result

@router.post("/programmable/controller/{controller_id}/stop")
async def stop_adaptive_controller_route(
    controller_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Stop an adaptive controller"""
    if current_user["role"] not in ["admin", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use the shared controller instance
    result = await programmable_optics_controller.stop_adaptive_controller(controller_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    
    return result

@router.get("/programmable/controllers")
async def get_all_controllers_route(
    current_user: Dict = Depends(get_current_user)
):
    """Get status of all adaptive controllers"""
    if current_user["role"] not in ["admin", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use the shared controller instance
    result = await programmable_optics_controller.get_controller_status()
    return {"status": "success", "controllers": result}

@router.get("/programmable/controller/{controller_id}")
async def get_controller_status_route(
    controller_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get status of a specific adaptive controller"""
    if current_user["role"] not in ["admin", "network_engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use the shared controller instance
    result = await programmable_optics_controller.get_controller_status(controller_id)
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    
    return {"status": "success", "controller": result}