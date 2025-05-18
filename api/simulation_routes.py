# Ensure these imports are at the top of the file
import base64
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, List # Added List
import logging

# Placeholder for an authentication dependency (can be shared with other route files)
async def get_current_user() -> Dict[str, Any]:
    return {"username": "dummy_simulator_user", "role": "engineer", "email": "simulator@example.com"}

logger = logging.getLogger(__name__)

# Initialize the CoherentLinkSimulator once
from ..simulation.coherent_link_simulator import CoherentLinkSimulator
coherent_link_sim_instance = CoherentLinkSimulator()

router = APIRouter(
    prefix="/simulation", # Prefix for simulation routes
    tags=["Simulation - Coherent Link"], # Tag for OpenAPI docs
)

# Add routes for coherent link simulation

@router.post("/coherent-link")
async def simulate_coherent_link_route(
    config: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """Simulate a coherent optical link with the given parameters"""
    if current_user["role"] not in ["admin", "network_engineer", "engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use shared simulator instance
    # simulator = CoherentLinkSimulator() # Old way
    
    try:
        # Run simulation
        results = await coherent_link_sim_instance.simulate(config)
        
        # Generate constellation plot
        # results["symbols"] and results["received_symbols"] should be lists of complex numbers
        # as CoherentLinkSimulator.simulate uses .tolist() on complex numpy arrays.

        constellation_plot = await coherent_link_sim_instance.generate_constellation_plot(
            results["symbols"], 
            results["received_symbols"],
            modulation=results.get("modulation", "QAM") # Pass modulation format for title
        )
        
        return {
            "status": "success",
            "results": results,
            "constellation_plot": constellation_plot
        }
    except Exception as e:
        logger.exception(f"Simulation error: {str(e)}") # Use logger.exception for stack trace
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.post("/coherent-link/parameter-sweep")
async def simulate_parameter_sweep_route(
    sweep_config: Dict[str, Any] = Body(...),
    current_user: Dict = Depends(get_current_user)
):
    """Run a parameter sweep simulation"""
    if current_user["role"] not in ["admin", "network_engineer", "engineer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Use shared simulator instance
    # simulator = CoherentLinkSimulator() # Old way
    
    try:
        # Extract parameters
        base_config = sweep_config.get("base_config")
        sweep_parameter = sweep_config.get("sweep_parameter")
        sweep_values = sweep_config.get("sweep_values") # This is a List[Any]
        
        if not all([base_config, sweep_parameter, sweep_values is not None]): # sweep_values can be an empty list
            raise HTTPException(status_code=400, detail="Missing required parameters: base_config, sweep_parameter, sweep_values")
        
        # Run parameter sweep with the updated signature
        results = await coherent_link_sim_instance.run_parameter_sweep(base_config, sweep_parameter, sweep_values)
        
        return {
            "status": "success",
            "sweep_parameter": sweep_parameter,
            # "sweep_values": sweep_values, # sweep_values is now part of results['param_values']
            "results": results # Return the whole result dict from run_parameter_sweep
            # "snr_values": results["snr_values"],
            # "ber_values": results["ber_values"]
        }
    except Exception as e:
        logger.exception(f"Parameter sweep error: {str(e)}") # Use logger.exception
        raise HTTPException(status_code=500, detail=f"Parameter sweep failed: {str(e)}")