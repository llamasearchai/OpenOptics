"""
Programmable Optics Controller Module

This module provides functionality to control programmable optical devices,
which allow real-time adaptation of optical signal parameters based on network
conditions for optimized performance.
"""

import asyncio
import logging
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union
import json

logger = logging.getLogger(__name__)

class ProgrammableOpticsController:
    """Controller for programmable optical devices and adaptive algorithms"""
    
    def __init__(self):
        """Initialize the programmable optics controller"""
        self.devices = {}
        self.controllers = {}
        self._controller_tasks = {}
        self._load_devices()
    
    def _load_devices(self):
        """Load programmable devices from configuration or database"""
        # In a real implementation, this would load from a database
        # For this example, we'll create some sample devices
        sample_devices = [
            {
                "id": "coherent_trx_101",
                "name": "Coherent Transceiver 101",
                "type": "coherent_transceiver",
                "manufacturer": "OptiWave Systems",
                "model": "CTX-400G-DR4",
                "capabilities": {
                    "programmable_parameters": [
                        "modulation_format",
                        "tx_power",
                        "fec_overhead",
                        "baud_rate"
                    ],
                    "modulation_formats": [
                        "QPSK",
                        "8QAM",
                        "16QAM",
                        "64QAM"
                    ],
                    "power_range": {
                        "min": -10,
                        "max": 3,
                        "step": 0.5
                    },
                    "fec_options": [
                        {"name": "SD-FEC Gen1", "overhead": 0.15},
                        {"name": "SD-FEC Gen2", "overhead": 0.25},
                        {"name": "SD-FEC Gen3", "overhead": 0.33}
                    ],
                    "baud_rate": {
                        "min": 30,
                        "max": 64,
                        "step": 0.5
                    }
                },
                "current_config": {
                    "modulation_format": "16QAM",
                    "tx_power": 0,
                    "fec_overhead": 0.25,
                    "baud_rate": 64
                }
            },
            {
                "id": "rof_201",
                "name": "ROF Link Controller 201",
                "type": "rof_controller",
                "manufacturer": "OptiWave Systems",
                "model": "ROF-100G",
                "capabilities": {
                    "programmable_parameters": [
                        "rf_gain",
                        "optical_power",
                        "bias_current"
                    ],
                    "rf_gain_range": {
                        "min": 0,
                        "max": 30,
                        "step": 1
                    },
                    "optical_power_range": {
                        "min": -6,
                        "max": 3,
                        "step": 0.5
                    },
                    "bias_current_range": {
                        "min": 5,
                        "max": 100,
                        "step": 1
                    }
                },
                "current_config": {
                    "rf_gain": 15,
                    "optical_power": 0,
                    "bias_current": 50
                }
            }
        ]
        
        for device in sample_devices:
            self.devices[device["id"]] = device
        
        logger.info(f"Loaded {len(self.devices)} programmable optical devices")
    
    async def get_programmable_devices(self) -> List[Dict[str, Any]]:
        """Get a list of all programmable optical devices
        
        Returns:
            List of device information dictionaries
        """
        return list(self.devices.values())
    
    async def get_device_parameters(self, device_id: str) -> Dict[str, Any]:
        """Get programmable parameters for a specific device
        
        Args:
            device_id: Device identifier
            
        Returns:
            Dictionary of device parameters and capabilities
        """
        if device_id not in self.devices:
            return {"status": "error", "message": f"Device not found: {device_id}"}
        
        device = self.devices[device_id]
        return {
            "device_id": device_id,
            "device_name": device["name"],
            "device_type": device["type"],
            "capabilities": device["capabilities"],
            "current_config": device["current_config"]
        }
    
    async def configure_device(self, device_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Configure programmable parameters for a device
        
        Args:
            device_id: Device identifier
            parameters: Dictionary of parameters to configure
            
        Returns:
            Status of configuration operation
        """
        if device_id not in self.devices:
            return {"status": "error", "message": f"Device not found: {device_id}"}
        
        device = self.devices[device_id]
        
        # Validate parameters against capabilities
        for param_name, param_value in parameters.items():
            if param_name not in device["capabilities"].get("programmable_parameters", []):
                return {
                    "status": "error", 
                    "message": f"Parameter not supported: {param_name}"
                }
            
            # Validate parameter values based on type
            if param_name == "modulation_format":
                if param_value not in device["capabilities"].get("modulation_formats", []):
                    return {
                        "status": "error",
                        "message": f"Invalid modulation format: {param_value}"
                    }
            elif "_range" in device["capabilities"].get(param_name + "_range", {}):
                range_info = device["capabilities"][param_name + "_range"]
                if param_value < range_info["min"] or param_value > range_info["max"]:
                    return {
                        "status": "error",
                        "message": f"Value out of range for {param_name}: {param_value}"
                    }
        
        # Update device configuration
        for param_name, param_value in parameters.items():
            device["current_config"][param_name] = param_value
        
        # In a real implementation, this would send commands to the actual hardware
        logger.info(f"Configured device {device_id} with parameters: {parameters}")
        
        # Record the configuration change
        timestamp = datetime.datetime.now().isoformat()
        config_change = {
            "device_id": device_id,
            "timestamp": timestamp,
            "parameters": parameters,
            "status": "success"
        }
        
        return {
            "status": "success",
            "device_id": device_id,
            "timestamp": timestamp,
            "applied_config": device["current_config"]
        }
    
    async def start_adaptive_controller(self, device_id: str, controller_type: str, 
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Start an adaptive controller for a device
        
        Args:
            device_id: Device identifier
            controller_type: Type of controller to start
            parameters: Controller parameters
            
        Returns:
            Status and controller information
        """
        if device_id not in self.devices:
            return {"status": "error", "message": f"Device not found: {device_id}"}
        
        # Check if a controller is already running for this device
        if device_id in self._controller_tasks and not self._controller_tasks[device_id].done():
            return {
                "status": "error",
                "message": f"Controller already running for device {device_id}"
            }
        
        # Create a new controller instance
        controller_id = str(uuid.uuid4())
        
        if controller_type == "snr_optimizer":
            controller = SNROptimizer(self, device_id, parameters)
        elif controller_type == "modulation_adaptor":
            controller = ModulationAdaptor(self, device_id, parameters)
        elif controller_type == "power_optimizer":
            controller = PowerOptimizer(self, device_id, parameters)
        else:
            return {
                "status": "error",
                "message": f"Unsupported controller type: {controller_type}"
            }
        
        # Store controller
        self.controllers[controller_id] = {
            "id": controller_id,
            "device_id": device_id,
            "type": controller_type,
            "parameters": parameters,
            "status": "starting",
            "start_time": datetime.datetime.now().isoformat(),
            "controller": controller
        }
        
        # Start controller as a background task
        self._controller_tasks[device_id] = asyncio.create_task(
            controller.run_loop()
        )
        
        # Update status
        self.controllers[controller_id]["status"] = "running"
        
        return {
            "status": "success",
            "controller_id": controller_id,
            "device_id": device_id,
            "controller_type": controller_type,
            "parameters": parameters
        }
    
    async def stop_adaptive_controller(self, controller_id: str) -> Dict[str, Any]:
        """Stop an adaptive controller
        
        Args:
            controller_id: Controller identifier
            
        Returns:
            Status of stop operation
        """
        if controller_id not in self.controllers:
            return {"status": "error", "message": f"Controller not found: {controller_id}"}
        
        controller_info = self.controllers[controller_id]
        device_id = controller_info["device_id"]
        
        if device_id in self._controller_tasks and not self._controller_tasks[device_id].done():
            # Cancel the controller task
            self._controller_tasks[device_id].cancel()
            try:
                await self._controller_tasks[device_id]
            except asyncio.CancelledError:
                pass
        
        # Update status
        controller_info["status"] = "stopped"
        controller_info["stop_time"] = datetime.datetime.now().isoformat()
        
        return {
            "status": "success",
            "controller_id": controller_id,
            "device_id": device_id,
            "message": "Controller stopped successfully"
        }
    
    async def get_controller_status(self, controller_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of adaptive controllers
        
        Args:
            controller_id: Optional controller identifier
            
        Returns:
            Status of specified controller or all controllers
        """
        if controller_id is not None:
            if controller_id not in self.controllers:
                return {"status": "error", "message": f"Controller not found: {controller_id}"}
            
            controller_info = self.controllers[controller_id].copy()
            # Remove the controller object from the info
            controller_info.pop("controller", None)
            return controller_info
        
        # Return all controllers
        result = []
        for c_id, c_info in self.controllers.items():
            info = c_info.copy()
            info.pop("controller", None)
            result.append(info)
        
        return result
    
    async def simulate_performance(self, device_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate performance of a device with specified parameters
        
        Args:
            device_id: Device identifier
            parameters: Device parameters to simulate
            
        Returns:
            Simulated performance metrics
        """
        if device_id not in self.devices:
            return {"status": "error", "message": f"Device not found: {device_id}"}
        
        device = self.devices[device_id]
        
        # Combine current config with new parameters
        config = device["current_config"].copy()
        config.update(parameters)
        
        # Perform a simulated performance evaluation
        # This is a simplified model for demonstration
        if device["type"] == "coherent_transceiver":
            performance = self._simulate_coherent_performance(config)
        elif device["type"] == "rof_controller":
            performance = self._simulate_rof_performance(config)
        else:
            performance = {"status": "error", "message": f"Unsupported device type: {device['type']}"}
        
        return {
            "status": "success",
            "device_id": device_id,
            "simulated_parameters": config,
            "performance": performance
        }
    
    def _simulate_coherent_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate performance of a coherent transceiver
        
        Args:
            config: Device configuration
            
        Returns:
            Simulated performance metrics
        """
        mod_format = config.get("modulation_format", "16QAM")
        tx_power = config.get("tx_power", 0)
        fec_overhead = config.get("fec_overhead", 0.25)
        baud_rate = config.get("baud_rate", 64)
        
        # Simplified performance model based on configuration
        # In a real system, this would be much more sophisticated
        
        # Calculate bits per symbol based on modulation format
        bits_per_symbol = {
            "QPSK": 2,
            "8QAM": 3,
            "16QAM": 4,
            "64QAM": 6
        }.get(mod_format, 4)
        
        # Calculate capacity
        gross_capacity = baud_rate * bits_per_symbol  # Gbps
        net_capacity = gross_capacity / (1 + fec_overhead)  # Gbps
        
        # Estimate SNR and BER based on modulation and power
        # These are simplified models
        snr_base = {
            "QPSK": 15,
            "8QAM": 18,
            "16QAM": 22,
            "64QAM": 28
        }.get(mod_format, 22)
        
        # Power contribution to SNR (simplified)
        snr_power = tx_power + 10  # dB
        
        # Combined SNR (simplified)
        snr_est = snr_base + 0.5 * snr_power
        
        # BER estimation (very simplified)
        # In a real system, this would use proper formulas based on modulation
        if snr_est > snr_base + 3:
            ber_est = 1e-15  # Very low BER
        elif snr_est > snr_base:
            ber_est = 1e-9   # Good BER
        elif snr_est > snr_base - 3:
            ber_est = 1e-6   # Marginal BER
        elif snr_est > snr_base - 6:
            ber_est = 1e-4   # Poor BER
        else:
            ber_est = 1e-2   # Very poor BER
        
        # Power efficiency
        power_efficiency = net_capacity / (10**(tx_power/10))  # Gbps/mW
        
        return {
            "gross_capacity_gbps": gross_capacity,
            "net_capacity_gbps": net_capacity,
            "estimated_snr_db": snr_est,
            "estimated_ber": ber_est,
            "power_efficiency": power_efficiency,
            "reach_km": self._estimate_reach(mod_format, tx_power, fec_overhead)
        }
    
    def _simulate_rof_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate performance of a Radio-over-Fiber (RoF) link
        
        Args:
            config: Device configuration
            
        Returns:
            Simulated performance metrics
        """
        rf_gain = config.get("rf_gain", 15)
        optical_power = config.get("optical_power", 0)
        bias_current = config.get("bias_current", 50)
        
        # Simplified RoF performance model
        
        # Estimate SNR based on configuration
        snr_base = 20  # dB baseline
        snr_power = optical_power * 2  # Optical power contribution
        snr_bias = (bias_current - 50) * 0.1  # Bias current impact
        snr_gain = rf_gain * 0.5  # RF gain impact
        
        snr_est = snr_base + snr_power + snr_bias + snr_gain
        
        # Estimate EVM (Error Vector Magnitude)
        evm_est = 100 / (10**(snr_est/20))  # Simplified conversion
        
        # Estimate bandwidth
        bandwidth = 2.0  # GHz baseline
        if rf_gain > 25:
            bandwidth *= 0.8  # High gain can reduce bandwidth
        
        # Estimate linearity
        iip3 = 10 - (rf_gain * 0.2) + (bias_current * 0.05)  # dBm
        
        return {
            "estimated_snr_db": snr_est,
            "estimated_evm_percent": evm_est,
            "bandwidth_ghz": bandwidth,
            "iip3_dbm": iip3,
            "reach_m": self._estimate_rof_reach(optical_power, rf_gain)
        }
    
    def _estimate_reach(self, mod_format: str, tx_power: float, fec_overhead: float) -> float:
        """Estimate reach for coherent transceiver
        
        Args:
            mod_format: Modulation format
            tx_power: Transmit power (dBm)
            fec_overhead: FEC overhead
            
        Returns:
            Estimated reach in km
        """
        # Base reach by modulation format
        base_reach = {
            "QPSK": 2000,
            "8QAM": 1000,
            "16QAM": 500,
            "64QAM": 200
        }.get(mod_format, 500)
        
        # Power contribution (simplified)
        power_factor = 1.0 + (tx_power / 10.0)
        
        # FEC contribution (simplified)
        fec_factor = 1.0 + (fec_overhead * 2)
        
        return base_reach * power_factor * fec_factor
    
    def _estimate_rof_reach(self, optical_power: float, rf_gain: float) -> float:
        """Estimate reach for RoF link
        
        Args:
            optical_power: Optical power (dBm)
            rf_gain: RF gain (dB)
            
        Returns:
            Estimated reach in meters
        """
        base_reach = 300  # meters
        power_factor = 1.0 + (optical_power / 5.0)
        gain_factor = 1.0 + ((rf_gain - 15) / 30.0)
        
        return base_reach * power_factor * gain_factor

    def get_controller_id_by_device(self, device_id: str) -> Optional[str]:
        for c_id, c_info in self.controllers.items():
            if c_info["device_id"] == device_id:
                return c_id
        return None


class AdaptiveController:
    """Base class for adaptive controllers"""
    
    def __init__(self, controller, device_id, parameters):
        self.controller = controller
        self.device_id = device_id
        self.parameters = parameters
        self.running = True
        self.last_update = None
    
    async def run_loop(self):
        """Main control loop"""
        raise NotImplementedError("Subclasses must implement run_loop")
    
    async def stop(self):
        """Stop the controller"""
        self.running = False


class SNROptimizer(AdaptiveController):
    """Controller to optimize SNR by adjusting parameters"""
    
    async def run_loop(self):
        """Run the SNR optimization loop"""
        update_interval = self.parameters.get("update_interval", 60)  # seconds
        target_snr_metric = self.parameters.get("target_snr_metric", "estimated_snr_db") # Metric name from simulation
        optimization_step = self.parameters.get("optimization_step", 0.2) # dB for power, or relative for others
        max_iterations_no_improvement = self.parameters.get("max_iterations_no_improvement", 5)
        
        logger.info(f"SNROptimizer for device {self.device_id} starting. Update interval: {update_interval}s, Target SNR metric: {target_snr_metric}, Step: {optimization_step}.")

        iterations_without_improvement = 0
        best_snr_achieved = -float('inf')

        while self.running:
            await asyncio.sleep(update_interval) # Wait for the interval first
            if not self.running: break

            device = self.controller.devices.get(self.device_id)
            if not device:
                logger.error(f"SNROptimizer ({self.device_id}): Device not found.")
                break
            
            original_config = device["current_config"].copy()
            logger.info(f"SNROptimizer ({self.device_id}): Current config: {original_config}")

            # Identify a parameter to tweak for SNR. For coherent_transceiver, tx_power is a candidate.
            param_to_tweak = None
            current_param_value = None
            param_range = None

            if device["type"] == "coherent_transceiver" and "tx_power" in original_config:
                param_to_tweak = "tx_power"
                current_param_value = original_config["tx_power"]
                if "power_range" in device["capabilities"]:
                    param_range = device["capabilities"]["power_range"]
            # Add more device types and parameters here if needed
            # elif device["type"] == "rof_controller" and "rf_gain" in original_config:
            #     param_to_tweak = "rf_gain"
            # ...

            if not param_to_tweak or current_param_value is None:
                logger.warning(f"SNROptimizer ({self.device_id}): No suitable parameter to tweak for device type {device['type']}. Skipping iteration.")
                continue

            # Simulate current performance
            current_sim_result = await self.controller.simulate_performance(self.device_id, {}) # Use current params
            current_snr = current_sim_result.get("performance", {}).get(target_snr_metric, -float('inf'))
            logger.info(f"SNROptimizer ({self.device_id}): Current simulated SNR is {current_snr:.2f} dB.")

            if current_snr > best_snr_achieved:
                best_snr_achieved = current_snr
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= max_iterations_no_improvement:
                logger.info(f"SNROptimizer ({self.device_id}): No improvement for {max_iterations_no_improvement} iterations. Pausing or reducing step.")
                # Consider reducing step size or pausing
                optimization_step *= 0.8 # Reduce step size
                if optimization_step < 0.01: optimization_step = 0.01 # Min step
                iterations_without_improvement = 0 # Reset counter


            # Try tweaking the parameter up and down
            best_tweak_param_value = current_param_value
            best_tweak_snr = current_snr
            
            for direction in [-1, 1]: # Try decreasing and increasing
                tweaked_param_value = current_param_value + (direction * optimization_step)
                
                # Clamp to range if available
                if param_range and "min" in param_range and "max" in param_range:
                    tweaked_param_value = max(param_range["min"], min(param_range["max"], tweaked_param_value))

                if abs(tweaked_param_value - current_param_value) < optimization_step / 10: # Avoid tiny ineffective changes
                    continue

                logger.info(f"SNROptimizer ({self.device_id}): Trying {param_to_tweak} = {tweaked_param_value:.2f}")
                sim_result = await self.controller.simulate_performance(self.device_id, {param_to_tweak: tweaked_param_value})
                sim_snr = sim_result.get("performance", {}).get(target_snr_metric, -float('inf'))
                
                if sim_snr > best_tweak_snr:
                    best_tweak_snr = sim_snr
                    best_tweak_param_value = tweaked_param_value
            
            if best_tweak_param_value != current_param_value and best_tweak_snr > current_snr:
                logger.info(f"SNROptimizer ({self.device_id}): Applying new {param_to_tweak} = {best_tweak_param_value:.2f} (Simulated SNR: {best_tweak_snr:.2f} dB)")
                await self.controller.configure_device(self.device_id, {param_to_tweak: best_tweak_param_value})
                if best_tweak_snr > best_snr_achieved: # Update overall best SNR
                     best_snr_achieved = best_tweak_snr
                     iterations_without_improvement = 0
            else:
                logger.info(f"SNROptimizer ({self.device_id}): No better configuration found in this iteration. Current param {param_to_tweak} = {current_param_value:.2f} (Simulated SNR: {current_snr:.2f} dB)")
            
            self.last_update = datetime.datetime.now().isoformat()
            controller_meta_id = self.controller.get_controller_id_by_device(self.device_id)
            if controller_meta_id and controller_meta_id in self.controller.controllers:
                 self.controller.controllers[controller_meta_id]["status"] = f"running - last SNR: {best_snr_achieved:.2f}dB"
                 self.controller.controllers[controller_meta_id]["last_metrics"] = {target_snr_metric: best_snr_achieved, "parameter_tweaked": param_to_tweak, "last_value": best_tweak_param_value}


        logger.info(f"SNROptimizer for device {self.device_id} stopped.")

class ModulationAdaptor(AdaptiveController):
    """Controller to adapt modulation format"""

    async def run_loop(self):
        update_interval = self.parameters.get("update_interval", 300)  # seconds
        # Define target performance metrics from simulation results
        target_ber_threshold = self.parameters.get("target_ber_threshold", 1e-5) 
        target_snr_threshold = self.parameters.get("target_snr_threshold", None) # e.g., 18 dB, if BER is not available
        prioritize_metric = self.parameters.get("prioritize_metric", "net_capacity_gbps") # or "ber", "snr"

        logger.info(f"ModulationAdaptor for device {self.device_id} starting. Interval: {update_interval}s, Target BER < {target_ber_threshold}, Target SNR > {target_snr_threshold}")

        while self.running:
            await asyncio.sleep(update_interval)
            if not self.running: break

            device = self.controller.devices.get(self.device_id)
            if not device:
                logger.error(f"ModulationAdaptor ({self.device_id}): Device not found.")
                break
            
            original_config = device["current_config"].copy()
            current_modulation = original_config.get("modulation_format")
            logger.info(f"ModulationAdaptor ({self.device_id}): Current config: {original_config}")

            if device["type"] != "coherent_transceiver":
                logger.warning(f"ModulationAdaptor ({self.device_id}): Device type {device['type']} not supported for modulation adaptation. Skipping.")
                continue

            available_modulations = device["capabilities"].get("modulation_formats", [])
            if not available_modulations:
                logger.warning(f"ModulationAdaptor ({self.device_id}): No modulation formats listed in capabilities. Skipping.")
                continue

            best_modulation = current_modulation
            best_performance_metric_val = -float('inf') # For capacity/snr
            if prioritize_metric == "ber":
                best_performance_metric_val = float('inf') # For BER, lower is better
            
            candidate_modulations = []

            for mod_format in available_modulations:
                logger.info(f"ModulationAdaptor ({self.device_id}): Simulating performance for modulation {mod_format}...")
                # Simulate with the new modulation format, keeping other params same as original_config
                # (or allow tweaking other params like FEC if tightly coupled)
                temp_config = {"modulation_format": mod_format} 
                sim_result_full = await self.controller.simulate_performance(self.device_id, temp_config)
                sim_performance = sim_result_full.get("performance", {})

                sim_ber = sim_performance.get("estimated_ber", float('inf'))
                sim_snr = sim_performance.get("estimated_snr_db", -float('inf'))
                sim_capacity = sim_performance.get("net_capacity_gbps", -float('inf'))

                logger.info(f"ModulationAdaptor ({self.device_id}): {mod_format} -> SNR: {sim_snr:.2f}dB, BER: {sim_ber:.2e}, Capacity: {sim_capacity:.2f}Gbps")

                # Check if performance meets thresholds
                meets_thresholds = False
                if target_snr_threshold is not None:
                    if sim_snr >= target_snr_threshold and sim_ber <= target_ber_threshold:
                        meets_thresholds = True
                elif sim_ber <= target_ber_threshold:
                    meets_thresholds = True
                
                if meets_thresholds:
                    candidate_modulations.append({
                        "format": mod_format,
                        "snr": sim_snr,
                        "ber": sim_ber,
                        "capacity": sim_capacity
                    })
            
            if not candidate_modulations:
                logger.warning(f"ModulationAdaptor ({self.device_id}): No modulation format met the required thresholds.")
                # Optionally, could revert to a very robust default like QPSK if current one also fails, 
                # but for now, we do nothing if no candidates are found better than current implicit thresholds.
            else:
                # Select the best candidate based on the prioritized metric
                if prioritize_metric == "net_capacity_gbps":
                    candidate_modulations.sort(key=lambda x: x["capacity"], reverse=True)
                elif prioritize_metric == "ber":
                    candidate_modulations.sort(key=lambda x: x["ber"], reverse=False)
                elif prioritize_metric == "snr":
                    candidate_modulations.sort(key=lambda x: x["snr"], reverse=True)
                
                best_candidate = candidate_modulations[0]
                best_modulation = best_candidate["format"]
                logger.info(f"ModulationAdaptor ({self.device_id}): Best candidate format is {best_modulation} with {best_candidate}")

            if best_modulation and best_modulation != current_modulation:
                logger.info(f"ModulationAdaptor ({self.device_id}): Changing modulation from {current_modulation} to {best_modulation}")
                await self.controller.configure_device(self.device_id, {"modulation_format": best_modulation})
            elif best_modulation:
                logger.info(f"ModulationAdaptor ({self.device_id}): Current modulation {current_modulation} is still optimal or best available.")
            else:
                 logger.info(f"ModulationAdaptor ({self.device_id}): No change in modulation, best_modulation was not determined or same as current.")

            self.last_update = datetime.datetime.now().isoformat()
            controller_meta_id = self.controller.get_controller_id_by_device(self.device_id)
            if controller_meta_id and controller_meta_id in self.controller.controllers:
                 self.controller.controllers[controller_meta_id]["status"] = f"running - current mod: {device['current_config'].get('modulation_format')}"
                 self.controller.controllers[controller_meta_id]["last_metrics"] = best_candidate if candidate_modulations and best_modulation else None

        logger.info(f"ModulationAdaptor for device {self.device_id} stopped.")

class PowerOptimizer(AdaptiveController):
    """Controller to optimize power consumption"""

    async def run_loop(self):
        update_interval = self.parameters.get("update_interval", 120)  # seconds
        power_param_name = self.parameters.get("power_param_name", "tx_power") # Parameter to adjust for power saving
        power_reduction_step = self.parameters.get("power_reduction_step", 0.1) # e.g., 0.1 dB for tx_power
        min_param_value = self.parameters.get("min_param_value", None) # e.g., -5 dBm for tx_power, device specific

        # Performance thresholds that must be maintained
        critical_metric_name = self.parameters.get("critical_metric_name", "estimated_ber") # e.g., "estimated_ber" or "estimated_snr_db"
        critical_metric_threshold = self.parameters.get("critical_metric_threshold", 1e-4) # Max BER or Min SNR
        is_metric_lower_better = self.parameters.get("is_metric_lower_better", True) # True for BER, False for SNR

        logger.info(f"PowerOptimizer for device {self.device_id} starting. Interval: {update_interval}s, Param: {power_param_name}, Step: {power_reduction_step}")
        logger.info(f"PowerOptimizer ({self.device_id}): Maintaining {critical_metric_name} {'<' if is_metric_lower_better else '>'} {critical_metric_threshold}")

        while self.running:
            await asyncio.sleep(update_interval)
            if not self.running: break

            device = self.controller.devices.get(self.device_id)
            if not device:
                logger.error(f"PowerOptimizer ({self.device_id}): Device not found.")
                break
            
            current_config = device["current_config"].copy()
            current_power_param_value = current_config.get(power_param_name)

            if current_power_param_value is None:
                logger.warning(f"PowerOptimizer ({self.device_id}): Monitored power parameter '{power_param_name}' not found in device config. Skipping.")
                continue
            
            logger.info(f"PowerOptimizer ({self.device_id}): Current {power_param_name} = {current_power_param_value}")

            # Get current performance against critical metric
            current_sim_result = await self.controller.simulate_performance(self.device_id, {})
            current_critical_metric_value = current_sim_result.get("performance", {}).get(critical_metric_name)

            if current_critical_metric_value is None:
                logger.warning(f"PowerOptimizer ({self.device_id}): Critical metric '{critical_metric_name}' not found in simulation results. Skipping.")
                continue
            
            logger.info(f"PowerOptimizer ({self.device_id}): Current critical metric {critical_metric_name} = {current_critical_metric_value:.2e if isinstance(current_critical_metric_value, float) and is_metric_lower_better else current_critical_metric_value}")

            # Check if current performance is acceptable
            current_perf_ok = (current_critical_metric_value <= critical_metric_threshold) if is_metric_lower_better else (current_critical_metric_value >= critical_metric_threshold)

            if not current_perf_ok:
                logger.warning(f"PowerOptimizer ({self.device_id}): Current performance ({current_critical_metric_value}) for {critical_metric_name} is already outside threshold ({critical_metric_threshold}). Attempting to restore by increasing power slightly if possible.")
                # Try to increase power slightly if it was reduced previously, or if this is the first run and it's bad
                # This part needs careful logic to avoid oscillation with other optimizers
                # For simplicity, we won't try to actively *increase* power here beyond its starting point, 
                # assuming other mechanisms or manual intervention would handle baseline bad performance.
                # We focus on *reducing* power if performance is good.
                self.last_update = datetime.datetime.now().isoformat()
                controller_meta_id = self.controller.get_controller_id_by_device(self.device_id)
                if controller_meta_id and controller_meta_id in self.controller.controllers:
                    self.controller.controllers[controller_meta_id]["status"] = f"running - perf warning: {critical_metric_name} @ {current_critical_metric_value}"
                continue # Skip power reduction attempt if already below threshold

            # Try reducing the power parameter
            new_power_param_value = current_power_param_value - power_reduction_step

            # Check against absolute minimum if defined for the parameter
            param_cap_info = device["capabilities"].get(f"{power_param_name}_range", {})
            effective_min_val = min_param_value if min_param_value is not None else param_cap_info.get("min")

            if effective_min_val is not None and new_power_param_value < effective_min_val:
                logger.info(f"PowerOptimizer ({self.device_id}): Proposed {power_param_name} ({new_power_param_value}) is below minimum ({effective_min_val}). Clamping or stopping reduction.")
                new_power_param_value = effective_min_val 
                if new_power_param_value == current_power_param_value: # Already at min
                    logger.info(f"PowerOptimizer ({self.device_id}): Already at minimum {power_param_name}. No further reduction possible.")
                    self.last_update = datetime.datetime.now().isoformat()
                    # Update controller status if needed
                    continue

            logger.info(f"PowerOptimizer ({self.device_id}): Testing with {power_param_name} = {new_power_param_value}")
            test_sim_result = await self.controller.simulate_performance(self.device_id, {power_param_name: new_power_param_value})
            test_critical_metric_value = test_sim_result.get("performance", {}).get(critical_metric_name)

            if test_critical_metric_value is None:
                logger.warning(f"PowerOptimizer ({self.device_id}): Critical metric '{critical_metric_name}' not found in test simulation. Reverting.")
                continue

            logger.info(f"PowerOptimizer ({self.device_id}): Test simulation with {power_param_name}={new_power_param_value} gives {critical_metric_name}={test_critical_metric_value:.2e if isinstance(test_critical_metric_value, float) and is_metric_lower_better else test_critical_metric_value}")
            
            # Check if performance with reduced power is still acceptable
            test_perf_ok = (test_critical_metric_value <= critical_metric_threshold) if is_metric_lower_better else (test_critical_metric_value >= critical_metric_threshold)

            if test_perf_ok:
                logger.info(f"PowerOptimizer ({self.device_id}): Performance OK with reduced {power_param_name}. Applying new value: {new_power_param_value}")
                await self.controller.configure_device(self.device_id, {power_param_name: new_power_param_value})
                current_critical_metric_value = test_critical_metric_value # Update current metric for controller status
            else:
                logger.info(f"PowerOptimizer ({self.device_id}): Performance NOT OK with reduced {power_param_name}. Reverting to {current_power_param_value}")
                # No change needed, already at optimal for this step
            
            self.last_update = datetime.datetime.now().isoformat()
            controller_meta_id = self.controller.get_controller_id_by_device(self.device_id)
            if controller_meta_id and controller_meta_id in self.controller.controllers:
                 final_power_val = device['current_config'].get(power_param_name)
                 self.controller.controllers[controller_meta_id]["status"] = f"running - {power_param_name}: {final_power_val}, {critical_metric_name}: {current_critical_metric_value:.2e if isinstance(current_critical_metric_value, float) and is_metric_lower_better else current_critical_metric_value}"
                 self.controller.controllers[controller_meta_id]["last_metrics"] = {power_param_name: final_power_val, critical_metric_name: current_critical_metric_value}

        logger.info(f"PowerOptimizer for device {self.device_id} stopped.")