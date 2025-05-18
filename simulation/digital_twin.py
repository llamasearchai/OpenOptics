import logging
logger = logging.getLogger(__name__)

import datetime
import copy
import uuid
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Enhanced OpticalPhysicsEngine with more functionality
class OpticalPhysicsEngine:
    """Engine for simulating optical physics in the network"""
    
    def __init__(self):
        """Initialize the optical physics engine"""
        self.c = 3e8  # Speed of light, m/s
        self.h = 6.63e-34  # Planck's constant, J⋅s
        self.global_default_parameters = {
            "dispersion": 16.5,  # ps/nm/km
            "alpha": 0.2,  # dB/km
            "gamma": 1.3,  # 1/W/km (nonlinear coefficient)
            "pmd": 0.1,  # ps/sqrt(km)
            "noise_figure": 5.0,  # dB (typical for EDFA, used for OSNR calc if no specific amp data)
            "connector_loss": 0.5, # dB per connector
            "splice_loss": 0.1, # dB per splice
        }
        logger.info("OpticalPhysicsEngine initialized with global default parameters.")
    
    def calculate_signal_properties(self, input_power_dbm: float, distance_km: float, 
                                  wavelength_nm: float, 
                                  fiber_parameters: Optional[Dict[str, float]] = None,
                                  amplifier_parameters: Optional[Dict[str, float]] = None 
                                  ) -> Dict[str, float]:
        """Calculate optical signal properties after transmission over a fiber segment, potentially with an amplifier.
        
        Args:
            input_power_dbm: Input power in dBm at the start of the segment.
            distance_km: Fiber distance in km for this segment.
            wavelength_nm: Signal wavelength in nm.
            fiber_parameters: Optional. Specific parameters for the fiber segment (e.g., {"alpha": 0.22, "dispersion": 17.0}).
                              Overrides global defaults.
            amplifier_parameters: Optional. Parameters for an amplifier at the end of this segment 
                                  (e.g., {"gain_db": 20, "noise_figure_db": 5.5}).
            
        Returns:
            Dictionary of signal properties at the end of the segment.
        """
        effective_params = self.global_default_parameters.copy()
        if fiber_parameters:
            effective_params.update(fiber_parameters)
        
        # Convert input power from dBm to W for physics calculations
        power_in_w = 10 ** ((input_power_dbm - 30) / 10)
        
        # --- Fiber Effects ---
        # Attenuation
        fiber_attenuation_db = effective_params["alpha"] * distance_km
        power_after_fiber_w = power_in_w * (10 ** (-fiber_attenuation_db / 10))
        
        # Chromatic Dispersion accumulated over this segment
        segment_dispersion_ps = effective_params["dispersion"] * distance_km
        
        # PMD accumulated over this segment
        segment_pmd_ps = effective_params["pmd"] * np.sqrt(distance_km)
        
        # Nonlinear Phase Shift (simplified SPM for this segment)
        # Effective length for nonlinearity
        alpha_linear_per_km = (effective_params["alpha"] / (10 * np.log10(np.exp(1)))) # Convert dB/km to linear 1/km
        l_eff_km = (1 - np.exp(-alpha_linear_per_km * distance_km)) / alpha_linear_per_km if alpha_linear_per_km > 0 else distance_km
        segment_nonlinear_phase_rad = effective_params["gamma"] * power_in_w * l_eff_km # Using input power as P_avg proxy

        power_out_segment_w = power_after_fiber_w
        osnr_db_after_segment = input_power_dbm - (effective_params["noise_figure"] if not amplifier_parameters else amplifier_parameters.get("noise_figure_db", effective_params["noise_figure"])) - 10 * np.log10(12.5e9/(self.c/(wavelength_nm*1e-9)**2 * 0.1e-9)) - fiber_attenuation_db # Simplified OSNR calc
        # This OSNR is a rough estimate based on input power and a generic NF. Amplifier section will refine this.

        # --- Amplifier Effects (if present at end of segment) ---
        final_power_w = power_out_segment_w
        final_osnr_db = osnr_db_after_segment # Will be updated if amplifier is present
        amplifier_gain_achieved_db = 0.0

        if amplifier_parameters:
            amp_gain_db = amplifier_parameters.get("gain_db", 0)
            amp_nf_db = amplifier_parameters.get("noise_figure_db", effective_params["noise_figure"])
            amplifier_gain_achieved_db = amp_gain_db

            power_before_amp_w = power_out_segment_w
            final_power_w = power_before_amp_w * (10**(amp_gain_db / 10))

            # OSNR calculation with an amplifier:
            # OSNR_in_db (to the amp) - NF_amp_db
            # OSNR_in_linear = P_signal_in_amp / (P_noise_in_ref_bw_in_amp)
            # P_noise_added_by_amp_in_ref_bw = (10**(amp_nf_db/10)) * (10**(amp_gain_db/10)) * self.h * (self.c/(wavelength_nm*1e-9)) * (12.5e9)
            # This is complex. A common simplification for cascaded amps:
            # 1/OSNR_total = sum(1/OSNR_stage_i)
            # OSNR_stage_i = P_signal_output_from_fiber_span / (NF_amp_linear * h * nu * B_ref)
            # For a single amp: OSNR_out_linear = (P_signal_in * G) / ( (P_noise_in * G) + (G-1)*NF*h*nu*B_ref )
            # If P_noise_in is negligible: OSNR_out_linear approx = P_signal_in / ( NF * h * nu * B_ref )
            
            signal_power_at_amp_input_dbm = 10 * np.log10(power_before_amp_w * 1000)
            # Noise power density at input of amp (from previous stages, or thermal)
            # For simplicity, let's consider the noise added by this amplifier predominantly.
            # ASE power added by this amplifier in reference bandwidth (0.1nm or 12.5GHz)
            freq_hz = self.c / (wavelength_nm * 1e-9)
            ref_optical_bw_hz = 12.5e9 
            nf_linear = 10**(amp_nf_db / 10)
            gain_linear = 10**(amp_gain_db / 10)
            # ASE power added by the amplifier itself
            p_ase_amplifier_w = (gain_linear -1) * nf_linear * self.h * freq_hz * ref_optical_bw_hz # if G>>1, approx G*NF*h*nu*B_o
            
            # Signal power at output of amplifier
            p_signal_amplifier_output_w = final_power_w

            # OSNR due to this amplifier (assuming input signal is noiseless for this calc, then combine)
            osnr_due_to_this_amp_linear = p_signal_amplifier_output_w / p_ase_amplifier_w if p_ase_amplifier_w > 0 else float('inf')
            # This is not quite right for overall OSNR. 
            # A better approximation for OSNR after an amp: OSNR_out = P_in / (NF * h * nu * B_ref) (if input OSNR was very high)
            # or OSNR_out_dB = P_in_dBm - NF_dB - 10log10(h*nu*B_ref in mW)
            # h*nu*B_ref in mW: (6.626e-34 * (3e8/1550e-9) * 12.5e9) * 1000 = 1.6e-6 mW approx -58 dBm
            osnr_contribution_this_amp_db = signal_power_at_amp_input_dbm - amp_nf_db - (-58) # Approx OSNR if this was only noise source
            final_osnr_db = osnr_contribution_this_amp_db # Simplified: assumes this amp is dominant noise source for the segment OSNR
            # In a cascade, one would sum 1/OSNR_linear from each stage. Here we are returning per-segment properties.

        final_output_power_dbm = 10 * np.log10(final_power_w * 1000) if final_power_w > 0 else -float('inf')

        return {
            "output_power_dbm": final_output_power_dbm,
            "fiber_attenuation_db": fiber_attenuation_db,
            "amplifier_gain_db": amplifier_gain_achieved_db,
            "segment_dispersion_ps": segment_dispersion_ps,
            "segment_pmd_ps": segment_pmd_ps,
            "segment_nonlinear_phase_rad": segment_nonlinear_phase_rad,
            "estimated_osnr_db_after_segment": final_osnr_db 
        }
    
    def model_impairments(self, signal_properties: Dict[str, float], modulation: str) -> Dict[str, float]:
        """Model the impact of impairments on signal quality
        
        Args:
            signal_properties: Signal properties
            modulation: Modulation format (QPSK, 16QAM, etc.)
            
        Returns:
            Dictionary with impairment impacts
        """
        # Get required OSNR for modulation format
        osnr_requirements = {
            "QPSK": 13.5,
            "8QAM": 16.5,
            "16QAM": 20.5,
            "64QAM": 26.5,
            "256QAM": 32.5
        }
        required_osnr = osnr_requirements.get(modulation, 20.0)
        
        # Get bits per symbol for modulation format
        bits_per_symbol = {
            "QPSK": 2,
            "8QAM": 3,
            "16QAM": 4,
            "64QAM": 6,
            "256QAM": 8
        }.get(modulation, 4)
        
        # Calculate OSNR margin
        osnr_margin = signal_properties["osnr_db"] - required_osnr
        
        # Calculate BER based on OSNR margin
        if osnr_margin > 6:
            ber = 1e-15
        elif osnr_margin > 3:
            ber = 1e-9
        elif osnr_margin > 0:
            ber = 1e-6
        elif osnr_margin > -3:
            ber = 1e-3
        else:
            ber = 1e-2
        
        # Calculate dispersion penalty
        dispersion_ps = signal_properties["dispersion_ps"]
        if dispersion_ps < 100:
            dispersion_penalty = 0.1
        elif dispersion_ps < 500:
            dispersion_penalty = 0.5
        elif dispersion_ps < 1000:
            dispersion_penalty = 1.0
        else:
            dispersion_penalty = 2.0
        
        # Calculate PMD penalty
        pmd_ps = signal_properties["pmd_ps"]
        symbol_rate = 32e9  # Example symbol rate, 32 GBaud
        symbol_period_ps = 1e12 / symbol_rate
        
        if pmd_ps < 0.1 * symbol_period_ps:
            pmd_penalty = 0.1
        elif pmd_ps < 0.2 * symbol_period_ps:
            pmd_penalty = 0.5
        elif pmd_ps < 0.3 * symbol_period_ps:
            pmd_penalty = 1.0
        else:
            pmd_penalty = 2.0
        
        # Calculate nonlinear penalty
        nonlinear_phase = signal_properties["segment_nonlinear_phase_rad"]
        if nonlinear_phase < 0.1:
            nonlinear_penalty = 0.1
        elif nonlinear_phase < 0.5:
            nonlinear_penalty = 0.5
        elif nonlinear_phase < 1.0:
            nonlinear_penalty = 1.0
        else:
            nonlinear_penalty = 2.0
        
        # Calculate total penalty
        total_penalty = dispersion_penalty + pmd_penalty + nonlinear_penalty
        
        # Calculate effective SNR
        effective_snr = signal_properties["estimated_osnr_db_after_segment"] - total_penalty
        
        # Calculate achievable capacity
        capacity_gbps = (symbol_rate / 1e9) * bits_per_symbol * (1 - ber)
        
        return {
            "ber": ber,
            "osnr_margin_db": osnr_margin,
            "dispersion_penalty_db": dispersion_penalty,
            "pmd_penalty_db": pmd_penalty,
            "nonlinear_penalty_db": nonlinear_penalty,
            "total_penalty_db": total_penalty,
            "effective_snr_db": effective_snr,
            "capacity_gbps": capacity_gbps
        }
    
    def calculate_link_budget(self, tx_power_dbm: float, rx_sensitivity_dbm: float, 
                           distance: float, components: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate the optical link power budget
        
        Args:
            tx_power_dbm: Transmitter output power in dBm
            rx_sensitivity_dbm: Receiver sensitivity in dBm
            distance: Link distance in km
            components: List of components in the link
            
        Returns:
            Dictionary with link budget analysis
        """
        # Calculate fiber loss
        fiber_loss = self.global_default_parameters["alpha"] * distance
        
        # Calculate component losses
        component_loss = sum(component.get("insertion_loss_db", 0) for component in components)
        
        # Calculate connector losses
        connector_count = sum(1 for component in components if component.get("type") == "connector")
        connector_loss = connector_count * 0.5  # Assuming 0.5 dB per connector
        
        # Calculate splice losses
        splice_count = sum(1 for component in components if component.get("type") == "splice")
        splice_loss = splice_count * 0.1  # Assuming 0.1 dB per splice
        
        # Calculate total loss
        total_loss = fiber_loss + component_loss + connector_loss + splice_loss
        
        # Calculate received power
        rx_power = tx_power_dbm - total_loss
        
        # Calculate power margin
        power_margin = rx_power - rx_sensitivity_dbm
        
        return {
            "tx_power_dbm": tx_power_dbm,
            "fiber_loss_db": fiber_loss,
            "component_loss_db": component_loss,
            "connector_loss_db": connector_loss,
            "splice_loss_db": splice_loss,
            "total_loss_db": total_loss,
            "rx_power_dbm": rx_power,
            "rx_sensitivity_dbm": rx_sensitivity_dbm,
            "power_margin_db": power_margin,
            "is_margin_sufficient": power_margin > 3.0  # Typically want at least 3 dB margin
        }
    
    def estimate_max_distance(self, tx_power_dbm: float, rx_sensitivity_dbm: float, 
                           margin_db: float = 3.0) -> float:
        """Estimate the maximum distance for a link
        
        Args:
            tx_power_dbm: Transmitter output power in dBm
            rx_sensitivity_dbm: Receiver sensitivity in dBm
            margin_db: Required power margin in dB
            
        Returns:
            Maximum distance in km
        """
        available_budget = tx_power_dbm - rx_sensitivity_dbm - margin_db
        
        if available_budget <= 0:
            return 0.0
        
        max_distance = available_budget / self.global_default_parameters["alpha"]
        return max_distance
    
    def simulate_wdm_system(self, channels: List[Dict[str, Any]], distance: float) -> List[Dict[str, Any]]:
        """Simulate a WDM system with multiple channels
        
        Args:
            channels: List of channel configurations
            distance: Link distance in km
            
        Returns:
            List of channel results
        """
        results = []
        
        # Calculate channel powers for nonlinear interactions
        channel_powers = [10 ** ((ch["power_dbm"] - 30) / 10) for ch in channels]
        total_power = sum(channel_powers)
        
        for i, channel in enumerate(channels):
            # Get channel parameters
            power_dbm = channel["power_dbm"]
            wavelength = channel["wavelength"]
            modulation = channel["modulation"]
            
            # Calculate signal properties
            signal_props = self.calculate_signal_properties(
                input_power=power_dbm,
                distance=distance,
                wavelength=wavelength
            )
            
            # Add cross-channel penalties
            if len(channels) > 1:
                # Calculate cross-phase modulation (XPM) penalty
                xpm_penalty = 0.5 * np.log10(len(channels))
                
                # Calculate four-wave mixing (FWM) penalty
                fwm_penalty = 0.3 * np.log10(len(channels)) * (total_power / channel_powers[i])
                
                # Update signal properties
                signal_props["estimated_osnr_db_after_segment"] -= (xpm_penalty + fwm_penalty)
                signal_props["xpm_penalty_db"] = xpm_penalty
                signal_props["fwm_penalty_db"] = fwm_penalty
            
            # Model impairments
            impairments = self.model_impairments(signal_props, modulation)
            
            # Combine results
            channel_result = {
                "channel_index": i,
                "wavelength_nm": wavelength,
                "modulation": modulation,
                "input_power_dbm": power_dbm,
                **signal_props,
                **impairments
            }
            
            results.append(channel_result)
        
        return results

# Placeholder for NetworkSimulationEngine (assuming it might be complex)
class NetworkSimulationEngine:
    """Engine for simulating network behavior and predicting failures"""
    
    def __init__(self, initial_state: Dict[str, Any], physics_engine: OpticalPhysicsEngine):
        """Initialize the network simulation engine
        
        Args:
            initial_state: Initial network state
            physics_engine: Optical physics engine for signal calculations
        """
        self.current_state = initial_state
        self.physics_engine = physics_engine
        self.simulation_history = []
        self.traffic_models = {
            "uniform": self._generate_uniform_traffic,
            "diurnal": self._generate_diurnal_traffic,
            "bursty": self._generate_bursty_traffic,
            "hotspot": self._generate_hotspot_traffic,
            "current": self._use_current_traffic
        }
        logger.info("NetworkSimulationEngine initialized with physics engine.")
    
    def update_state(self, new_state: Dict[str, Any]):
        """Update the current network state
        
        Args:
            new_state: New network state
        """
        self.current_state = new_state
        logger.info("NetworkSimulationEngine state updated.")
    
    async def simulate(self, state: Dict[str, Any], duration_seconds: int, traffic_model: str) -> Dict[str, Any]:
        """Simulate network behavior for a specified duration
        
        Args:
            state: Network state to simulate
            duration_seconds: Duration of simulation in seconds
            traffic_model: Traffic model to use
            
        Returns:
            Simulation results
        """
        logger.info(f"Simulating network with duration {duration_seconds}s and traffic {traffic_model}.")
        
        # Generate traffic matrix
        if traffic_model in self.traffic_models:
            traffic_matrix = await self.traffic_models[traffic_model](state, duration_seconds)
        else:
            logger.warning(f"Unknown traffic model: {traffic_model}. Using uniform traffic.")
            traffic_matrix = await self._generate_uniform_traffic(state, duration_seconds)
        
        # Initialize result metrics
        metrics = {
            "start_time": datetime.datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "traffic_model": traffic_model,
            "performance_metrics": {},
            "link_metrics": {},
            "device_metrics": {},
            "issues": []
        }
        
        # Simulate network behavior
        metrics = await self._simulate_network_behavior(state, traffic_matrix, duration_seconds, metrics)
        
        # Calculate overall impact
        traffic_impact = self._calculate_traffic_impact(metrics)
        power_impact = self._calculate_power_impact(metrics)
        reliability_impact = self._calculate_reliability_impact(metrics)
        
        # Identify issues
        issues = self._identify_issues(metrics)
        
        # Record simulation in history
        sim_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "traffic_model": traffic_model,
            "traffic_impact": traffic_impact,
            "power_impact": power_impact,
            "reliability_impact": reliability_impact,
            "issues_count": len(issues)
        }
        self.simulation_history.append(sim_record)
        
        return {
            "traffic_impact": traffic_impact,
            "power_impact": power_impact,
            "reliability_impact": reliability_impact,
            "performance_metrics": metrics["performance_metrics"],
            "issues": issues
        }
    
    async def predict_failures(self, current_state: Dict[str, Any], timeframe_hours: int) -> Dict[str, Any]:
        """Predict potential failures in the network within given timeframe
        
        Args:
            current_state: Current network state
            timeframe_hours: Time frame for prediction in hours
            
        Returns:
            Prediction results
        """
        logger.info(f"Predicting failures for timeframe {timeframe_hours}h.")
        
        # List to store predicted failures
        predicted_failures = []
        
        # Analyze devices for potential failures
        for device_id, device in current_state.get("devices", {}).items():
            device_age = self._calculate_device_age(device)
            failure_probability = self._calculate_failure_probability(device, device_age, timeframe_hours)
            
            if failure_probability > 0.05:  # Only include significant failure probabilities
                failure_types = self._predict_failure_types(device)
                time_to_failure = self._estimate_time_to_failure(device, failure_types[0], timeframe_hours)
                
                predicted_failures.append({
                    "component_id": device_id,
                    "component_type": "device",
                    "failure_type": failure_types[0],
                    "probability": failure_probability,
                    "time_to_failure_hours": time_to_failure,
                    "potential_causes": self._identify_potential_causes(device, failure_types[0])
                })
        
        # Analyze links for potential failures
        for i, link in enumerate(current_state.get("links", [])):
            link_id = link.get("id", f"link_{i}")
            link_age = self._calculate_link_age(link)
            failure_probability = self._calculate_failure_probability(link, link_age, timeframe_hours)
            
            if failure_probability > 0.05:  # Only include significant failure probabilities
                failure_types = self._predict_failure_types(link)
                time_to_failure = self._estimate_time_to_failure(link, failure_types[0], timeframe_hours)
                
                predicted_failures.append({
                    "component_id": link_id,
                    "component_type": "link",
                    "failure_type": failure_types[0],
                    "probability": failure_probability,
                    "time_to_failure_hours": time_to_failure,
                    "potential_causes": self._identify_potential_causes(link, failure_types[0])
                })
        
        # Analyze transceivers for potential failures
        for i, tx in enumerate(current_state.get("transceivers", [])):
            tx_id = tx.get("id", f"transceiver_{i}")
            tx_age = self._calculate_transceiver_age(tx)
            failure_probability = self._calculate_failure_probability(tx, tx_age, timeframe_hours)
            
            if failure_probability > 0.05:  # Only include significant failure probabilities
                failure_types = self._predict_failure_types(tx)
                time_to_failure = self._estimate_time_to_failure(tx, failure_types[0], timeframe_hours)
                
                predicted_failures.append({
                    "component_id": tx_id,
                    "component_type": "transceiver",
                    "failure_type": failure_types[0],
                    "probability": failure_probability,
                    "time_to_failure_hours": time_to_failure,
                    "potential_causes": self._identify_potential_causes(tx, failure_types[0])
                })
        
        # Calculate potential overall impact
        potential_impact = self._calculate_potential_impact(predicted_failures, current_state)
        
        # Sort failures by probability
        predicted_failures.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "predicted_failures": predicted_failures,
            "potential_impact": potential_impact
        }
    
    async def _simulate_network_behavior(self, state: Dict[str, Any], traffic_matrix: Dict[str, Any], 
                                      duration_seconds: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate network behavior with the given traffic matrix
        
        Args:
            state: Network state to simulate
            traffic_matrix: Traffic matrix for simulation
            duration_seconds: Duration of simulation in seconds
            metrics: Metrics dictionary to update
            
        Returns:
            Updated metrics dictionary
        """
        # Simulate link utilization
        link_utilization = self._calculate_link_utilization(state, traffic_matrix)
        
        # Simulate device utilization
        device_utilization = self._calculate_device_utilization(state, traffic_matrix)
        
        # Calculate network-wide metrics
        total_throughput = sum(link["throughput_gbps"] for link in link_utilization.values())
        avg_link_utilization = sum(link["utilization"] for link in link_utilization.values()) / len(link_utilization) if link_utilization else 0
        congested_links = [link_id for link_id, link in link_utilization.items() if link["utilization"] > 0.9]
        power_consumption = sum(device["power_consumption_watts"] for device in device_utilization.values())
        
        # Calculate latency and packet loss
        latency_data = self._calculate_latency(state, link_utilization, device_utilization)
        packet_loss_data = self._calculate_packet_loss(state, link_utilization)
        
        # Update performance metrics
        metrics["performance_metrics"] = {
            "total_throughput_gbps": total_throughput,
            "avg_link_utilization": avg_link_utilization,
            "congested_links_count": len(congested_links),
            "power_consumption_watts": power_consumption,
            "avg_latency_ms": latency_data["avg_latency_ms"],
            "max_latency_ms": latency_data["max_latency_ms"],
            "packet_loss_rate": packet_loss_data["packet_loss_rate"]
        }
        
        # Store detailed metrics
        metrics["link_metrics"] = link_utilization
        metrics["device_metrics"] = device_utilization
        metrics["latency_metrics"] = latency_data
        metrics["packet_loss_metrics"] = packet_loss_data
        
        # Identify issues
        metrics["issues"] = self._identify_real_time_issues(metrics)
        
        return metrics
    
    def _calculate_link_utilization(self, state: Dict[str, Any], traffic_matrix: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate link utilization based on traffic matrix
        
        Args:
            state: Network state
            traffic_matrix: Traffic matrix
            
        Returns:
            Dictionary of link utilization data
        """
        link_utilization = {}
        
        for i, link in enumerate(state.get("links", [])):
            link_id = link.get("id", f"link_{i}")
            capacity_gbps = link.get("capacity_gbps", 100)  # Default to 100G if not specified
            
            # Get traffic for this link
            link_traffic = traffic_matrix.get("link_traffic", {}).get(link_id, 0)
            
            # Calculate utilization
            utilization = link_traffic / capacity_gbps if capacity_gbps > 0 else 1.0
            
            link_utilization[link_id] = {
                "throughput_gbps": link_traffic,
                "capacity_gbps": capacity_gbps,
                "utilization": utilization,
                "is_congested": utilization > 0.9
            }
        
        return link_utilization
    
    def _calculate_device_utilization(self, state: Dict[str, Any], traffic_matrix: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate device utilization based on traffic matrix
        
        Args:
            state: Network state
            traffic_matrix: Traffic matrix
            
        Returns:
            Dictionary of device utilization data
        """
        device_utilization = {}
        
        for device_id, device in state.get("devices", {}).items():
            # Get device type and capacity
            device_type = device.get("type", "unknown")
            device_capacity_gbps = device.get("switching_capacity_gbps", 1000)  # Default to 1T if not specified
            
            # Get traffic for this device
            device_traffic = traffic_matrix.get("device_traffic", {}).get(device_id, 0)
            
            # Calculate utilization
            utilization = device_traffic / device_capacity_gbps if device_capacity_gbps > 0 else 1.0
            
            # Calculate power consumption based on utilization
            base_power_watts = device.get("base_power_watts", 100)
            max_power_watts = device.get("max_power_watts", 300)
            power_consumption_watts = base_power_watts + (max_power_watts - base_power_watts) * utilization
            
            device_utilization[device_id] = {
                "throughput_gbps": device_traffic,
                "capacity_gbps": device_capacity_gbps,
                "utilization": utilization,
                "power_consumption_watts": power_consumption_watts,
                "is_overloaded": utilization > 0.9
            }
        
        return device_utilization
    
    def _calculate_latency(self, state: Dict[str, Any], link_utilization: Dict[str, Dict[str, Any]], 
                        device_utilization: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate network latency based on utilization
        
        Args:
            state: Network state
            link_utilization: Link utilization data
            device_utilization: Device utilization data
            
        Returns:
            Dictionary of latency metrics
        """
        # Calculate link latencies
        link_latencies = {}
        for link_id, link_data in link_utilization.items():
            # Get link length
            link_obj = next((l for l in state.get("links", []) if l.get("id") == link_id), {})
            link_length_km = link_obj.get("length_km", 1)
            
            # Calculate propagation delay
            propagation_delay_ms = (link_length_km * 1000) / (self.physics_engine.c / 1000)  # ms
            
            # Calculate queueing delay based on utilization
            utilization = link_data["utilization"]
            if utilization < 0.5:
                queueing_delay_ms = 0.01 * utilization
            elif utilization < 0.8:
                queueing_delay_ms = 0.05 * utilization
            elif utilization < 0.95:
                queueing_delay_ms = 0.2 * utilization
            else:
                # Exponential increase for high utilization
                queueing_delay_ms = 0.5 * np.exp(5 * (utilization - 0.95))
            
            # Total link latency
            link_latencies[link_id] = propagation_delay_ms + queueing_delay_ms
        
        # Calculate device latencies
        device_latencies = {}
        for device_id, device_data in device_utilization.items():
            # Base latency depending on device type
            device_obj = state["devices"].get(device_id, {})
            device_type = device_obj.get("type", "switch")
            
            if device_type == "router":
                base_latency_ms = 0.2
            elif device_type == "switch":
                base_latency_ms = 0.05
            else:
                base_latency_ms = 0.1
            
            # Calculate processing delay based on utilization
            utilization = device_data["utilization"]
            if utilization < 0.7:
                processing_delay_ms = base_latency_ms
            elif utilization < 0.9:
                processing_delay_ms = base_latency_ms * (1 + (utilization - 0.7) / 0.2)
            else:
                # Exponential increase for high utilization
                processing_delay_ms = base_latency_ms * (2 + 3 * (utilization - 0.9) / 0.1)
            
            device_latencies[device_id] = processing_delay_ms
        
        # Calculate path latencies using shortest paths
        # Note: In a real implementation, this would use actual routing tables
        # For this placeholder, we assume a simple average of device and link latencies
        
        avg_latency_ms = (sum(link_latencies.values()) / len(link_latencies) if link_latencies else 0) + \
                          (sum(device_latencies.values()) / len(device_latencies) if device_latencies else 0)
        
        max_latency_ms = (max(link_latencies.values()) if link_latencies else 0) + \
                          (max(device_latencies.values()) if device_latencies else 0)
        
        return {
            "avg_latency_ms": avg_latency_ms,
            "max_latency_ms": max_latency_ms,
            "link_latencies_ms": link_latencies,
            "device_latencies_ms": device_latencies
        }
    
    def _calculate_packet_loss(self, state: Dict[str, Any], link_utilization: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate packet loss based on link utilization
        
        Args:
            state: Network state
            link_utilization: Link utilization data
            
        Returns:
            Dictionary of packet loss metrics
        """
        # Calculate packet loss for each link
        link_packet_loss = {}
        for link_id, link_data in link_utilization.items():
            utilization = link_data["utilization"]
            
            # Packet loss model (simplified)
            if utilization < 0.8:
                packet_loss_rate = 0.0
            elif utilization < 0.9:
                packet_loss_rate = 0.0001 * (utilization - 0.8) / 0.1
            elif utilization < 0.95:
                packet_loss_rate = 0.0001 + 0.001 * (utilization - 0.9) / 0.05
            else:
                # Exponential increase for high utilization
                packet_loss_rate = 0.0011 + 0.01 * (utilization - 0.95) / 0.05
            
            link_packet_loss[link_id] = packet_loss_rate
        
        # Calculate overall packet loss
        congested_links = [link_id for link_id, link_data in link_utilization.items() if link_data["utilization"] > 0.9]
        total_packet_loss_rate = sum(link_packet_loss.values())
        
        return {
            "packet_loss_rate": total_packet_loss_rate,
            "link_packet_loss": link_packet_loss,
            "congested_links": congested_links
        }
    
    def _identify_real_time_issues(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify issues in the network based on simulation metrics
        
        Args:
            metrics: Simulation metrics
            
        Returns:
            List of identified issues
        """
        issues = []
        
        # Check for congested links
        for link_id, link_data in metrics["link_metrics"].items():
            if link_data["utilization"] > 0.9:
                issues.append({
                    "type": "congestion",
                    "component_id": link_id,
                    "component_type": "link",
                    "severity": "high" if link_data["utilization"] > 0.95 else "medium",
                    "description": f"Link {link_id} is congested at {link_data['utilization']*100:.1f}% utilization"
                })
        
        # Check for overloaded devices
        for device_id, device_data in metrics["device_metrics"].items():
            if device_data["utilization"] > 0.9:
                issues.append({
                    "type": "high_utilization",
                    "component_id": device_id,
                    "component_type": "device",
                    "severity": "high" if device_data["utilization"] > 0.95 else "medium",
                    "description": f"Device {device_id} is at {device_data['utilization']*100:.1f}% utilization"
                })
        
        # Check for high packet loss
        if metrics["packet_loss_metrics"]["packet_loss_rate"] > 0.001:
            issues.append({
                "type": "packet_loss",
                "component_id": "network",
                "component_type": "network",
                "severity": "high" if metrics["packet_loss_metrics"]["packet_loss_rate"] > 0.01 else "medium",
                "description": f"Network experiencing {metrics['packet_loss_metrics']['packet_loss_rate']*100:.3f}% packet loss"
            })
        
        # Check for high latency
        if metrics["latency_metrics"]["max_latency_ms"] > 10:
            issues.append({
                "type": "high_latency",
                "component_id": "network",
                "component_type": "network",
                "severity": "medium",
                "description": f"Network experiencing high max latency of {metrics['latency_metrics']['max_latency_ms']:.2f} ms"
            })
        
        return issues
    
    # Methods for traffic generation
    async def _generate_uniform_traffic(self, state: Dict[str, Any], duration_seconds: int) -> Dict[str, Any]:
        """Generate uniform traffic matrix
        
        Args:
            state: Network state
            duration_seconds: Duration of simulation in seconds
            
        Returns:
            Traffic matrix
        """
        link_traffic = {}
        device_traffic = {}
        
        # Generate traffic for links
        for i, link in enumerate(state.get("links", [])):
            link_id = link.get("id", f"link_{i}")
            capacity_gbps = link.get("capacity_gbps", 100)
            # Uniform traffic at 70% utilization
            link_traffic[link_id] = capacity_gbps * 0.7
        
        # Generate traffic for devices
        for device_id, device in state.get("devices", {}).items():
            capacity_gbps = device.get("switching_capacity_gbps", 1000)
            # Uniform traffic at 60% utilization
            device_traffic[device_id] = capacity_gbps * 0.6
        
        return {
            "link_traffic": link_traffic,
            "device_traffic": device_traffic,
            "traffic_type": "uniform"
        }
    
    async def _generate_diurnal_traffic(self, state: Dict[str, Any], duration_seconds: int) -> Dict[str, Any]:
        """Generate diurnal (time-of-day) traffic pattern
        
        Args:
            state: Network state
            duration_seconds: Duration of simulation in seconds
            
        Returns:
            Traffic matrix
        """
        # Get current hour (0-23)
        current_hour = datetime.datetime.now().hour
        
        # Define diurnal traffic pattern (hourly multipliers)
        hourly_multipliers = [
            0.2, 0.15, 0.1, 0.1, 0.1, 0.2,  # 0-5 (night)
            0.3, 0.5, 0.7, 0.8, 0.85, 0.9,  # 6-11 (morning)
            0.95, 0.9, 0.85, 0.8, 0.85, 0.9,  # 12-17 (afternoon)
            0.95, 0.9, 0.8, 0.6, 0.4, 0.3  # 18-23 (evening)
        ]
        
        # Get multiplier for current hour
        multiplier = hourly_multipliers[current_hour]
        
        link_traffic = {}
        device_traffic = {}
        
        # Generate traffic for links
        for i, link in enumerate(state.get("links", [])):
            link_id = link.get("id", f"link_{i}")
            capacity_gbps = link.get("capacity_gbps", 100)
            link_traffic[link_id] = capacity_gbps * multiplier
        
        # Generate traffic for devices
        for device_id, device in state.get("devices", {}).items():
            capacity_gbps = device.get("switching_capacity_gbps", 1000)
            device_traffic[device_id] = capacity_gbps * multiplier
        
        return {
            "link_traffic": link_traffic,
            "device_traffic": device_traffic,
            "traffic_type": "diurnal",
            "time_multiplier": multiplier
        }
    
    async def _generate_bursty_traffic(self, state: Dict[str, Any], duration_seconds: int) -> Dict[str, Any]:
        """Generate bursty traffic pattern
        
        Args:
            state: Network state
            duration_seconds: Duration of simulation in seconds
            
        Returns:
            Traffic matrix
        """
        link_traffic = {}
        device_traffic = {}
        
        # Define burst parameters
        burst_probability = 0.3  # 30% chance of a link experiencing a burst
        burst_multiplier = 1.5  # Burst increases traffic by 50%
        
        # Generate traffic for links
        for i, link in enumerate(state.get("links", [])):
            link_id = link.get("id", f"link_{i}")
            capacity_gbps = link.get("capacity_gbps", 100)
            
            # Baseline traffic at 60% utilization
            baseline_traffic = capacity_gbps * 0.6
            
            # Apply burst randomly
            is_burst = np.random.random() < burst_probability
            multiplier = burst_multiplier if is_burst else 1.0
            
            link_traffic[link_id] = baseline_traffic * multiplier
        
        # Generate traffic for devices
        for device_id, device in state.get("devices", {}).items():
            capacity_gbps = device.get("switching_capacity_gbps", 1000)
            
            # Baseline traffic at 50% utilization
            baseline_traffic = capacity_gbps * 0.5
            
            # Apply burst randomly
            is_burst = np.random.random() < burst_probability
            multiplier = burst_multiplier if is_burst else 1.0
            
            device_traffic[device_id] = baseline_traffic * multiplier
        
        return {
            "link_traffic": link_traffic,
            "device_traffic": device_traffic,
            "traffic_type": "bursty",
            "burst_links": [link_id for link_id in link_traffic if link_traffic[link_id] > 0.9]
        }
    
    async def _generate_hotspot_traffic(self, state: Dict[str, Any], duration_seconds: int) -> Dict[str, Any]:
        """Generate hotspot traffic pattern (concentrated on specific nodes/links)
        
        Args:
            state: Network state
            duration_seconds: Duration of simulation in seconds
            
        Returns:
            Traffic matrix
        """
        link_traffic = {}
        device_traffic = {}
        
        # Choose hotspot nodes (random selection of 20% of devices)
        devices = list(state.get("devices", {}).keys())
        hotspot_count = max(1, int(len(devices) * 0.2))
        hotspot_devices = np.random.choice(devices, hotspot_count, replace=False) if devices else []
        
        # Generate traffic for links
        for i, link in enumerate(state.get("links", [])):
            link_id = link.get("id", f"link_{i}")
            capacity_gbps = link.get("capacity_gbps", 100)
            
            # Check if this link connects to a hotspot device
            source_device = link.get("source", "")
            target_device = link.get("target", "")
            
            is_hotspot_link = source_device in hotspot_devices or target_device in hotspot_devices
            
            # Assign traffic based on hotspot status
            if is_hotspot_link:
                # High traffic for hotspot links (90% utilization)
                link_traffic[link_id] = capacity_gbps * 0.9
            else:
                # Normal traffic for non-hotspot links (50% utilization)
                link_traffic[link_id] = capacity_gbps * 0.5
        
        # Generate traffic for devices
        for device_id, device in state.get("devices", {}).items():
            capacity_gbps = device.get("switching_capacity_gbps", 1000)
            
            if device_id in hotspot_devices:
                # High traffic for hotspot devices (85% utilization)
                device_traffic[device_id] = capacity_gbps * 0.85
            else:
                # Normal traffic for non-hotspot devices (45% utilization)
                device_traffic[device_id] = capacity_gbps * 0.45
        
        return {
            "link_traffic": link_traffic,
            "device_traffic": device_traffic,
            "traffic_type": "hotspot",
            "hotspot_devices": list(hotspot_devices)
        }
    
    async def _use_current_traffic(self, state: Dict[str, Any], duration_seconds: int) -> Dict[str, Any]:
        """Use current traffic from state (if available)
        
        Args:
            state: Network state
            duration_seconds: Duration of simulation in seconds
            
        Returns:
            Traffic matrix
        """
        # Check if state contains traffic information
        if "telemetry" in state and "traffic" in state["telemetry"]:
            return state["telemetry"]["traffic"]
        
        # If no traffic information, fall back to uniform traffic
        return await self._generate_uniform_traffic(state, duration_seconds)

    def _calculate_traffic_impact(self, metrics: Dict[str, Any]) -> str:
        """Calculate the traffic impact based on simulation metrics
        
        Args:
            metrics: Simulation metrics
            
        Returns:
            Impact level (none, low, medium, high, critical)
        """
        # Extract metrics
        avg_link_utilization = metrics["performance_metrics"].get("avg_link_utilization", 0)
        congested_links_count = metrics["performance_metrics"].get("congested_links_count", 0)
        packet_loss_rate = metrics["performance_metrics"].get("packet_loss_rate", 0)
        
        # Determine impact based on metrics
        if packet_loss_rate > 0.01:
            return "critical"
        elif congested_links_count > 0 and packet_loss_rate > 0.001:
            return "high"
        elif avg_link_utilization > 0.8:
            return "medium"
        elif avg_link_utilization > 0.6:
            return "low"
        else:
            return "none"
    
    def _calculate_power_impact(self, metrics: Dict[str, Any]) -> str:
        """Calculate the power impact based on simulation metrics
        
        Args:
            metrics: Simulation metrics
            
        Returns:
            Impact level (positive, neutral, negative, high)
        """
        # Extract metrics
        power_consumption = metrics["performance_metrics"].get("power_consumption_watts", 0)
        total_throughput = metrics["performance_metrics"].get("total_throughput_gbps", 0)
        
        # Calculate power efficiency (Gbps/Watt)
        power_efficiency = total_throughput / power_consumption if power_consumption > 0 else 0
        
        # Determine impact based on efficiency
        if power_efficiency > 0.5:
            return "positive"
        elif power_efficiency > 0.3:
            return "neutral"
        elif power_efficiency > 0.1:
            return "negative"
        else:
            return "high"
    
    def _calculate_reliability_impact(self, metrics: Dict[str, Any]) -> str:
        """Calculate the reliability impact based on simulation metrics
        
        Args:
            metrics: Simulation metrics
            
        Returns:
            Impact level (none, low, medium, high)
        """
        # Extract metrics
        issues = metrics.get("issues", [])
        high_severity_issues = [issue for issue in issues if issue.get("severity") == "high"]
        
        # Determine impact based on issues
        if len(high_severity_issues) > 3:
            return "high"
        elif len(high_severity_issues) > 0:
            return "medium"
        elif len(issues) > 0:
            return "low"
        else:
            return "none"
    
    def _identify_issues(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify issues based on simulation metrics
        
        Args:
            metrics: Simulation metrics
            
        Returns:
            List of identified issues
        """
        # Get issues identified during real-time simulation
        return metrics.get("issues", [])
    
    def _calculate_device_age(self, device: Dict[str, Any]) -> int:
        """Calculate the age of a device in days
        
        Args:
            device: Device information
            
        Returns:
            Age in days
        """
        # Get installation date
        installation_date = device.get("installation_date")
        if not installation_date:
            return 0
        
        # Parse date and calculate age
        try:
            install_dt = datetime.datetime.fromisoformat(installation_date)
            now = datetime.datetime.now()
            age_days = (now - install_dt).days
            return max(0, age_days)
        except (ValueError, TypeError):
            return 0
    
    def _calculate_link_age(self, link: Dict[str, Any]) -> int:
        """Calculate the age of a link in days
        
        Args:
            link: Link information
            
        Returns:
            Age in days
        """
        # Get installation date
        installation_date = link.get("installation_date")
        if not installation_date:
            return 0
        
        # Parse date and calculate age
        try:
            install_dt = datetime.datetime.fromisoformat(installation_date)
            now = datetime.datetime.now()
            age_days = (now - install_dt).days
            return max(0, age_days)
        except (ValueError, TypeError):
            return 0
    
    def _calculate_transceiver_age(self, tx: Dict[str, Any]) -> int:
        """Calculate the age of a transceiver in days
        
        Args:
            tx: Transceiver information
            
        Returns:
            Age in days
        """
        # Get installation date
        installation_date = tx.get("installation_date")
        if not installation_date:
            return 0
        
        # Parse date and calculate age
        try:
            install_dt = datetime.datetime.fromisoformat(installation_date)
            now = datetime.datetime.now()
            age_days = (now - install_dt).days
            return max(0, age_days)
        except (ValueError, TypeError):
            return 0
    
    def _calculate_failure_probability(self, component: Dict[str, Any], age_days: int, timeframe_hours: int) -> float:
        """Calculate the failure probability for a component
        
        Args:
            component: Component information
            age_days: Age of the component in days
            timeframe_hours: Time frame for prediction in hours
            
        Returns:
            Failure probability (0.0 to 1.0)
        """
        # Get component type
        component_type = component.get("type", "unknown")
        
        # Get MTBF (Mean Time Between Failures) in days
        mtbf_days = component.get("mtbf_days")
        if not mtbf_days:
            # Use default MTBF based on component type
            if component_type == "transceiver":
                mtbf_days = 365 * 5  # 5 years
            elif component_type == "switch":
                mtbf_days = 365 * 7  # 7 years
            elif component_type == "router":
                mtbf_days = 365 * 6  # 6 years
            elif component_type == "fiber":
                mtbf_days = 365 * 10  # 10 years
            else:
                mtbf_days = 365 * 5  # Default 5 years
        
        # Apply Weibull distribution for failure probability calculation
        shape = 1.5  # Weibull shape parameter (>1 indicates increasing failure rate with age)
        scale = mtbf_days / 0.886  # Scale parameter (adjusts the MTBF to the Weibull distribution)
        
        # Calculate probability based on current age
        failure_prob_per_day = (shape / scale) * ((age_days / scale) ** (shape - 1)) * np.exp(-((age_days / scale) ** shape))
        
        # Convert to probability for the timeframe
        timeframe_days = timeframe_hours / 24
        failure_probability = 1 - np.exp(-(timeframe_days * failure_prob_per_day))
        
        # Get any known issues
        known_issues = component.get("known_issues", [])
        if known_issues:
            # Increase probability for components with known issues
            failure_probability = min(0.95, failure_probability * 1.5)
        
        # Get utilization if available
        utilization = component.get("utilization", 0.5)
        if utilization > 0.8:
            # Increase probability for heavily utilized components
            failure_probability = min(0.95, failure_probability * 1.3)
        
        return failure_probability
    
    def _predict_failure_types(self, component: Dict[str, Any]) -> List[str]:
        """Predict potential failure types for a component
        
        Args:
            component: Component information
            
        Returns:
            List of potential failure types, ordered by probability
        """
        # Get component type
        component_type = component.get("type", "unknown")
        
        # Different failure types based on component type
        if component_type == "transceiver":
            failure_types = ["laser_degradation", "receiver_sensitivity_drift", "thermal_failure", "connector_issue"]
        elif component_type == "switch" or component_type == "router":
            failure_types = ["power_supply", "fan_failure", "memory_error", "software_issue", "port_failure"]
        elif component_type == "fiber" or component_type == "link":
            failure_types = ["physical_break", "attenuation_increase", "connector_issue", "splice_degradation"]
        else:
            failure_types = ["hardware_failure", "software_issue", "power_supply", "environmental_issue"]
        
        # Get known issues
        known_issues = component.get("known_issues", [])
        
        # Move any known issue types to the front of the list
        for issue in known_issues:
            issue_type = issue.get("type")
            if issue_type in failure_types:
                failure_types.remove(issue_type)
                failure_types.insert(0, issue_type)
        
        return failure_types
    
    def _estimate_time_to_failure(self, component: Dict[str, Any], failure_type: str, timeframe_hours: int) -> float:
        """Estimate time to failure for a specific failure type
        
        Args:
            component: Component information
            failure_type: Type of failure
            timeframe_hours: Time frame for prediction in hours
            
        Returns:
            Estimated time to failure in hours
        """
        # Get component age
        age_days = self._calculate_device_age(component)
        
        # Random time within the timeframe, weighted towards the end for older components
        age_ratio = min(1.0, age_days / 365)  # Age as a ratio of a year
        
        # Adjust time based on failure type
        if failure_type == "power_supply":
            # Power supply failures tend to happen suddenly
            ttf_ratio = np.random.beta(1, 2)  # More weight towards the beginning
        elif failure_type == "laser_degradation":
            # Laser degradation is gradual
            ttf_ratio = np.random.beta(2, 1)  # More weight towards the end
        else:
            # Other failures have more uniform distribution
            ttf_ratio = np.random.random()
        
        # Mix the age ratio with the failure type ratio
        final_ratio = 0.7 * ttf_ratio + 0.3 * age_ratio
        
        # Calculate time to failure
        time_to_failure = timeframe_hours * final_ratio
        
        return time_to_failure
    
    def _identify_potential_causes(self, component: Dict[str, Any], failure_type: str) -> List[str]:
        """Identify potential causes for a specific failure type
        
        Args:
            component: Component information
            failure_type: Type of failure
            
        Returns:
            List of potential causes
        """
        # Common causes by failure type
        causes = {
            "power_supply": [
                "End of life for power supply unit",
                "Voltage fluctuations",
                "Overheating due to poor ventilation",
                "Component degradation"
            ],
            "fan_failure": [
                "Dust accumulation",
                "Bearing wear",
                "Motor failure",
                "Obstruction"
            ],
            "memory_error": [
                "Hardware degradation",
                "Manufacturing defect",
                "Memory leak in software",
                "Overheating"
            ],
            "laser_degradation": [
                "End of life for laser diode",
                "Operating at high temperature",
                "Operating at high current",
                "Manufacturing defect"
            ],
            "receiver_sensitivity_drift": [
                "Aging of photodetector",
                "Temperature variations",
                "Circuit degradation",
                "Signal chain noise increase"
            ],
            "physical_break": [
                "External damage (construction, rodents)",
                "Fiber stress due to improper installation",
                "Environmental factors (freezing, heat)",
                "Connector damage during maintenance"
            ],
            "attenuation_increase": [
                "Bend radius violation",
                "Water penetration",
                "Aging of fiber material",
                "Connector contamination"
            ],
            "connector_issue": [
                "Dust or contamination",
                "Physical damage",
                "Improper seating",
                "Excessive insertion/removal cycles"
            ]
        }
        
        # Get causes for the specific failure type, or default
        failure_causes = causes.get(failure_type, [
            "Component aging",
            "Environmental factors",
            "Manufacturing defect",
            "Operational stress"
        ])
        
        # Get component-specific information that might influence causes
        environment = component.get("environment", {})
        temp = environment.get("temperature_c", 25)
        humidity = environment.get("humidity_percent", 50)
        
        # Add environment-specific causes
        if temp > 30:
            failure_causes.append("High operating temperature")
        if humidity > 70:
            failure_causes.append("High humidity environment")
        
        return failure_causes
    
    def _calculate_potential_impact(self, predicted_failures: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
        """Calculate the potential impact of predicted failures
        
        Args:
            predicted_failures: List of predicted failures
            state: Current network state
            
        Returns:
            Impact level (low, medium, high, critical)
        """
        # If no failures, impact is low
        if not predicted_failures:
            return "low"
        
        # Count high probability failures
        high_prob_failures = len([f for f in predicted_failures if f["probability"] > 0.7])
        
        # Check for critical components
        critical_component_failures = []
        for failure in predicted_failures:
            component_id = failure["component_id"]
            component_type = failure["component_type"]
            
            if component_type == "device":
                device = state["devices"].get(component_id, {})
                if device.get("is_critical", False) or device.get("role") in ["core", "border", "gateway"]:
                    critical_component_failures.append(failure)
        
        # Determine impact based on failure analysis
        if critical_component_failures and high_prob_failures:
            return "critical"
        elif critical_component_failures:
            return "high"
        elif high_prob_failures:
            return "medium"
        else:
            return "low"

class NetworkDigitalTwin:
    """Complete digital replica of the physical network for simulation and training"""
    
    def __init__(self, network_inventory, telemetry_service, topology_service):
        self.network_inventory = network_inventory
        self.telemetry_service = telemetry_service
        self.topology_service = topology_service
        self.twin_state = {}
        self.simulation_engine = None
        self.sync_time = None
        self.physics_engine = OpticalPhysicsEngine()
    
    async def initialize_twin(self):
        """Initialize the digital twin from current network state"""
        # Get physical inventory
        devices = await self.network_inventory.get_all_devices()
        links = await self.network_inventory.get_all_links()
        transceivers = await self.network_inventory.get_all_transceivers()
        
        # Get current topology
        topology = await self.topology_service.get_current_topology()
        
        # Get current telemetry
        telemetry = await self.telemetry_service.get_current_telemetry()
        
        # Initialize twin state
        self.twin_state = {
            "devices": devices,
            "links": links,
            "transceivers": transceivers,
            "topology": topology,
            "telemetry": telemetry,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Initialize simulation engine with twin state
        self.simulation_engine = NetworkSimulationEngine(
            initial_state=self.twin_state,
            physics_engine=self.physics_engine
        )
        
        self.sync_time = datetime.datetime.now()
        
        return {
            "status": "initialized",
            "devices_count": len(devices),
            "links_count": len(links),
            "transceivers_count": len(transceivers)
        }
    
    async def sync_with_physical_network(self):
        """Synchronize digital twin with current physical network state"""
        # Get latest telemetry
        telemetry = await self.telemetry_service.get_current_telemetry()
        
        # Get topology changes
        topology_changes = await self.topology_service.get_topology_changes(since=self.sync_time)
        
        # Get inventory changes
        inventory_changes = await self.network_inventory.get_changes(since=self.sync_time)
        
        # Update twin state
        self.twin_state["telemetry"] = telemetry
        
        if topology_changes:
            self.twin_state["topology"] = await self.topology_service.get_current_topology()
        
        if inventory_changes.get("devices", []):
            # Update changed devices
            for device_id in inventory_changes["devices"]:
                device_data = await self.network_inventory.get_device(device_id)
                self._update_device_in_twin(device_id, device_data)
        
        if inventory_changes.get("links", []):
            # Update changed links
            for link_id in inventory_changes["links"]:
                link_data = await self.network_inventory.get_link(link_id)
                self._update_link_in_twin(link_id, link_data)
        
        if inventory_changes.get("transceivers", []):
            # Update changed transceivers
            for tx_id in inventory_changes["transceivers"]:
                tx_data = await self.network_inventory.get_transceiver(tx_id)
                self._update_transceiver_in_twin(tx_id, tx_data)
        
        # Update simulation engine with new state
        self.simulation_engine.update_state(self.twin_state)
        
        self.sync_time = datetime.datetime.now()
        self.twin_state["timestamp"] = self.sync_time.isoformat()
        
        return {
            "status": "synchronized",
            "telemetry_updated": True,
            "topology_updated": bool(topology_changes),
            "inventory_updated": bool(inventory_changes)
        }
    
    async def simulate_change(self, change_description):
        """Simulate a change to the network without affecting the physical network"""
        # Create a copy of the current twin state for simulation
        simulation_state = copy.deepcopy(self.twin_state)
        
        # Apply the described changes to the simulation state
        await self._apply_changes_to_state(simulation_state, change_description)
        
        # Run simulation
        simulation_result = await self.simulation_engine.simulate(
            state=simulation_state,
            duration_seconds=3600,  # Simulate 1 hour of operation
            traffic_model="current"  # Use current traffic patterns
        )
        
        return {
            "pre_change_state": {
                "device_count": len(self.twin_state["devices"]),
                "link_count": len(self.twin_state["links"]),
                "metrics": self._extract_key_metrics(self.twin_state)
            },
            "post_change_state": {
                "device_count": len(simulation_state["devices"]),
                "link_count": len(simulation_state["links"]),
                "metrics": self._extract_key_metrics(simulation_state)
            },
            "simulation_results": {
                "traffic_impact": simulation_result["traffic_impact"],
                "power_impact": simulation_result["power_impact"],
                "reliability_impact": simulation_result["reliability_impact"],
                "performance_metrics": simulation_result["performance_metrics"],
                "issues_detected": simulation_result["issues"]
            },
            "recommendation": self._generate_recommendation(simulation_result)
        }
    
    async def predict_failures(self, timeframe_hours=24):
        """Predict potential failures in the network within given timeframe"""
        # Run predictive simulation
        prediction_result = await self.simulation_engine.predict_failures(
            current_state=self.twin_state,
            timeframe_hours=timeframe_hours
        )
        
        # Analyze component health trends
        health_trends = await self._analyze_component_health_trends()
        
        # Combine prediction results with health trends
        failure_predictions = []
        
        for prediction in prediction_result["predicted_failures"]:
            # Enrich prediction with health trend data if available
            component_id = prediction["component_id"]
            if component_id in health_trends:
                prediction["health_trend"] = health_trends[component_id]
            
            failure_predictions.append(prediction)
        
        # Sort by probability (descending)
        failure_predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "failure_predictions": failure_predictions,
            "total_predictions": len(failure_predictions),
            "high_risk_count": len([p for p in failure_predictions if p["probability"] > 0.7]),
            "medium_risk_count": len([p for p in failure_predictions if 0.3 < p["probability"] <= 0.7]),
            "low_risk_count": len([p for p in failure_predictions if p["probability"] <= 0.3]),
            "potential_impact": prediction_result["potential_impact"]
        }
    
    async def train_operators(self, scenario_type="failure_recovery"):
        """Create training scenarios for network operators"""
        # Generate training scenario
        scenario = await self._generate_training_scenario(scenario_type)
        
        # Create a copy of twin for training
        training_twin = copy.deepcopy(self.twin_state)
        
        # Apply scenario events to the training twin
        for event in scenario["events"]:
            await self._apply_event_to_state(training_twin, event)
        
        return {
            "scenario_id": str(uuid.uuid4()),
            "scenario_type": scenario_type,
            "scenario_description": scenario["description"],
            "difficulty_level": scenario["difficulty"],
            "events": scenario["events"],
            "training_state": training_twin,
            "objectives": scenario["objectives"],
            "success_criteria": scenario["success_criteria"],
            "time_limit_minutes": scenario["time_limit_minutes"]
        }
    
    def _update_device_in_twin(self, device_id: str, device_data: Dict[str, Any]):
        logger.info(f"Updating device {device_id} in twin with data: {device_data}")
        if "devices" not in self.twin_state:
            self.twin_state["devices"] = {}
        self.twin_state["devices"][device_id] = device_data
        # Add more sophisticated update logic if needed

    def _update_link_in_twin(self, link_id: str, link_data: Dict[str, Any]):
        logger.info(f"Updating link {link_id} in twin with data: {link_data}")
        # Assuming links are a list of dicts, find and update or append
        if "links" not in self.twin_state:
            self.twin_state["links"] = []
        
        found = False
        for i, link in enumerate(self.twin_state["links"]):
            if link.get("id") == link_id: # Assuming links have an 'id' field
                self.twin_state["links"][i] = link_data
                found = True
                break
        if not found:
            self.twin_state["links"].append(link_data)

    def _update_transceiver_in_twin(self, tx_id: str, tx_data: Dict[str, Any]):
        logger.info(f"Updating transceiver {tx_id} in twin with data: {tx_data}")
        if "transceivers" not in self.twin_state:
            self.twin_state["transceivers"] = [] # Or Dict if identified by tx_id key
        # Assuming transceivers are a list of dicts, find and update or append
        found = False
        for i, tx in enumerate(self.twin_state["transceivers"]):
            if tx.get("id") == tx_id: # Assuming transceivers have an 'id' field
                self.twin_state["transceivers"][i] = tx_data
                found = True
                break
        if not found:
            self.twin_state["transceivers"].append(tx_data)

    async def _apply_changes_to_state(self, simulation_state: Dict[str, Any], change_description: Dict[str, Any]):
        logger.info(f"Applying changes to simulation state: {change_description}")
        # Placeholder: Actual logic to parse change_description and modify simulation_state
        # Example: if change_description.get('type') == 'add_device':
        # simulation_state['devices'][change_description['device_id']] = change_description['device_data']
        pass

    def _generate_recommendation(self, simulation_result: Dict[str, Any]) -> str:
        logger.info(f"Generating recommendation based on simulation result: {simulation_result}")
        if simulation_result.get("issues"):
            return "Consider addressing detected issues. Specific recommendations require deeper analysis."
        return "Simulation indicates the change is acceptable. Monitor post-implementation."

    async def _analyze_component_health_trends(self) -> Dict[str, Any]:
        logger.info("Analyzing component health trends.")
        # Placeholder: Actual logic to analyze telemetry or historical data for trends
        return {
            "device123": {"trend": "degrading_performance", "metric": "error_rate", "value": 0.05}
        }

    async def _generate_training_scenario(self, scenario_type: str) -> Dict[str, Any]:
        logger.info(f"Generating training scenario for type: {scenario_type}")
        # Placeholder: Actual logic to create diverse and realistic scenarios
        return {
            "description": f"Simulated {scenario_type} scenario for training.",
            "difficulty": "medium",
            "events": [{"type": "link_failure", "component_id": "link_AB", "timestamp_offset_sec": 60}],
            "objectives": ["Restore connectivity", "Identify root cause"],
            "success_criteria": "Connectivity restored within 10 minutes.",
            "time_limit_minutes": 30
        }

    async def _apply_event_to_state(self, training_state: Dict[str, Any], event: Dict[str, Any]):
        logger.info(f"Applying event to training state: {event}")
        # Placeholder: Actual logic to modify training_state based on event
        # Example: if event.get('type') == 'link_failure':
        #    for link in training_state['links']:
        #        if link['id'] == event['component_id']:
        #            link['status'] = 'down'
        pass

    def _extract_key_metrics(self, state):
        """Extract key performance metrics from network state"""
        # Calculate average utilization
        link_utilization = [link["utilization"] for link in state.get("links", []) if "utilization" in link]
        avg_utilization = sum(link_utilization) / len(link_utilization) if link_utilization else 0
        
        # Calculate power consumption
        power_consumption = sum(device.get("power_watts", 0) for device in state.get("devices", {}).values())
        
        # Calculate reliability score
        reliability_components = [
            component.get("reliability_score", 1.0) 
            for component_list_key in ["devices", "links", "transceivers"]
            for component in (state.get(component_list_key, {}).values() if isinstance(state.get(component_list_key), dict) else state.get(component_list_key, []))
        ]
        reliability_score = np.prod(reliability_components) if reliability_components else 1.0
        
        return {
            "average_link_utilization": avg_utilization,
            "power_consumption_watts": power_consumption,
            "reliability_score": reliability_score,
            "total_bandwidth_gbps": sum(link.get("capacity_gbps", 0) for link in state.get("links", [])),
            "active_ports": sum(1 for device in state.get("devices", {}).values() 
                              for port in device.get("ports", []) if port.get("status") == "up")
        }